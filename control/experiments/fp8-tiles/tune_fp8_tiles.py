#!/usr/bin/env python3
"""Tune compiled SM120 block-scaled FP8 swapAB tile-N tactics.

The coordinator reads exact two-dimensional block-FP8 weight shapes from a
Qwen safetensors checkpoint, derives the target verification M values from the
DFlash proposal length and batch sizes, and launches one subprocess per
compiled tile. Each worker sets the C++ dispatch override before importing
vLLM, so tile 8, 16, and 32 exercise distinct template instantiations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

TILES = (8, 16, 32)
SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale")
FORCE_ENV = "VLLM_SM120_BLOCKWISE_FP8_SWAPAB_TILE_N"


def csv_ints(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item})
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def resolve_model(model: str, revision: str | None) -> Path:
    candidate = Path(model)
    if candidate.is_dir():
        return candidate.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            local_files_only=True,
        )
    )


def load_weight_map(model_dir: Path) -> dict[str, str]:
    indexes = sorted(model_dir.glob("*.safetensors.index.json"))
    if len(indexes) == 1:
        payload = json.loads(indexes[0].read_text())
        return dict(payload["weight_map"])
    if indexes:
        raise RuntimeError(f"expected one safetensors index, found {len(indexes)}")
    files = sorted(model_dir.glob("*.safetensors"))
    if len(files) != 1:
        raise RuntimeError("checkpoint needs one safetensors file or one index")
    from safetensors import safe_open

    with safe_open(str(files[0]), framework="pt", device="cpu") as handle:
        return {key: files[0].name for key in handle.keys()}


def tensor_shapes(model_dir: Path, weight_map: dict[str, str]) -> dict[str, tuple[int, ...]]:
    from safetensors import safe_open

    by_file: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        by_file.setdefault(filename, []).append(name)
    shapes: dict[str, tuple[int, ...]] = {}
    for filename, names in sorted(by_file.items()):
        with safe_open(str(model_dir / filename), framework="pt", device="cpu") as handle:
            for name in names:
                shapes[name] = tuple(handle.get_slice(name).get_shape())
    return shapes


def qwen_packed_weight_name(weight_name: str) -> str:
    """Map checkpoint shards to the fused TP=1 linear used by vLLM."""
    replacements = (
        (".q_proj.weight", ".qkv_proj.weight"),
        (".k_proj.weight", ".qkv_proj.weight"),
        (".v_proj.weight", ".qkv_proj.weight"),
        (".gate_proj.weight", ".gate_up_proj.weight"),
        (".up_proj.weight", ".gate_up_proj.weight"),
        (".in_proj_qkv.weight", ".in_proj_qkvz.weight"),
        (".in_proj_z.weight", ".in_proj_qkvz.weight"),
        (".in_proj_b.weight", ".in_proj_ba.weight"),
        (".in_proj_a.weight", ".in_proj_ba.weight"),
    )
    for source, destination in replacements:
        if source in weight_name:
            return weight_name.replace(source, destination)
    return weight_name


def discover_block_fp8_shapes(
    model_dir: Path,
    include_experts: bool,
) -> list[dict[str, Any]]:
    weight_map = load_weight_map(model_dir)
    shapes = tensor_shapes(model_dir, weight_map)
    packed: dict[str, dict[str, Any]] = {}
    for scale_name, scale_shape in shapes.items():
        suffix = next((suffix for suffix in SCALE_SUFFIXES if scale_name.endswith(suffix)), None)
        if suffix is None or len(scale_shape) != 2:
            continue
        weight_name = scale_name[: -len(suffix)] + ".weight"
        weight_shape = shapes.get(weight_name)
        if weight_shape is None or len(weight_shape) != 2:
            continue
        if not include_experts and ".experts." in weight_name:
            # Per-expert tensors are packed and dispatched through grouped MoE,
            # not cutlass_scaled_mm_blockwise_sm120_fp8.
            continue
        n, k = weight_shape
        expected_scale = (math.ceil(n / 128), math.ceil(k / 128))
        if tuple(scale_shape) != expected_scale:
            continue
        packed_name = qwen_packed_weight_name(weight_name)
        entry = packed.setdefault(
            packed_name,
            {"n": 0, "k": k, "components": []},
        )
        if entry["k"] != k:
            raise RuntimeError(f"incompatible packed K dimensions for {packed_name}")
        entry["n"] += n
        entry["components"].append(weight_name)
    grouped: dict[tuple[int, int], list[str]] = {}
    for packed_name, shape in packed.items():
        grouped.setdefault((shape["n"], shape["k"]), []).append(packed_name)
    if not grouped:
        raise RuntimeError(
            "no 2-D 128x128 block-FP8 linear weights found; this experiment "
            "does not apply to tensorwise FP8 or NVFP4 checkpoints"
        )
    return [
        {
            "n": n,
            "k": k,
            "occurrences": len(names),
            "weights": sorted(names),
        }
        for (n, k), names in sorted(grouped.items())
    ]


def make_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.tensor_parallel_size != 1:
        raise RuntimeError(
            "the Qwen packing map is exact only at tensor parallel size 1"
        )
    model_dir = resolve_model(args.model, args.revision)
    nk_shapes = discover_block_fp8_shapes(model_dir, args.include_experts)
    tokens_per_verification = args.num_speculative_tokens + 1
    m_values = sorted({batch * tokens_per_verification for batch in args.batch_sizes})
    if args.extra_m_values:
        m_values = sorted(set(m_values) | set(args.extra_m_values))
    cases = []
    for shape in nk_shapes:
        for m in m_values:
            if m <= 64 or m % 4 != 0:
                cases.append(
                    {
                        "m": m,
                        "n": shape["n"],
                        "k": shape["k"],
                        "occurrences": shape["occurrences"],
                    }
                )
    if not cases:
        raise RuntimeError("no requested M values enter the swapAB dispatcher")
    return {
        "model": args.model,
        "revision": args.revision,
        "resolved_model_dir": str(model_dir),
        "num_speculative_tokens": args.num_speculative_tokens,
        "tokens_per_verification": tokens_per_verification,
        "batch_sizes": args.batch_sizes,
        "m_values": m_values,
        "tensor_parallel_size": 1,
        "include_experts": args.include_experts,
        "shapes": nk_shapes,
        "cases": cases,
    }


def invoke_cutlass(a: Any, b: Any, scale_a: Any, scale_b: Any, torch: Any) -> Any:
    from vllm import _custom_ops as ops

    return ops.cutlass_scaled_mm(
        a,
        b,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.bfloat16,
    )


def benchmark_worker(args: argparse.Namespace) -> None:
    os.environ[FORCE_ENV] = str(args.tile_n)
    import torch
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmark workers")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 12:
        raise RuntimeError(f"SM120-family GPU required, found capability {capability}")
    manifest = json.loads(Path(args.manifest).read_text())
    results = []
    for case in manifest["cases"]:
        m, n, k = case["m"], case["n"], case["k"]
        record: dict[str, Any] = {"m": m, "n": n, "k": k, "tile_n": args.tile_n}
        x = a = scale_a = weight = weight_scale = b = scale_b = output = None
        try:
            torch.manual_seed((m * 1000003 + n * 1009 + k) & 0x7FFFFFFF)
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.125
            a, scale_a = per_token_group_quant_fp8(
                x,
                group_size=128,
                column_major_scales=True,
                dtype=torch.float8_e4m3fn,
                use_ue8m0=False,
            )
            weight = (
                torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.125
            ).to(torch.float8_e4m3fn)
            weight_scale = torch.ones(
                (math.ceil(n / 128), math.ceil(k / 128)),
                device="cuda",
                dtype=torch.float32,
            )
            b = weight.T
            scale_b = weight_scale.T
            output = invoke_cutlass(a, b, scale_a, scale_b, torch)
            torch.cuda.synchronize()
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError("kernel returned non-finite output")
            for _ in range(args.warmup):
                output = invoke_cutlass(a, b, scale_a, scale_b, torch)
            torch.cuda.synchronize()
            samples_ms = []
            for _ in range(args.repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    output = invoke_cutlass(a, b, scale_a, scale_b, torch)
                end.record()
                end.synchronize()
                samples_ms.append(start.elapsed_time(end) / args.iterations)
            flat = output.flatten()
            sample_count = min(args.correctness_samples, flat.numel())
            indices = torch.linspace(
                0,
                flat.numel() - 1,
                sample_count,
                device=flat.device,
                dtype=torch.int64,
            )
            record.update(
                median_ms=statistics.median(samples_ms),
                samples_ms=samples_ms,
                output_samples=flat[indices].float().cpu().tolist(),
                output_absmax=float(output.float().abs().max().cpu()),
            )
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        results.append(record)
        del x, a, scale_a, weight, weight_scale, b, scale_b, output
        torch.cuda.empty_cache()
    Path(args.worker_output).write_text(
        json.dumps({"tile_n": args.tile_n, "results": results}, indent=2) + "\n"
    )


def result_key(result: dict[str, Any]) -> tuple[int, int, int]:
    return result["m"], result["n"], result["k"]


def validate_against_32(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    rtol: float,
    atol: float,
) -> bool:
    if "error" in candidate or "error" in baseline:
        return False
    lhs = candidate["output_samples"]
    rhs = baseline["output_samples"]
    return all(abs(a - b) <= atol + rtol * abs(b) for a, b in zip(lhs, rhs))


def render_header(tactics: list[dict[str, int]], manifest: dict[str, Any]) -> str:
    rows = "\n".join(
        f"    Sm120BlockwiseFp8Tactic{{{row['m']}, {row['n']}, {row['k']}, {row['tile_n']}}},"
        for row in tactics
    )
    return f"""#pragma once

#include <array>
#include <cstdint>

namespace vllm {{

struct Sm120BlockwiseFp8Tactic {{
  int32_t m;
  int32_t n;
  int32_t k;
  int32_t tile_n;
}};

// Generated from exact {manifest['model']} DFlash verification shapes.
inline constexpr std::array<Sm120BlockwiseFp8Tactic, {len(tactics)}>
    kSm120BlockwiseFp8SwapABTactics = {{{{
{rows}
}}}};

}}  // namespace vllm
"""


def coordinate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(args)
    manifest_path = output_dir / "shape_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.manifest_only:
        return
    worker_files = []
    for tile_n in TILES:
        worker_output = output_dir / f"tile_{tile_n}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--manifest",
            str(manifest_path),
            "--tile-n",
            str(tile_n),
            "--worker-output",
            str(worker_output),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--iterations",
            str(args.iterations),
            "--correctness-samples",
            str(args.correctness_samples),
        ]
        subprocess.run(command, check=True)
        worker_files.append(worker_output)
    by_tile: dict[int, dict[tuple[int, int, int], dict[str, Any]]] = {}
    for worker_file in worker_files:
        payload = json.loads(worker_file.read_text())
        by_tile[payload["tile_n"]] = {
            result_key(result): result for result in payload["results"]
        }
    tactics = []
    summary = []
    for case in manifest["cases"]:
        key = case["m"], case["n"], case["k"]
        baseline = by_tile[32][key]
        if "error" in baseline:
            raise RuntimeError(f"tile 32 baseline failed for {key}: {baseline['error']}")
        valid = [
            by_tile[tile][key]
            for tile in TILES
            if validate_against_32(by_tile[tile][key], baseline, args.rtol, args.atol)
        ]
        if not valid:
            raise RuntimeError(f"no correctness-valid tactic for {key}")
        best = min(valid, key=lambda result: result["median_ms"])
        improvement = 100.0 * (baseline["median_ms"] - best["median_ms"]) / baseline[
            "median_ms"
        ]
        if best["tile_n"] != 32 and improvement < args.min_improvement_pct:
            best = baseline
        tactics.append({"m": key[0], "n": key[1], "k": key[2], "tile_n": best["tile_n"]})
        summary.append(
            {
                "m": key[0],
                "n": key[1],
                "k": key[2],
                "occurrences": case["occurrences"],
                "selected_tile_n": best["tile_n"],
                "timings_ms": {
                    str(tile): by_tile[tile][key].get("median_ms") for tile in TILES
                },
                "errors": {
                    str(tile): by_tile[tile][key].get("error") for tile in TILES
                },
            }
        )
    tactics.sort(key=lambda row: (row["m"], row["n"], row["k"]))
    Path(args.header_output).resolve().write_text(render_header(tactics, manifest))
    (output_dir / "tuning_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    result.add_argument("--revision")
    result.add_argument("--num-speculative-tokens", type=int, default=15)
    result.add_argument("--batch-sizes", type=csv_ints, default=csv_ints("1,2,3,4"))
    result.add_argument("--extra-m-values", type=csv_ints)
    result.add_argument("--tensor-parallel-size", type=int, default=1)
    result.add_argument("--include-experts", action="store_true")
    result.add_argument("--output-dir", default="tuning")
    result.add_argument("--header-output", default="generated_tactics.hpp")
    result.add_argument("--manifest-only", action="store_true")
    result.add_argument("--warmup", type=int, default=20)
    result.add_argument("--repeats", type=int, default=7)
    result.add_argument("--iterations", type=int, default=100)
    result.add_argument("--correctness-samples", type=int, default=64)
    result.add_argument("--rtol", type=float, default=0.02)
    result.add_argument("--atol", type=float, default=0.5)
    result.add_argument("--min-improvement-pct", type=float, default=1.0)
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--manifest", help=argparse.SUPPRESS)
    result.add_argument("--tile-n", type=int, choices=TILES, help=argparse.SUPPRESS)
    result.add_argument("--worker-output", help=argparse.SUPPRESS)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.worker:
        if not args.manifest or not args.worker_output or args.tile_n not in TILES:
            raise SystemExit("worker mode requires --manifest, --worker-output, and --tile-n")
        benchmark_worker(args)
    else:
        coordinate(args)


if __name__ == "__main__":
    main()
