#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
HARNESS = ROOT / "run_omp_candidate.py"
CORPUS = ROOT / "dflash-kv-dtype-study-20260718"
OUTPUT_ROOT = ROOT / "thinkingcap-omp-v6-spec-sweep-spark1-20260720-corrected"
IMAGE = "vllm:thinkingcap-spec-sweep-arm64-20260720"
CONTAINER = "thinkingcap-spec-sweep"
CACHE_VOLUME = "vllm-compile-cache-thinkingcap-spec-sweep"
MODEL = (
    "/models/hub/models--morosystems--ThinkingCap-Qwen3.6-27B-NVFP4/"
    "snapshots/656627c8f7ea4785413ab1e06f6ccd20bba6622f"
)
DRAFTER = (
    "/models/hub/models--z-lab--Qwen3.6-27B-DFlash/"
    "snapshots/0919688658996800f86b895034249700e9481106"
)
TASK_ORDER = [
    "tool_route_read",
    "structured_debug",
    "agent_implementation",
    "long_code_review",
    "cancellation_reasoning",
    "targeted_edit",
]
SEED = 424242
BASE_URL = "http://127.0.0.1:8000"
BLOCK_SIZE_RE = re.compile(r"attention block size to ([0-9]+) tokens")


def command(
    args: list[str],
    *,
    check: bool = True,
    timeout: float = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def remove_container() -> None:
    command(["docker", "rm", "--force", CONTAINER], check=False, timeout=60)


def available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024 / 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def speculative_config(method: str, depth: int) -> dict[str, Any] | None:
    if method == "plain":
        return None
    if method == "dflash":
        return {
            "method": "dflash",
            "model": DRAFTER,
            "num_speculative_tokens": depth,
        }
    if method == "mtp":
        return {"method": "mtp", "num_speculative_tokens": depth}
    raise ValueError(f"unknown method: {method}")


def launch_command(
    method: str,
    depth: int,
    gpu_memory_utilization: float,
    enforce_eager: bool = False,
) -> list[str]:
    served = f"thinkingcap_{method}{depth}_spark1"
    args = [
        "docker",
        "run",
        "--detach",
        "--name",
        CONTAINER,
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "--pids-limit",
        "4096",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=4g",
        "-p",
        "127.0.0.1:8000:8000",
        "-e",
        "HOME=/cache",
        "-e",
        "HF_HOME=/models/hub",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "-e",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "-e",
        "CUDA_VISIBLE_DEVICES=0",
        "-e",
        "VLLM_WORKER_MULTIPROC_METHOD=spawn",
        "-e",
        "NCCL_CUMEM_ENABLE=0",
        "-e",
        "VLLM_FORCE_UVA=1",
        "-e",
        "FLASHINFER_DISABLE_VERSION_CHECK=1",
        "-v",
        "/home/jjlink/models:/models:ro",
        "-v",
        f"{CACHE_VOLUME}:/cache",
        IMAGE,
        MODEL,
        "--served-model-name",
        served,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "262144",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--kv-cache-dtype",
        "bfloat16",
        "--max-num-batched-tokens",
        "8688",
        "--max-num-seqs",
        "16",
        "--mamba-ssm-cache-dtype",
        "float32",
        "--mamba-cache-dtype",
        "float16",
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--disable-custom-all-reduce",
        "--linear-backend",
        "cutlass",
        "--limit-mm-per-prompt",
        '{"image": 0, "video": 0}',
        "--generation-config",
        "vllm",
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--enable-flashinfer-autotune",
    ]
    if enforce_eager:
        args.append("--enforce-eager")
    spec = speculative_config(method, depth)
    if spec is not None:
        args.extend(
            [
                "--speculative-config",
                json.dumps(spec, sort_keys=True, separators=(",", ":")),
            ]
        )
    return args


def container_state() -> str:
    result = command(
        ["docker", "inspect", "--format", "{{.State.Status}}/{{.State.ExitCode}}", CONTAINER],
        check=False,
        timeout=30,
    )
    return (result.stdout or result.stderr).strip()


def fetch_json(path: str, timeout: float = 10) -> dict[str, Any]:
    with urllib.request.urlopen(BASE_URL + path, timeout=timeout) as response:
        return json.loads(response.read())


def wait_ready(timeout_seconds: float = 1200) -> tuple[bool, float, str]:
    started = time.monotonic()
    state = "unknown"
    while time.monotonic() - started < timeout_seconds:
        state = container_state()
        if state.startswith("exited/") or state.startswith("dead/"):
            return False, time.monotonic() - started, state
        try:
            model_info = fetch_json("/v1/models", timeout=5)["data"][0]
            if model_info.get("max_model_len") == 262144:
                return True, time.monotonic() - started, state
        except (OSError, KeyError, IndexError, json.JSONDecodeError):
            pass
        time.sleep(5)
    return False, time.monotonic() - started, state


def docker_logs() -> str:
    result = command(["docker", "logs", CONTAINER], check=False, timeout=120)
    return result.stdout + result.stderr


def detect_block_size(logs: str, method: str) -> int:
    matches = BLOCK_SIZE_RE.findall(logs)
    if matches:
        return int(matches[-1])
    return 848 if method == "dflash" else 800


def per_round_decode_tps(result: dict[str, Any]) -> list[float]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for request in result["requests"]:
        grouped.setdefault(int(request["round"]), []).append(request)
    values: list[float] = []
    for round_index in sorted(grouped):
        requests = grouped[round_index]
        completion_tokens = sum(
            int((request.get("usage") or {}).get("completion_tokens") or 0)
            for request in requests
        )
        decode_seconds = sum(
            float(request.get("decode_duration_seconds") or 0) for request in requests
        )
        if decode_seconds:
            values.append((completion_tokens - len(requests)) / decode_seconds)
    return values










def candidate_summary(
    result: dict[str, Any], *, method: str, depth: int, stage: str
) -> dict[str, Any]:
    round_tps = per_round_decode_tps(result)
    failed_tasks = sorted(
        {
            request["task"]
            for request in result["requests"]
            if not request["quality"]["passed"]
        }
    )
    return {
        "stage": stage,
        "method": method,
        "depth": depth,
        "rounds": result["rounds"],
        "workload": result["workload"],
        "corpus_reference": result["corpus"]["reference_run"],
        "quality_pass_count": result["quality_pass_count"],
        "quality_total": result["quality_total"],
        "all_quality_passed": result["all_quality_passed"],
        "failed_tasks": failed_tasks,
        "decode_tokens_per_second": result["aggregate_decode_tokens_per_second"],
        "e2e_tokens_per_second": result["aggregate_e2e_tokens_per_second"],
        "workload_wall_seconds": result["workload_wall_seconds"],
        "completion_tokens": result["total_completion_tokens"],
        "round_decode_tokens_per_second": round_tps,
        "round_decode_mean": statistics.fmean(round_tps) if round_tps else None,
        "round_decode_stdev": statistics.stdev(round_tps) if len(round_tps) > 1 else 0.0,
        "speculative_metrics": result["speculative_metrics"],
        "peak_gpu_memory_mib": result["peak_gpu_memory_mib"],
    }


def run_candidate(
    *,
    method: str,
    depth: int,
    stage: str,
    rounds: int,
    gpu_memory_utilization: float,
    resume: bool,
    enforce_eager: bool = False,
) -> dict[str, Any] | None:
    label = f"{stage}-{method}{depth}-r{rounds}"
    candidate_dir = OUTPUT_ROOT / label
    run_dir = candidate_dir / "run"
    result_path = run_dir / "result.json"
    if resume and result_path.is_file():
        print(f"{label}: retaining completed result", flush=True)
        return json.loads(result_path.read_text())
    if candidate_dir.exists():
        if not resume:
            raise RuntimeError(
                f"{candidate_dir} exists without a reusable result; rerun with --resume"
            )
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True)

    available_before = available_memory_gib()
    if available_before < 24:
        raise RuntimeError(
            f"only {available_before:.1f} GiB MemAvailable before {label}; refusing startup"
        )

    remove_container()
    launch = launch_command(
        method, depth, gpu_memory_utilization, enforce_eager=enforce_eager
    )
    (candidate_dir / "launch.json").write_text(
        json.dumps(launch, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{label}: launching with {available_before:.1f} GiB available",
        flush=True,
    )
    created = command(launch, timeout=180)
    container_id = created.stdout.strip()
    ready = False
    startup_seconds = 0.0
    state = "unknown"
    try:
        ready, startup_seconds, state = wait_ready()
        startup_logs = docker_logs()
        (candidate_dir / "server-startup.log").write_text(
            startup_logs, encoding="utf-8"
        )
        block_size = detect_block_size(startup_logs, method)
        startup = {
            "container_id": container_id,
            "ready": ready,
            "startup_seconds": startup_seconds,
            "state": state,
            "block_size": block_size,
            "available_memory_gib_before": available_before,
            "available_memory_gib_ready": available_memory_gib(),
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        (candidate_dir / "startup.json").write_text(
            json.dumps(startup, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"{label}: ready={ready} startup={startup_seconds:.1f}s "
            f"state={state} block={block_size}",
            flush=True,
        )
        if not ready:
            return None

        model_info = fetch_json("/v1/models")["data"][0]
        if model_info.get("max_model_len") != 262144:
            raise RuntimeError(f"unexpected model info: {model_info}")

        harness_args = [
            "python3",
            str(HARNESS),
            "--depth",
            str(depth),
            "--block-size",
            str(block_size),
            "--corpus",
            str(CORPUS),
            "--output",
            str(run_dir),
            "--runtime-label",
            label,
            "--rounds",
            str(rounds),
            "--disable-thinking",
            "--task-sequence",
            *TASK_ORDER,
        ]
        process = command(harness_args, check=False, timeout=10800)
        (candidate_dir / "candidate-process.json").write_text(
            json.dumps(
                {
                    "returncode": process.returncode,
                    "stdout": process.stdout,
                    "stderr": process.stderr,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if not result_path.is_file():
            raise RuntimeError(
                f"{label}: harness exited {process.returncode} without result.json"
            )
        result = json.loads(result_path.read_text())
        summary = candidate_summary(result, method=method, depth=depth, stage=stage)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return result
    finally:
        try:
            (candidate_dir / "server-full.log").write_text(
                docker_logs(), encoding="utf-8"
            )
        except Exception as error:
            (candidate_dir / "log-capture-error.txt").write_text(
                f"{type(error).__name__}: {error}\n", encoding="utf-8"
            )
        command(
            ["docker", "stop", "--time", "20", CONTAINER],
            check=False,
            timeout=60,
        )
        remove_container()
        time.sleep(5)


def completed_summaries() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    if not OUTPUT_ROOT.is_dir():
        return summaries
    for candidate_dir in sorted(OUTPUT_ROOT.iterdir()):
        result_path = candidate_dir / "run" / "result.json"
        if not result_path.is_file():
            continue
        match = re.fullmatch(
            r"(baseline|screen|qualify)-(plain|dflash|mtp)([0-9]+)-r([0-9]+)",
            candidate_dir.name,
        )
        if not match:
            continue
        stage, method, depth, _ = match.groups()
        result = json.loads(result_path.read_text())
        summaries.append(
            candidate_summary(
                result,
                method=method,
                depth=int(depth),
                stage=stage,
            )
        )
    return summaries


def top_depths(method: str, count: int = 2) -> list[int]:
    candidates = [
        summary
        for summary in completed_summaries()
        if summary["stage"] == "screen" and summary["method"] == method
    ]
    candidates.sort(
        key=lambda item: (
            item["all_quality_passed"],
            item["quality_pass_count"] / item["quality_total"],
            item["decode_tokens_per_second"],
        ),
        reverse=True,
    )
    return [int(item["depth"]) for item in candidates[:count]]


def write_summary(config: dict[str, Any]) -> dict[str, Any]:
    candidates = completed_summaries()
    qualification_results = [
        item for item in candidates if item["stage"] == "qualify"
    ]
    valid_results = [
        item for item in qualification_results if item["all_quality_passed"]
    ]
    valid_by_method: dict[str, dict[str, Any] | None] = {}
    observed_by_method: dict[str, dict[str, Any] | None] = {}
    for method in ("dflash", "mtp"):
        method_valid = [
            item for item in valid_results if item["method"] == method
        ]
        method_valid.sort(
            key=lambda item: item["decode_tokens_per_second"], reverse=True
        )
        valid_by_method[method] = method_valid[0] if method_valid else None

        method_observed = [
            item for item in qualification_results if item["method"] == method
        ]
        method_observed.sort(
            key=lambda item: (
                item["quality_pass_count"] / item["quality_total"],
                item["decode_tokens_per_second"],
            ),
            reverse=True,
        )
        observed_by_method[method] = method_observed[0] if method_observed else None

    valid_overall = [item for item in valid_by_method.values() if item is not None]
    valid_overall.sort(
        key=lambda item: item["decode_tokens_per_second"], reverse=True
    )
    summary = {
        "schema_version": 1,
        "objective_metric": "aggregate_decode_tokens_per_second",
        "seed": SEED,
        "task_order": TASK_ORDER,
        "thinking_enabled": False,
        "correctness_criterion": "all OMP-v6 validators pass in every measured round",
        "correctness_valid_qualified_count": len(valid_results),
        "config": config,
        "candidates": sorted(
            candidates,
            key=lambda item: (
                {"baseline": 0, "screen": 1, "qualify": 2}[item["stage"]],
                item["method"],
                item["depth"],
            ),
        ),
        "best_correctness_valid_by_method": valid_by_method,
        "best_correctness_valid_overall": (
            valid_overall[0] if valid_overall else None
        ),
        "best_observed_by_method": observed_by_method,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("baseline", "screen", "qualify", "all"), default="all"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--screen-rounds", type=int, default=1)
    parser.add_argument("--qualification-rounds", type=int, default=3)
    parser.add_argument(
        "--dflash-values",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 20, 24],
    )
    parser.add_argument("--mtp-values", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    if args.output_root is not None:
        OUTPUT_ROOT = args.output_root.resolve()

    if not HARNESS.is_file():
        parser.error(f"missing harness: {HARNESS}")
    if not CORPUS.is_dir():
        parser.error(f"missing support corpus: {CORPUS}")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")

    config = {
        "image": IMAGE,
        "model": MODEL,
        "drafter": DRAFTER,
        "max_model_len": 262144,
        "kv_cache_dtype": "bfloat16",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "harness": str(HARNESS),
        "support_corpus": str(CORPUS),
        "screen_rounds": args.screen_rounds,
        "qualification_rounds": args.qualification_rounds,
        "dflash_values": args.dflash_values,
        "mtp_values": args.mtp_values,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "objective_metric": "aggregate_decode_tokens_per_second",
                "aggregate_formula": (
                    "(total_completion_tokens - request_count) / "
                    "sum_decode_durations"
                ),
                "seed": SEED,
                "task_order": TASK_ORDER,
                "thinking_enabled": False,
                "correctness_criterion": (
                    "all OMP-v6 validators must pass in every measured round"
                ),
                **config,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        if args.phase in ("baseline", "all"):
            baseline = run_candidate(
                method="plain",
                depth=0,
                stage="baseline",
                rounds=args.qualification_rounds,
                gpu_memory_utilization=args.gpu_memory_utilization,
                resume=args.resume,
                enforce_eager=args.enforce_eager,
            )
            if baseline is None or len(baseline["requests"]) != (
                len(TASK_ORDER) * args.qualification_rounds
            ):
                raise RuntimeError("plain baseline did not complete every task")
            if any(
                request["http_status"] != 200
                or request["finish_reason"] not in {"stop", "tool_calls"}
                for request in baseline["requests"]
            ):
                raise RuntimeError("plain baseline did not finish cleanly")

        if args.phase in ("screen", "all"):
            for depth in args.dflash_values:
                run_candidate(
                    method="dflash",
                    depth=depth,
                    stage="screen",
                    rounds=args.screen_rounds,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    resume=args.resume,
                    enforce_eager=args.enforce_eager,
                )
            for depth in args.mtp_values:
                run_candidate(
                    method="mtp",
                    depth=depth,
                    stage="screen",
                    rounds=args.screen_rounds,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    resume=args.resume,
                    enforce_eager=args.enforce_eager,
                )

        if args.phase in ("qualify", "all"):
            for method in ("dflash", "mtp"):
                depths = top_depths(method)
                if not depths:
                    raise RuntimeError(
                        f"no quality-valid {method} screen result to qualify"
                    )
                for depth in depths:
                    run_candidate(
                        method=method,
                        depth=depth,
                        stage="qualify",
                        rounds=args.qualification_rounds,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        resume=args.resume,
                        enforce_eager=args.enforce_eager,
                    )
    finally:
        remove_container()
        summary = write_summary(config)
        print(OUTPUT_ROOT / "summary.json", flush=True)
        print(
            json.dumps(
                {
                    "best_correctness_valid": summary.get(
                        "best_correctness_valid_by_method"
                    ),
                    "best_observed": summary.get("best_observed_by_method"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
