#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from run_depth_candidate import (
    GpuMemoryPoller,
    fetch_json,
    fetch_text,
    parse_spec_metrics,
    run_chat_sse,
    subtract_metrics,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_source(text: str) -> str:
    return text.rstrip() + "\n"


def tokenize(model: str, text: str) -> list[int]:
    body = json.dumps(
        {"model": model, "prompt": text, "add_special_tokens": False},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:8000/tokenize",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read())
    return payload["tokens"]


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def boundary_analysis(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    common_prefix_tokens: int,
    block_size: int,
) -> dict[str, Any]:
    first_sequence_index = max(prompt_tokens - 1, 0)
    last_sequence_index = max(prompt_tokens + completion_tokens - 1, 0)
    start_block = first_sequence_index // block_size
    end_block = last_sequence_index // block_size
    crossed = max(end_block - start_block, 0)
    boundaries = [
        block * block_size - prompt_tokens
        for block in range(start_block + 1, end_block + 1)
    ]
    absolute_divergence = prompt_tokens + common_prefix_tokens
    return {
        "block_size": block_size,
        "decode_boundaries_crossed": crossed,
        "completion_token_offsets_of_boundaries": boundaries,
        "common_prefix_completion_tokens": common_prefix_tokens,
        "first_divergence_absolute_token": absolute_divergence,
        "first_divergence_block": absolute_divergence // block_size,
        "first_divergence_offset_in_block": absolute_divergence % block_size,
    }


def verify_corpus(corpus: Path) -> dict[str, Any]:
    manifest_path = corpus / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported corpus schema")
    if manifest.get("case_order") != ["front", "middle", "end"]:
        raise ValueError("unexpected corpus case order")
    for case in manifest["cases"]:
        for kind in ("input", "expected", "request"):
            data = (corpus / case[f"{kind}_path"]).read_bytes()
            actual = sha256(data)
            expected = case[f"{kind}_sha256"]
            if actual != expected:
                raise ValueError(
                    f"{case['name']} {kind} hash {actual} does not match {expected}"
                )
    manifest["manifest_sha256"] = sha256(manifest_bytes)
    return manifest


def external_test_source(case: str, generated_path: Path, helper_count: int) -> str:
    checks: str
    if case == "front":
        checks = '''
assert module.parse_window("jobs:12") == ("jobs", 12)
for value in ("", "jobs:", ":3", "jobs:0", "jobs:-1", "jobs:x"):
    try:
        module.parse_window(value)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError(f"parse_window accepted {value!r}")
'''
    elif case == "middle":
        checks = '''
assert module.take_prefix([1, 2, 3, 4], 0) == []
assert module.take_prefix([1, 2, 3, 4], 1) == [1]
assert module.take_prefix([1, 2, 3, 4], 4) == [1, 2, 3, 4]
assert module.take_prefix([1, 2, 3, 4], 8) == [1, 2, 3, 4]
for value in (-1, True, 1.5):
    try:
        module.take_prefix([1, 2], value)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError(f"take_prefix accepted {value!r}")
'''
    elif case == "end":
        checks = '''
assert module.contains_token(["Alpha", "BETA", "gamma"], "Alpha") is True
assert module.contains_token(["Alpha", "BETA", "gamma"], "alpha") is False
assert module.contains_token(["Alpha", "BETA", "gamma"], "BETA") is True
try:
    module.contains_token(["Alpha"], 3)
except TypeError:
    pass
else:
    raise AssertionError("contains_token accepted a non-string expected value")
'''
    else:
        raise ValueError(f"unknown case: {case}")
    sampled_helpers = sorted({0, helper_count // 3, helper_count // 2, helper_count - 1})
    helper_checks = "\n".join(
        f"assert module.helper_{index:04d}(12345) == "
        f"(12345 * {17 + (index % 19)} + {31 + (index * 7 % 101)}) % 1000003"
        for index in sampled_helpers
    )
    return f'''import importlib.util

spec = importlib.util.spec_from_file_location("generated", {str(generated_path)!r})
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
assert module.MODULE_SENTINEL == "qwen27-multiboundary-v1"
{helper_checks}
{checks}
print("PASS")
'''


def execute_external_test(case: str, source: str, validation_dir: Path, helper_count: int) -> dict[str, Any]:
    validation_dir.mkdir(parents=True, exist_ok=True)
    generated_path = validation_dir / "generated.py"
    generated_path.write_text(source, encoding="utf-8")
    test_path = validation_dir / "external_contract.py"
    test_path.write_text(
        external_test_source(case, generated_path, helper_count), encoding="utf-8"
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=validation_dir,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return {
            "exit_code": completed.returncode,
            "timed_out": False,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "exit_code": None,
            "timed_out": True,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "elapsed_seconds": time.perf_counter() - started,
        }


def validate_response(
    *,
    case: str,
    text: str,
    finish_reason: str | None,
    usage: dict[str, Any] | None,
    expected: str,
    expected_tokens: list[int],
    actual_tokens: list[int],
    block_size: int,
    minimum_boundaries: int,
    validation_dir: Path,
    helper_count: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    canonical_actual = canonical_source(text)
    canonical_expected = canonical_source(expected)
    exact = canonical_actual == canonical_expected
    if finish_reason != "stop":
        reasons.append(f"finish_reason={finish_reason!r}, expected 'stop'")
    if "```" in text:
        reasons.append("response contains a Markdown fence")
    if not exact:
        reasons.append("canonical full-file output differs from expected output")
        diff = "\n".join(
            list(
                difflib.unified_diff(
                    canonical_expected.splitlines(),
                    canonical_actual.splitlines(),
                    fromfile="expected.py",
                    tofile="actual.py",
                    lineterm="",
                )
            )[:300]
        )
        (validation_dir / "output.diff").parent.mkdir(parents=True, exist_ok=True)
        (validation_dir / "output.diff").write_text(diff + "\n", encoding="utf-8")

    common_tokens = common_prefix_length(expected_tokens, actual_tokens)
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    boundaries = boundary_analysis(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        common_prefix_tokens=common_tokens,
        block_size=block_size,
    )
    if exact:
        boundaries["first_divergence_absolute_token"] = None
        boundaries["first_divergence_block"] = None
        boundaries["first_divergence_offset_in_block"] = None
    if boundaries["decode_boundaries_crossed"] < minimum_boundaries:
        reasons.append(
            f"crossed {boundaries['decode_boundaries_crossed']} decode boundaries, "
            f"required at least {minimum_boundaries}"
        )

    external: dict[str, Any] | None = None
    if exact:
        external = execute_external_test(
            case,
            canonical_actual,
            validation_dir,
            helper_count,
        )
        if external["timed_out"]:
            reasons.append("external semantic test timed out")
        elif external["exit_code"] != 0:
            detail = (external["stderr"] or external["stdout"]).strip()
            reasons.append(
                f"external semantic test exited {external['exit_code']}: {detail[-1000:]}"
            )

    return {
        "passed": not reasons,
        "failure_reasons": reasons,
        "exact_expected_output": exact,
        "canonical_output_sha256": sha256(canonical_actual.encode("utf-8")),
        "expected_canonical_sha256": sha256(canonical_expected.encode("utf-8")),
        "expected_completion_tokens": len(expected_tokens),
        "actual_retokenized_completion_tokens": len(actual_tokens),
        "boundary_analysis": boundaries,
        "external_semantic_test": external,
    }


def warmup_payload(model: str) -> dict[str, Any]:
    return {
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 32,
        "messages": [
            {
                "role": "system",
                "content": "Return the requested literal and nothing else.",
            },
            {"role": "user", "content": "Return exactly: WARMUP_READY"},
        ],
        "model": model,
        "seed": 424242,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "top_p": 1.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-label", required=True)
    parser.add_argument("--request-limit", type=int)
    args = parser.parse_args()

    manifest = verify_corpus(args.corpus)
    args.output.mkdir(parents=True, exist_ok=False)
    raw_dir = args.output / "raw"
    outputs_dir = args.output / "outputs"
    validation_root = args.output / "validation"
    for directory in (raw_dir, outputs_dir, validation_root):
        directory.mkdir(parents=True, exist_ok=False)

    model_info = fetch_json("/v1/models")["data"][0]
    model = model_info["id"]
    prepared: dict[str, dict[str, Any]] = {}
    for case_info in manifest["cases"]:
        case = case_info["name"]
        request = json.loads((args.corpus / case_info["request_path"]).read_text())
        request["model"] = model
        expected = (args.corpus / case_info["expected_path"]).read_text()
        expected_tokens = tokenize(model, canonical_source(expected))
        prepared[case] = {
            "request": request,
            "expected": expected,
            "expected_tokens": expected_tokens,
        }

    warmup = run_chat_sse(warmup_payload(model), raw_dir / "warmup.response.sse")
    warmup_text = warmup.pop("text")
    warmup_reasoning = warmup.pop("reasoning")
    (outputs_dir / "warmup.txt").write_text(warmup_text, encoding="utf-8")
    (outputs_dir / "warmup.reasoning.txt").write_text(
        warmup_reasoning, encoding="utf-8"
    )
    warmup["passed"] = (
        warmup["finish_reason"] == "stop" and warmup_text.strip() == "WARMUP_READY"
    )

    requests: list[dict[str, Any]] = []
    metrics_before_all = parse_spec_metrics(fetch_text("/metrics"))
    workload_started = time.perf_counter()
    with GpuMemoryPoller() as memory:
        for round_index in range(1, int(manifest["rounds"]) + 1):
            for case in manifest["case_order"]:
                if (
                    args.request_limit is not None
                    and len(requests) >= args.request_limit
                ):
                    break
                request_id = f"round-{round_index}-{case}"
                metrics_before = parse_spec_metrics(fetch_text("/metrics"))
                generated = run_chat_sse(
                    prepared[case]["request"],
                    raw_dir / f"{request_id}.response.sse",
                )
                metrics_after = parse_spec_metrics(fetch_text("/metrics"))
                text = generated.pop("text")
                reasoning = generated.pop("reasoning")
                (outputs_dir / f"{request_id}.txt").write_text(
                    text, encoding="utf-8"
                )
                (outputs_dir / f"{request_id}.reasoning.txt").write_text(
                    reasoning, encoding="utf-8"
                )
                actual_tokens = tokenize(model, canonical_source(text))
                quality = validate_response(
                    case=case,
                    text=text,
                    finish_reason=generated["finish_reason"],
                    usage=generated["usage"],
                    expected=prepared[case]["expected"],
                    expected_tokens=prepared[case]["expected_tokens"],
                    actual_tokens=actual_tokens,
                    block_size=args.block_size,
                    minimum_boundaries=int(manifest["minimum_decode_boundaries"]),
                    validation_dir=validation_root / request_id,
                    helper_count=int(manifest["helper_count"]),
                )
                generated.update(
                    {
                        "request_id": request_id,
                        "round": round_index,
                        "case": case,
                        "quality": quality,
                        "speculative_metrics": subtract_metrics(
                            metrics_after, metrics_before
                        ),
                    }
                )
                requests.append(generated)
            if (
                args.request_limit is not None
                and len(requests) >= args.request_limit
            ):
                break
    workload_seconds = time.perf_counter() - workload_started
    metrics_after_all = parse_spec_metrics(fetch_text("/metrics"))

    completion_tokens = sum(
        int((request.get("usage") or {}).get("completion_tokens") or 0)
        for request in requests
    )
    decode_seconds = sum(
        float(request.get("decode_duration_seconds") or 0.0) for request in requests
    )
    e2e_seconds = sum(float(request["e2e_latency_seconds"]) for request in requests)
    result = {
        "schema_version": 1,
        "runtime_label": args.runtime_label,
        "depth": args.depth,
        "block_size": args.block_size,
        "served_model": model,
        "max_model_len": model_info.get("max_model_len"),
        "corpus_manifest_sha256": manifest["manifest_sha256"],
        "warmup": warmup,
        "requests": requests,
        "quality_pass_count": sum(request["quality"]["passed"] for request in requests),
        "quality_total": len(requests),
        "all_quality_passed": all(
            request["quality"]["passed"] for request in requests
        ),
        "all_requests_crossed_minimum_boundaries": all(
            request["quality"]["boundary_analysis"]["decode_boundaries_crossed"]
            >= manifest["minimum_decode_boundaries"]
            for request in requests
        ),
        "workload_wall_seconds": workload_seconds,
        "aggregate_decode_tokens_per_second": (
            max(completion_tokens - len(requests), 0) / decode_seconds
            if decode_seconds
            else None
        ),
        "aggregate_e2e_tokens_per_second": (
            completion_tokens / e2e_seconds if e2e_seconds else None
        ),
        "speculative_metrics": subtract_metrics(metrics_after_all, metrics_before_all),
        "peak_gpu_memory_mib": memory.peak_mib,
    }
    result_path = args.output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result_path)
    print(
        json.dumps(
            {
                "runtime_label": args.runtime_label,
                "depth": args.depth,
                "quality": f"{result['quality_pass_count']}/{result['quality_total']}",
                "all_multi_boundary": result[
                    "all_requests_crossed_minimum_boundaries"
                ],
                "wall_seconds": workload_seconds,
                "decode_tps": result["aggregate_decode_tokens_per_second"],
                "speculative_metrics": result["speculative_metrics"],
                "peak_gpu_memory_mib": result["peak_gpu_memory_mib"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["all_quality_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
