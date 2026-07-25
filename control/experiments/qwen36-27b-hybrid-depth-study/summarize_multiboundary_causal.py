#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any


def scan_server_log(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {
            "errors_tracebacks": [f"missing {path}"],
            "warnings": [],
            "scheduler_capacity_errors": [],
            "speculative_warnings": [],
        }
    lines = path.read_text(errors="replace").splitlines()
    errors = [
        line
        for line in lines
        if " ERROR " in line
        or "Traceback (most recent call last)" in line
        or "RuntimeError:" in line
        or "device-side assert triggered" in line
    ]
    warnings = [line for line in lines if " WARNING " in line]
    scheduler_capacity = [
        line
        for line in lines
        if "capacity" in line.lower()
        and any(
            marker in line.lower()
            for marker in ("error", "cannot", "insufficient", "reserve")
        )
    ]
    speculative_warnings = [
        line
        for line in warnings
        if "spec" in line.lower() or "draft" in line.lower()
    ]
    return {
        "errors_tracebacks": errors,
        "warnings": warnings,
        "scheduler_capacity_errors": scheduler_capacity,
        "speculative_warnings": speculative_warnings,
    }


def load_depths(root: Path) -> dict[int, dict[str, Any]]:
    depths: dict[int, dict[str, Any]] = {}
    for depth_dir in root.glob("depth-*"):
        result_path = depth_dir / "run" / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        depth = int(result["depth"])
        result["artifact_root"] = str(depth_dir)
        result["server_log_findings"] = scan_server_log(
            depth_dir / "server-full.log"
        )
        depths[depth] = result
    return depths

NUM_COMPUTED_RE = re.compile(r"num_computed_tokens=\[(\d+)\]")
NUM_OUTPUT_RE = re.compile(r"num_output_tokens=\[(\d+)\]")


def load_execution_failures(root: Path) -> dict[str, dict[str, Any]]:
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        return {}
    failures: dict[str, dict[str, Any]] = {}
    for entry in json.loads(summary_path.read_text()):
        if entry["status"] == "completed":
            continue
        depth = int(entry["depth"])
        depth_root = root / f"depth-{depth}"
        server_path = depth_root / "server-full.log"
        server_text = (
            server_path.read_text(errors="replace") if server_path.is_file() else ""
        )
        computed_matches = NUM_COMPUTED_RE.findall(server_text)
        output_matches = NUM_OUTPUT_RE.findall(server_text)
        num_computed = int(computed_matches[-1]) if computed_matches else None
        num_output = int(output_matches[-1]) if output_matches else None
        prompt_tokens = (
            num_computed - num_output + 1
            if num_computed is not None and num_output is not None
            else None
        )
        block_size = entry.get("startup", {}).get("block_size")
        boundaries = (
            max(
                (prompt_tokens + num_output - 1) // block_size
                - max(prompt_tokens - 1, 0) // block_size,
                0,
            )
            if prompt_tokens is not None
            and num_output is not None
            and block_size is not None
            else None
        )
        process_path = depth_root / "candidate-process.json"
        process = (
            json.loads(process_path.read_text()) if process_path.is_file() else None
        )
        spec_metric_lines = [
            line
            for line in server_text.splitlines()
            if "SpecDecoding metrics:" in line
        ]
        failures[str(depth)] = {
            "status": entry["status"],
            "startup": entry.get("startup"),
            "candidate_process": process,
            "server_log_findings": scan_server_log(server_path),
            "last_speculative_metrics_log_line": (
                spec_metric_lines[-1] if spec_metric_lines else None
            ),
            "scheduler_dump_num_computed_tokens": num_computed,
            "scheduler_dump_num_output_tokens": num_output,
            "inferred_prompt_tokens": prompt_tokens,
            "inferred_decode_boundaries_crossed": boundaries,
            "fatal_device_assert": "device-side assert triggered" in server_text,
            "artifact_root": str(depth_root),
        }
    return failures




def summarize_depth(result: dict[str, Any]) -> dict[str, Any]:
    failures: Counter[str] = Counter()
    hashes_by_case: dict[str, set[str]] = defaultdict(set)
    first_divergences: list[int] = []
    requests: list[dict[str, Any]] = []
    for request in result["requests"]:
        quality = request["quality"]
        failures.update(quality["failure_reasons"])
        hashes_by_case[request["case"]].add(request["output_sha256"])
        divergence = quality["boundary_analysis"][
            "first_divergence_absolute_token"
        ]
        if divergence is not None:
            first_divergences.append(divergence)
        requests.append(
            {
                "request_id": request["request_id"],
                "round": request["round"],
                "case": request["case"],
                "passed": quality["passed"],
                "failure_reasons": quality["failure_reasons"],
                "exact_expected_output": quality["exact_expected_output"],
                "prompt_tokens": request["usage"]["prompt_tokens"],
                "completion_tokens": request["usage"]["completion_tokens"],
                "ttft_seconds": request["ttft_seconds"],
                "decode_duration_seconds": request["decode_duration_seconds"],
                "decode_tokens_per_second": request["decode_tokens_per_second"],
                "e2e_latency_seconds": request["e2e_latency_seconds"],
                "e2e_tokens_per_second": request["e2e_tokens_per_second"],
                "finish_reason": request["finish_reason"],
                "output_sha256": request["output_sha256"],
                "actual_retokenized_completion_tokens": quality[
                    "actual_retokenized_completion_tokens"
                ],
                "boundaries_crossed": quality["boundary_analysis"][
                    "decode_boundaries_crossed"
                ],
                "first_divergence_absolute_token": divergence,
                "first_divergence_block": quality["boundary_analysis"][
                    "first_divergence_block"
                ],
                "first_divergence_offset_in_block": quality[
                    "boundary_analysis"
                ]["first_divergence_offset_in_block"],
                "draft_tokens": request["speculative_metrics"].get(
                    "draft_tokens"
                ),
                "accepted_tokens": request["speculative_metrics"].get(
                    "accepted_tokens"
                ),
                "draft_steps": request["speculative_metrics"].get("draft_steps"),
                "acceptance_rate": request["speculative_metrics"].get(
                    "acceptance_rate"
                ),
                "accepted_tokens_per_draft_step": request[
                    "speculative_metrics"
                ].get("accepted_tokens_per_draft_step"),
            }
        )

    deterministic_by_case = {
        case: len(hashes) == 1 for case, hashes in hashes_by_case.items()
    }
    return {
        "depth": result["depth"],
        "runtime_label": result["runtime_label"],
        "block_size": result["block_size"],
        "quality_pass_count": result["quality_pass_count"],
        "quality_total": result["quality_total"],
        "all_quality_passed": result["all_quality_passed"],
        "all_requests_crossed_minimum_boundaries": result[
            "all_requests_crossed_minimum_boundaries"
        ],
        "failure_reason_counts": dict(sorted(failures.items())),
        "deterministic_by_case": deterministic_by_case,
        "unique_hash_count_by_case": {
            case: len(hashes) for case, hashes in hashes_by_case.items()
        },
        "earliest_first_divergence_absolute_token": (
            min(first_divergences) if first_divergences else None
        ),
        "aggregate_decode_tokens_per_second": result[
            "aggregate_decode_tokens_per_second"
        ],
        "aggregate_e2e_tokens_per_second": result[
            "aggregate_e2e_tokens_per_second"
        ],
        "workload_wall_seconds": result["workload_wall_seconds"],
        "peak_gpu_memory_mib": result["peak_gpu_memory_mib"],
        "speculative_metrics": result["speculative_metrics"],
        "server_log_findings": result["server_log_findings"],
        "artifact_root": result["artifact_root"],
        "requests": requests,
    }


def compare_runtime_pair(
    left_label: str,
    left_depths: dict[int, dict[str, Any]],
    right_label: str,
    right_depths: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for depth in sorted(set(left_depths) & set(right_depths)):
        left = left_depths[depth]
        right = right_depths[depth]
        left_requests = {
            request["request_id"]: request for request in left["requests"]
        }
        right_requests = {
            request["request_id"]: request for request in right["requests"]
        }
        request_comparisons: list[dict[str, Any]] = []
        for request_id in sorted(set(left_requests) & set(right_requests)):
            left_request = left_requests[request_id]
            right_request = right_requests[request_id]
            request_comparisons.append(
                {
                    "request_id": request_id,
                    "left_passed": left_request["quality"]["passed"],
                    "right_passed": right_request["quality"]["passed"],
                    "output_hash_equal": left_request["output_sha256"]
                    == right_request["output_sha256"],
                    "left_output_sha256": left_request["output_sha256"],
                    "right_output_sha256": right_request["output_sha256"],
                    "left_e2e_latency_seconds": left_request[
                        "e2e_latency_seconds"
                    ],
                    "right_e2e_latency_seconds": right_request[
                        "e2e_latency_seconds"
                    ],
                    "left_accepted_tokens": left_request[
                        "speculative_metrics"
                    ].get("accepted_tokens"),
                    "right_accepted_tokens": right_request[
                        "speculative_metrics"
                    ].get("accepted_tokens"),
                }
            )
        comparisons[str(depth)] = {
            "left_label": left_label,
            "right_label": right_label,
            "left_all_quality_passed": left["all_quality_passed"],
            "right_all_quality_passed": right["all_quality_passed"],
            "left_block_size": left["block_size"],
            "right_block_size": right["block_size"],
            "request_comparisons": request_comparisons,
        }
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime_depths: dict[str, dict[int, dict[str, Any]]] = {}
    runtime_roots: dict[str, str] = {}
    runtime_failures: dict[str, dict[str, dict[str, Any]]] = {}
    for runtime_arg in args.runtime:
        if "=" not in runtime_arg:
            raise ValueError("--runtime must be LABEL=PATH")
        label, raw_root = runtime_arg.split("=", 1)
        root = Path(raw_root)
        runtime_roots[label] = str(root)
        runtime_depths[label] = load_depths(root)
        runtime_failures[label] = load_execution_failures(root)

    report: dict[str, Any] = {
        "schema_version": 1,
        "runtimes": {
            label: {
                "root": runtime_roots[label],
                "depths": {
                    str(depth): summarize_depth(result)
                    for depth, result in sorted(depths.items())
                },
                "execution_failures": runtime_failures[label],
            }
            for label, depths in runtime_depths.items()
        },
        "comparisons": {},
    }
    labels = list(runtime_depths)
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1 :]:
            key = f"{left_label}__vs__{right_label}"
            report["comparisons"][key] = compare_runtime_pair(
                left_label,
                runtime_depths[left_label],
                right_label,
                runtime_depths[right_label],
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
