#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
from run_multiboundary_candidate import external_test_source
from summarize_multiboundary_causal import scan_server_log


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(text: str) -> str:
    return text.rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--runtime", action="append", required=True)
    parser.add_argument("--depths", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_order = [
        f"round-{round_index}-{case_name}"
        for round_index in range(1, int(manifest["rounds"]) + 1)
        for case_name in manifest["case_order"]
    ]
    expected_by_case = {
        case["name"]: (args.corpus / case["expected_path"]).read_text()
        for case in manifest["cases"]
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "corpus_manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "expected_depths": args.depths,
        "runtimes": {},
        "errors": [],
    }

    for runtime_arg in args.runtime:
        if "=" not in runtime_arg:
            raise ValueError("--runtime must be LABEL=PATH")
        label, raw_root = runtime_arg.split("=", 1)
        root = Path(raw_root)
        runtime_report: dict[str, Any] = {"root": str(root), "depths": {}}
        for depth in args.depths:
            depth_root = root / f"depth-{depth}"
            result_path = depth_root / "run" / "result.json"
            depth_errors: list[str] = []
            quality_observations: list[str] = []
            external_rechecks: list[dict[str, Any]] = []
            if not result_path.is_file():
                depth_errors.append(f"missing {result_path}")
                runtime_report["depths"][str(depth)] = {
                    "passed": False,
                    "errors": depth_errors,
                }
                report["errors"].extend(f"{label}/depth-{depth}: {e}" for e in depth_errors)
                continue

            result = json.loads(result_path.read_text())
            startup_path = depth_root / "startup.json"
            if not startup_path.is_file():
                depth_errors.append("startup.json is missing")
            else:
                startup = json.loads(startup_path.read_text())
                if not startup.get("ready"):
                    depth_errors.append("startup did not report ready=true")
                if startup.get("block_size") != result.get("block_size"):
                    depth_errors.append("startup/result block-size mismatch")

            request_ids = [request["request_id"] for request in result["requests"]]
            if request_ids != expected_order:
                depth_errors.append(
                    f"request order mismatch: {request_ids!r} != {expected_order!r}"
                )
            if result["quality_total"] != len(expected_order):
                depth_errors.append("quality_total does not match corpus")
            if result["quality_pass_count"] != sum(
                int(request["quality"]["passed"]) for request in result["requests"]
            ):
                depth_errors.append("quality_pass_count is inconsistent")
            if result["all_quality_passed"] != all(
                request["quality"]["passed"] for request in result["requests"]
            ):
                depth_errors.append("all_quality_passed is inconsistent")

            for request in result["requests"]:
                request_id = request["request_id"]
                case = request["case"]
                output_path = depth_root / "run" / "outputs" / f"{request_id}.txt"
                sse_path = depth_root / "run" / "raw" / f"{request_id}.response.sse"
                semantic_recheck_root = (
                    depth_root / "run" / "semantic-recheck" / request_id
                )
                if not output_path.is_file():
                    depth_errors.append(f"{request_id}: output file is missing")
                    continue
                output_bytes = output_path.read_bytes()
                output_text = output_bytes.decode("utf-8")
                if sha256_bytes(output_bytes) != request["output_sha256"]:
                    depth_errors.append(f"{request_id}: output SHA-256 mismatch")
                exact = canonical(output_text) == canonical(expected_by_case[case])
                if exact != request["quality"]["exact_expected_output"]:
                    depth_errors.append(f"{request_id}: exact-output flag mismatch")
                if not sse_path.is_file() or sse_path.stat().st_size == 0:
                    depth_errors.append(f"{request_id}: raw SSE is missing or empty")
                boundaries = request["quality"]["boundary_analysis"][
                    "decode_boundaries_crossed"
                ]
                if boundaries < int(manifest["minimum_decode_boundaries"]):
                    quality_observations.append(
                        f"{request_id}: crossed {boundaries} boundaries, required at least "
                        f"{manifest['minimum_decode_boundaries']}"
                    )

                semantic_recheck_root.mkdir(parents=True, exist_ok=True)
                generated_path = semantic_recheck_root / "generated.py"
                generated_path.write_text(output_text, encoding="utf-8")
                external_path = semantic_recheck_root / "external_contract.py"
                external_path.write_text(
                    external_test_source(
                        case,
                        generated_path,
                        int(manifest["helper_count"]),
                    ),
                    encoding="utf-8",
                )
                try:
                    completed = subprocess.run(
                        ["python3", str(external_path)],
                        cwd=semantic_recheck_root,
                        capture_output=True,
                        text=True,
                        timeout=20,
                        check=False,
                    )
                    recheck = {
                        "request_id": request_id,
                        "returncode": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                        "timed_out": False,
                    }
                except subprocess.TimeoutExpired as exc:
                    recheck = {
                        "request_id": request_id,
                        "returncode": None,
                        "stdout": exc.stdout or "",
                        "stderr": exc.stderr or "",
                        "timed_out": True,
                    }
                external_rechecks.append(recheck)
                semantic_passed = (
                    not recheck["timed_out"] and recheck["returncode"] == 0
                )
                if not semantic_passed:
                    quality_observations.append(
                        f"{request_id}: external semantic recheck failed"
                    )
                if request["quality"]["passed"] and not semantic_passed:
                    depth_errors.append(
                        f"{request_id}: recorded quality pass failed semantic recheck"
                    )
                recorded_external = request["quality"]["external_semantic_test"]
                if recorded_external is not None:
                    if recheck["returncode"] != recorded_external["exit_code"]:
                        depth_errors.append(
                            f"{request_id}: external recheck exit differs from recorded exit"
                        )
                    if recheck["timed_out"] != recorded_external["timed_out"]:
                        depth_errors.append(
                            f"{request_id}: external recheck timeout differs from recorded timeout"
                        )

            if depth > 0:
                metrics = result["speculative_metrics"]
                if not metrics.get("draft_tokens") or not metrics.get("draft_steps"):
                    depth_errors.append(
                        "aggregate speculative counters did not increase"
                    )

            findings = scan_server_log(depth_root / "server-full.log")
            if findings["errors_tracebacks"]:
                depth_errors.append("server log contains an error or traceback")
            if findings["scheduler_capacity_errors"]:
                depth_errors.append("server log contains a scheduler-capacity error")
            if result["peak_gpu_memory_mib"] > 97887:
                depth_errors.append("peak GPU memory exceeds hardware limit")

            runtime_report["depths"][str(depth)] = {
                "passed": not depth_errors,
                "errors": depth_errors,
                "quality_observations": quality_observations,
                "quality_pass_count": result["quality_pass_count"],
                "quality_total": result["quality_total"],
                "external_rechecks": external_rechecks,
                "server_log_findings": findings,
            }
            report["errors"].extend(f"{label}/depth-{depth}: {e}" for e in depth_errors)
        report["runtimes"][label] = runtime_report

    report["passed"] = not report["errors"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
