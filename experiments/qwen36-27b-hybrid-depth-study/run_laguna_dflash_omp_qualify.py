#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from run_laguna_dflash_omp_sweep import (
    BASE_URL,
    BLOCK_SIZE,
    CORPUS,
    HARNESS,
    OUTPUT_ROOT,
    ROOT,
    launch_server,
    stop_server,
)

DEPTHS = (16, 6, 7)
ROUNDS = 3


def collect_summary() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for result_path in sorted(OUTPUT_ROOT.glob("qualify-dflash*-r3/run/result.json")):
        result = json.loads(result_path.read_text())
        per_round: list[float] = []
        for round_index in range(1, ROUNDS + 1):
            requests = [
                request
                for request in result["requests"]
                if request["round"] == round_index
            ]
            tokens = sum(
                int((request.get("usage") or {}).get("completion_tokens") or 0)
                for request in requests
            )
            seconds = sum(
                float(request.get("decode_duration_seconds") or 0.0)
                for request in requests
            )
            per_round.append((tokens - len(requests)) / seconds)
        candidates.append(
            {
                "depth": result["depth"],
                "decode_tokens_per_second": result[
                    "aggregate_decode_tokens_per_second"
                ],
                "round_decode_tokens_per_second": per_round,
                "e2e_tokens_per_second": result[
                    "aggregate_e2e_tokens_per_second"
                ],
                "quality_pass_count": result["quality_pass_count"],
                "quality_total": result["quality_total"],
                "all_quality_passed": result["all_quality_passed"],
                "completion_tokens": result["total_completion_tokens"],
                "workload_wall_seconds": result["workload_wall_seconds"],
                "acceptance_rate": result["speculative_metrics"]["acceptance_rate"],
                "accepted_tokens_per_draft_step": result["speculative_metrics"][
                    "accepted_tokens_per_draft_step"
                ],
                "failed_tasks": [
                    request["request_id"]
                    for request in result["requests"]
                    if not request["quality"]["passed"]
                ],
                "result": str(result_path.relative_to(OUTPUT_ROOT)),
            }
        )
    candidates.sort(
        key=lambda item: (
            item["all_quality_passed"],
            item["quality_pass_count"] / item["quality_total"],
            item["decode_tokens_per_second"],
        ),
        reverse=True,
    )
    valid = [item for item in candidates if item["all_quality_passed"]]
    valid.sort(key=lambda item: item["decode_tokens_per_second"], reverse=True)
    observed = candidates[0] if candidates else None
    return {
        "schema_version": 1,
        "workload": "omp-six-task-v6",
        "objective": "aggregate_decode_tokens_per_second",
        "correctness_criterion": "all six OMP-v6 validators pass in every round",
        "target_model": "poolside/Laguna-S-2.1-NVFP4",
        "draft_model": "poolside/Laguna-S-2.1-DFlash-NVFP4",
        "qualification_rounds": ROUNDS,
        "seed": 424242,
        "qualified_depths": list(DEPTHS),
        "candidates": candidates,
        "fastest_correctness_valid": valid[0] if valid else None,
        "best_observed": observed,
    }


def main() -> int:
    for depth in DEPTHS:
        candidate = OUTPUT_ROOT / f"qualify-dflash{depth}-r3"
        result_path = candidate / "run" / "result.json"
        if result_path.is_file():
            print(f"[skip] depth={depth} qualification exists", flush=True)
            continue
        candidate.mkdir(parents=True, exist_ok=False)
        process: subprocess.Popen[bytes] | None = None
        log_file = None
        try:
            print(f"[launch] depth={depth}", flush=True)
            process, log_file, startup_seconds = launch_server(
                depth, candidate / "server.log"
            )
            (candidate / "startup.json").write_text(
                json.dumps(
                    {"depth": depth, "startup_seconds": startup_seconds}, indent=2
                )
                + "\n"
            )
            print(
                f"[ready] depth={depth} startup={startup_seconds:.1f}s", flush=True
            )
            env = os.environ.copy()
            env["VLLM_BASE_URL"] = BASE_URL
            env["PYTHONUNBUFFERED"] = "1"
            with (candidate / "harness.log").open("w", encoding="utf-8") as output:
                completed = subprocess.run(
                    [
                        "python3",
                        str(HARNESS),
                        "--depth",
                        str(depth),
                        "--block-size",
                        str(BLOCK_SIZE),
                        "--corpus",
                        str(CORPUS),
                        "--output",
                        str(candidate / "run"),
                        "--runtime-label",
                        f"laguna-dflash{depth}-qualify-r3",
                        "--rounds",
                        str(ROUNDS),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=21600,
                )
            if completed.returncode not in (0, 2):
                raise RuntimeError(
                    f"depth {depth} harness exited with {completed.returncode}"
                )
            result = json.loads(result_path.read_text())
            print(
                f"[result] depth={depth} "
                f"decode_tps={result['aggregate_decode_tokens_per_second']:.3f} "
                f"quality={result['quality_pass_count']}/{result['quality_total']}",
                flush=True,
            )
        finally:
            stop_server()
            if process is not None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if log_file is not None:
                log_file.close()
            summary = collect_summary()
            (OUTPUT_ROOT / "qualification-summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
    print(OUTPUT_ROOT / "qualification-summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
