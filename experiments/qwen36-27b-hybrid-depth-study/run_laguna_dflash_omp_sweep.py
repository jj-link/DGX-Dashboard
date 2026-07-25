#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
HARNESS = ROOT / "run_omp_candidate.py"
CORPUS = ROOT / "omp-six-task-corpus-v6.json"
OUTPUT_ROOT = ROOT / "laguna-dflash-omp-v6-sweep-20260722"
BASE_URL = "http://100.86.3.45:8000"
SSH_HOST = "spark1-ts"
REMOTE_LAUNCHER = (
    "/home/jjlink/inference/serve/vllm/"
    "serve_poolside_laguna_s_2_1_nvfp4_dflash.sh"
)
MODEL_ID = "poolside/Laguna-S-2.1-NVFP4"
DEPTHS = (4, 5, 6, 8, 9, 10, 11, 12, 16, 20, 24)
BLOCK_SIZE = 16


def fetch_json(path: str, timeout: float = 10) -> dict[str, Any]:
    with urllib.request.urlopen(BASE_URL + path, timeout=timeout) as response:
        return json.loads(response.read())


def stop_server() -> None:
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            SSH_HOST,
            "pkill -TERM -f '^/home/jjlink/venvs/vllm025/bin/python3 "
            "/home/jjlink/venvs/vllm025/bin/vllm serve "
            "/home/jjlink/hf-cache/hub/models--poolside--Laguna-S-2.1-NVFP4'",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            fetch_json("/v1/models", timeout=2)
        except (OSError, urllib.error.URLError):
            return
        time.sleep(2)
    raise RuntimeError("Laguna endpoint did not stop within 180 seconds")


def launch_server(depth: int, log_path: Path) -> tuple[subprocess.Popen[bytes], Any, float]:
    stop_server()
    log_file = log_path.open("wb")
    started = time.monotonic()
    process = subprocess.Popen(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            SSH_HOST,
            f"NUM_SPECULATIVE_TOKENS={depth} bash {REMOTE_LAUNCHER}",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    deadline = started + 1200
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_file.flush()
            raise RuntimeError(
                f"depth {depth} server exited with {process.returncode}; see {log_path}"
            )
        try:
            models = fetch_json("/v1/models", timeout=5).get("data", [])
            if models and models[0].get("id") == MODEL_ID:
                return process, log_file, time.monotonic() - started
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(5)
    process.terminate()
    log_file.close()
    raise RuntimeError(f"depth {depth} server was not ready within 1200 seconds")


def collect_summary() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for result_path in sorted(OUTPUT_ROOT.glob("screen-dflash*-r1/run/result.json")):
        result = json.loads(result_path.read_text())
        candidates.append(
            {
                "depth": result["depth"],
                "decode_tokens_per_second": result[
                    "aggregate_decode_tokens_per_second"
                ],
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
                    request["task"]
                    for request in result["requests"]
                    if not request["quality"]["passed"]
                ],
                "result": str(result_path.relative_to(OUTPUT_ROOT)),
            }
        )
    candidates.sort(key=lambda item: item["decode_tokens_per_second"], reverse=True)
    return {
        "schema_version": 1,
        "workload": "omp-six-task-v6",
        "objective": "aggregate_decode_tokens_per_second",
        "correctness_criterion": "all six OMP-v6 validators pass",
        "target_model": MODEL_ID,
        "draft_model": "poolside/Laguna-S-2.1-DFlash-NVFP4",
        "screen_rounds": 1,
        "seed": 424242,
        "depths": sorted(item["depth"] for item in candidates),
        "candidates": candidates,
    }


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for depth in DEPTHS:
        candidate = OUTPUT_ROOT / f"screen-dflash{depth}-r1"
        result_path = candidate / "run" / "result.json"
        if result_path.is_file():
            print(f"[skip] depth={depth} result exists", flush=True)
            continue
        candidate.mkdir(parents=True, exist_ok=False)
        server_log = candidate / "server.log"
        process: subprocess.Popen[bytes] | None = None
        log_file = None
        try:
            print(f"[launch] depth={depth}", flush=True)
            process, log_file, startup_seconds = launch_server(depth, server_log)
            (candidate / "startup.json").write_text(
                json.dumps(
                    {"depth": depth, "startup_seconds": startup_seconds}, indent=2
                )
                + "\n"
            )
            print(
                f"[ready] depth={depth} startup={startup_seconds:.1f}s", flush=True
            )
            harness_log = candidate / "harness.log"
            env = os.environ.copy()
            env["VLLM_BASE_URL"] = BASE_URL
            env["PYTHONUNBUFFERED"] = "1"
            with harness_log.open("w", encoding="utf-8") as output:
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
                        f"laguna-dflash{depth}-screen-r1",
                        "--rounds",
                        "1",
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=7200,
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
            (OUTPUT_ROOT / "screen-summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
    print(OUTPUT_ROOT / "screen-summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
