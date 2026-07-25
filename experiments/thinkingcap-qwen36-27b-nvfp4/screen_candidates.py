#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path("/home/workbench/inference")
HARNESS = ROOT / "experiments/qwen36-27b-hybrid-depth-study/run_omp_candidate.py"
LAUNCHER = "morosystems_thinkingcap_qwen36_27b_nvfp4"
BASE_URL = "http://127.0.0.1:8000"
BLOCK_SIZE_RE = re.compile(r"Setting attention block size to ([0-9]+) tokens")


def command(
    args: list[str],
    *,
    check: bool = True,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=check,
        timeout=timeout,
    )


def container_name(method: str, depth: int) -> str:
    return f"thinkingcap-screen-{method}{depth}"


def remove_container(name: str) -> None:
    command(["docker", "rm", "--force", name], check=False, timeout=120)


def container_state(name: str) -> str:
    result = command(
        ["docker", "inspect", "--format", "{{.State.Status}}/{{.State.ExitCode}}", name],
        check=False,
        timeout=30,
    )
    return result.stdout.strip() or result.stderr.strip()


def container_logs(name: str) -> str:
    result = command(["docker", "logs", name], check=False, timeout=60)
    return result.stdout + result.stderr


def healthy() -> bool:
    try:
        with urllib.request.urlopen(BASE_URL + "/health", timeout=2) as response:
            return response.status == 200
    except (OSError, TimeoutError, urllib.error.URLError):
        return False


def launch(method: str, depth: int, timeout_seconds: int) -> dict[str, Any]:
    name = container_name(method, depth)
    remove_container(name)
    env = os.environ.copy()
    env.pop("DFLASH_MAX_CONTEXT", None)
    env.update(
        {
            "DETACH": "1",
            "KEEP": "1",
            "CONTAINER_NAME": name,
            "CACHE_VOL": "vllm-mrv2-compile-cache-standard",
            "SPEC": method,
            "NUM_SPEC": str(depth),
            "SERVED": f"thinkingcap_qwen36_27b_nvfp4_{method}{depth}_bf16kv",
        }
    )
    started = time.monotonic()
    process = command(
        [str(ROOT / "serve.sh"), LAUNCHER],
        check=False,
        timeout=120,
        env=env,
    )
    if process.returncode != 0:
        return {
            "ready": False,
            "elapsed_seconds": time.monotonic() - started,
            "launch_stdout": process.stdout,
            "launch_stderr": process.stderr,
            "state": container_state(name),
            "server_log": container_logs(name),
        }

    deadline = started + timeout_seconds
    while time.monotonic() < deadline:
        state = container_state(name)
        if state.startswith(("exited", "dead")):
            return {
                "ready": False,
                "elapsed_seconds": time.monotonic() - started,
                "launch_stdout": process.stdout,
                "launch_stderr": process.stderr,
                "state": state,
                "server_log": container_logs(name),
            }
        if healthy():
            logs = container_logs(name)
            match = BLOCK_SIZE_RE.search(logs)
            if match:
                return {
                    "ready": True,
                    "elapsed_seconds": time.monotonic() - started,
                    "block_size": int(match.group(1)),
                    "launch_stdout": process.stdout,
                    "launch_stderr": process.stderr,
                    "state": state,
                    "server_log": logs,
                }
        time.sleep(2)

    return {
        "ready": False,
        "elapsed_seconds": time.monotonic() - started,
        "launch_stdout": process.stdout,
        "launch_stderr": process.stderr,
        "state": container_state(name),
        "server_log": container_logs(name),
    }


def run_harness(
    *,
    method: str,
    depth: int,
    block_size: int,
    output: Path,
    rounds: int,
) -> subprocess.CompletedProcess[str]:
    return command(
        [
            sys.executable,
            str(HARNESS),
            "--depth",
            str(depth),
            "--block-size",
            str(block_size),
            "--corpus",
            str(HARNESS),
            "--output",
            str(output),
            "--runtime-label",
            f"thinkingcap-{method}{depth}-screen",
            "--rounds",
            str(rounds),
        ],
        check=False,
        timeout=7200,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("mtp", "dflash"), required=True)
    parser.add_argument("--depths", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--startup-timeout", type=int, default=600)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    summary: list[dict[str, Any]] = []
    try:
        for depth in args.depths:
            depth_root = args.output / f"depth-{depth}"
            depth_root.mkdir()
            name = container_name(args.method, depth)
            startup = launch(args.method, depth, args.startup_timeout)
            startup_log = startup.pop("server_log")
            (depth_root / "startup.json").write_text(
                json.dumps(startup, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (depth_root / "server-startup.log").write_text(
                startup_log,
                encoding="utf-8",
            )
            if not startup["ready"]:
                summary.append({"depth": depth, "status": "startup-failed"})
                remove_container(name)
                continue

            process = run_harness(
                method=args.method,
                depth=depth,
                block_size=startup["block_size"],
                output=depth_root / "run",
                rounds=args.rounds,
            )
            (depth_root / "server-full.log").write_text(
                container_logs(name),
                encoding="utf-8",
            )
            (depth_root / "candidate-process.json").write_text(
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
            result_path = depth_root / "run/result.json"
            result = json.loads(result_path.read_text()) if result_path.is_file() else None
            summary.append(
                {
                    "depth": depth,
                    "status": "completed" if result is not None else "candidate-failed",
                    "candidate_returncode": process.returncode,
                    "result": result,
                }
            )
            if result is None:
                print(f"{args.method}{depth}: candidate failed", flush=True)
            else:
                print(
                    f"{args.method}{depth}: quality "
                    f"{result['quality_pass_count']}/{result['quality_total']} "
                    f"wall={result['workload_wall_seconds']:.3f}s "
                    f"decode={result['aggregate_decode_tokens_per_second']:.3f}",
                    flush=True,
                )
            remove_container(name)
            time.sleep(3)
    finally:
        for depth in args.depths:
            remove_container(container_name(args.method, depth))

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 1 if any(item["status"] != "completed" for item in summary) else 0


if __name__ == "__main__":
    raise SystemExit(main())
