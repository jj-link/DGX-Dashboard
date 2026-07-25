#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

IMAGE = "vllm:v0.24.0-hybrid-recovered-uva"
CONTAINER = "qwen27-hybrid-depth-sweep"
BASE_URL = "http://127.0.0.1:8000"


def command(args: list[str], *, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        text=True,
        capture_output=True,
        check=check,
        timeout=timeout,
    )


def remove_container() -> None:
    command(["docker", "rm", "--force", CONTAINER], check=False, timeout=60)


def create_container(root: Path, depth: int) -> str:
    served = f"qwen27_hybrid_depth{depth}"
    result = command(
        [
            "docker",
            "create",
            "--device",
            "nvidia.com/gpu=all",
            "--name",
            CONTAINER,
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--pids-limit",
            "4096",
            "--shm-size",
            "16g",
            "-p",
            "127.0.0.1:8000:8000",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,exec,size=8g",
            "-e",
            "HOME=/cache",
            "-e",
            "HF_HOME=/models",
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
            f"DEPTH={depth}",
            "-e",
            f"SERVED={served}",
            "-v",
            "/home/workbench/.cache/huggingface:/models:ro",
            "-v",
            f"{root / 'serve_depth_matrix.sh'}:/workspace/serve.sh:ro",
            "--entrypoint",
            "bash",
            IMAGE,
            "/workspace/serve.sh",
        ],
        timeout=120,
    )
    return result.stdout.strip()


def container_state() -> str:
    result = command(
        ["docker", "inspect", "--format", "{{.State.Status}}/{{.State.ExitCode}}", CONTAINER],
        check=False,
        timeout=30,
    )
    return result.stdout.strip() or result.stderr.strip()


def wait_ready(timeout_seconds: int = 600) -> tuple[bool, float, str]:
    started = time.monotonic()
    deadline = started + timeout_seconds
    last_state = "unknown"
    while time.monotonic() < deadline:
        last_state = container_state()
        if last_state.startswith("exited") or last_state.startswith("dead"):
            return False, time.monotonic() - started, last_state
        try:
            with urllib.request.urlopen(BASE_URL + "/health", timeout=2) as response:
                if response.status == 200:
                    return True, time.monotonic() - started, last_state
        except (OSError, TimeoutError):
            pass
        time.sleep(2)
    return False, time.monotonic() - started, last_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", type=int, nargs="+", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    try:
        for depth in args.depths:
            result_path = args.output / f"depth-{depth}" / "result.json"
            if args.resume and result_path.is_file():
                summary.append(json.loads(result_path.read_text()))
                print(f"depth {depth}: retained existing result", flush=True)
                continue

            print(f"depth {depth}: creating fresh container", flush=True)
            remove_container()
            container_id = create_container(args.root, depth)
            command(["docker", "start", CONTAINER], timeout=60)
            ready, startup_seconds, state = wait_ready()
            print(
                f"depth {depth}: ready={ready} startup_seconds={startup_seconds:.3f} state={state}",
                flush=True,
            )
            depth_dir = args.output / f"depth-{depth}"
            depth_dir.mkdir(parents=True, exist_ok=True)
            if not ready:
                logs = command(["docker", "logs", CONTAINER], check=False, timeout=60)
                (depth_dir / "server.log").write_text(logs.stdout + logs.stderr)
                (depth_dir / "startup-failure.json").write_text(
                    json.dumps(
                        {
                            "depth": depth,
                            "container_id": container_id,
                            "startup_seconds": startup_seconds,
                            "state": state,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                raise RuntimeError(f"depth {depth} failed startup: {state}")

            benchmark = command(
                [
                    "python3",
                    str(args.root / "run_depth_candidate.py"),
                    "--depth",
                    str(depth),
                    "--root",
                    str(args.root),
                    "--output",
                    str(args.output),
                ],
                check=False,
                timeout=1800,
            )
            print(benchmark.stdout, end="", flush=True)
            if benchmark.stderr:
                print(benchmark.stderr, end="", flush=True)
            if benchmark.returncode not in (0, 2):
                raise RuntimeError(
                    f"depth {depth} benchmark exited {benchmark.returncode}"
                )

            result = json.loads(result_path.read_text())
            result["container_id"] = container_id
            result["startup_seconds"] = startup_seconds
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            summary.append(result)

            logs = command(["docker", "logs", CONTAINER], check=False, timeout=60)
            (depth_dir / "server.log").write_text(logs.stdout + logs.stderr)
            command(["docker", "stop", "--time", "10", CONTAINER], check=False, timeout=60)
            remove_container()
    finally:
        remove_container()

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
