#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def load_sweep_module(root: Path):
    spec = importlib.util.spec_from_file_location(
        "depth_sweep_helpers", root / "run_depth_sweep.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument(
        "--image", default="vllm:v0.24.0-hybrid-state-commit-fix"
    )
    parser.add_argument("--container", default="qwen27-hybrid-state-fix")
    args = parser.parse_args()

    sweep = load_sweep_module(args.root)
    sweep.IMAGE = args.image
    sweep.CONTAINER = args.container
    sweep.remove_container()
    container_id = sweep.create_container(
        args.root / "fixed-state-launch", args.depth
    )
    subprocess.run(
        ["docker", "start", args.container],
        check=True,
        text=True,
        capture_output=True,
        timeout=120,
    )
    ready, startup_seconds, state = sweep.wait_ready()
    if not ready:
        logs = sweep.command(
            ["docker", "logs", args.container], check=False, timeout=60
        )
        sys.stderr.write(logs.stdout)
        sys.stderr.write(logs.stderr)
        raise RuntimeError(
            f"fixed-state server did not become ready: state={state}"
        )

    candidate = subprocess.run(
        [
            sys.executable,
            str(args.root / "run_depth_candidate.py"),
            "--depth",
            str(args.depth),
            "--root",
            str(args.root),
            "--output",
            str(args.output),
        ],
        text=True,
        capture_output=True,
        timeout=1800,
    )
    result_path = args.output / f"depth-{args.depth}" / "result.json"
    if not result_path.exists():
        sys.stderr.write(candidate.stdout)
        sys.stderr.write(candidate.stderr)
        raise RuntimeError(
            f"fixed-state candidate failed without result: exit={candidate.returncode}"
        )

    result = json.loads(result_path.read_text())
    result["container_id"] = container_id
    result["container_name"] = args.container
    result["image"] = args.image
    result["startup_seconds"] = startup_seconds
    result["candidate_exit_code"] = candidate.returncode
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    logs = sweep.command(
        ["docker", "logs", args.container], check=False, timeout=60
    )
    (result_path.parent / "server.log").write_text(logs.stdout + logs.stderr)
    print(result_path, flush=True)
    print(candidate.stdout, end="", flush=True)
    if candidate.stderr:
        print(candidate.stderr, file=sys.stderr, end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
