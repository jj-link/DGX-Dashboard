#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "http://127.0.0.1:8000"
BLOCK_SIZE_RE = re.compile(r"Setting attention block size to ([0-9]+) tokens")


def command(
    args: list[str],
    *,
    check: bool = True,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        text=True,
        capture_output=True,
        check=check,
        timeout=timeout,
    )


def container_name(label: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", label)
    return f"qwen27-multiboundary-{safe}"


def remove_container(name: str) -> None:
    command(["docker", "rm", "--force", name], check=False, timeout=120)


def create_container(
    *,
    name: str,
    image: str,
    root: Path,
    depth: int,
    served: str,
    spec_method: str,
    kv_cache_dtype: str,
    max_num_batched_tokens: int,
    triton_force_first: bool,
    flashinfer_autotune: bool,
    use_v2_model_runner: bool,
) -> str:
    result = command(
        [
            "docker",
            "create",
            "--device",
            "nvidia.com/gpu=all",
            "--name",
            name,
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
            *(
                []
                if use_v2_model_runner
                else ["-e", "VLLM_USE_V2_MODEL_RUNNER=0"]
            ),
            "-e",
            f"VLLM_TRITON_FORCE_FIRST_CONFIG={int(triton_force_first)}",
            "-e",
            f"FLASHINFER_AUTOTUNE={int(flashinfer_autotune)}",
            "-e",
            f"DEPTH={depth}",
            "-e",
            f"SERVED={served}",
            "-e",
            f"SPEC_METHOD={spec_method}",
            "-e",
            f"KV_CACHE_DTYPE={kv_cache_dtype}",
            "-e",
            f"MAX_NUM_BATCHED={max_num_batched_tokens}",
            "-v",
            "/home/workbench/.cache/huggingface:/models:ro",
            "-v",
            f"{root / 'serve_depth_matrix.sh'}:/workspace/serve.sh:ro",
            "--entrypoint",
            "bash",
            image,
            "/workspace/serve.sh",
        ],
        timeout=120,
    )
    return result.stdout.strip()


def state(name: str) -> str:
    result = command(
        ["docker", "inspect", "--format", "{{.State.Status}}/{{.State.ExitCode}}", name],
        check=False,
        timeout=30,
    )
    return result.stdout.strip() or result.stderr.strip()


def logs(name: str) -> str:
    result = command(["docker", "logs", name], check=False, timeout=60)
    return result.stdout + result.stderr


def health() -> bool:
    try:
        with urllib.request.urlopen(BASE_URL + "/health", timeout=2) as response:
            return response.status == 200
    except (OSError, TimeoutError, urllib.error.URLError):
        return False


def wait_ready(name: str, timeout_seconds: int) -> tuple[bool, float, str, str]:
    started = time.monotonic()
    deadline = started + timeout_seconds
    latest_logs = ""
    latest_state = "unknown"
    while time.monotonic() < deadline:
        latest_state = state(name)
        latest_logs = logs(name)
        if latest_state.startswith(("exited", "dead")):
            return False, time.monotonic() - started, latest_state, latest_logs
        if health() and BLOCK_SIZE_RE.search(latest_logs):
            return True, time.monotonic() - started, latest_state, latest_logs
        time.sleep(2)
    return False, time.monotonic() - started, latest_state, latest_logs


def start_fresh(
    *,
    name: str,
    image: str,
    root: Path,
    depth: int,
    served: str,
    spec_method: str,
    timeout_seconds: int,
    kv_cache_dtype: str,
    max_num_batched_tokens: int,
    triton_force_first: bool,
    flashinfer_autotune: bool,
    use_v2_model_runner: bool,
    attempts: int,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        remove_container(name)
        container_id = create_container(
            name=name,
            image=image,
            root=root,
            depth=depth,
            served=served,
            spec_method=spec_method,
            kv_cache_dtype=kv_cache_dtype,
            max_num_batched_tokens=max_num_batched_tokens,
            triton_force_first=triton_force_first,
            flashinfer_autotune=flashinfer_autotune,
            use_v2_model_runner=use_v2_model_runner,
        )
        command(["docker", "start", name], timeout=60)
        ready, elapsed, final_state, server_log = wait_ready(name, timeout_seconds)
        if ready:
            block_matches = BLOCK_SIZE_RE.findall(server_log)
            return {
                "ready": True,
                "attempt": attempt,
                "container_id": container_id,
                "startup_seconds": elapsed,
                "state": final_state,
                "block_size": int(block_matches[-1]),
                "kv_cache_dtype": kv_cache_dtype,
                "max_num_batched_tokens": max_num_batched_tokens,
                "failures": failures,
                "server_log": server_log,
            }
        failures.append(
            {
                "attempt": attempt,
                "container_id": container_id,
                "startup_seconds": elapsed,
                "state": final_state,
                "server_log": server_log,
            }
        )
        remove_container(name)
        time.sleep(15)
    return {"ready": False, "failures": failures}


def inspect_image(image: str) -> dict[str, Any]:
    result = command(["docker", "image", "inspect", image], timeout=60)
    return json.loads(result.stdout)[0]


def run_candidate(
    *,
    root: Path,
    corpus: Path,
    output: Path,
    runtime_label: str,
    depth: int,
    block_size: int,
    request_limit: int | None,
    request_start: int,
    task_sequence: list[str] | None,
    candidate_script: str,
    candidate_disable_thinking: bool,
    candidate_thinking_token_budget: int | None,
    candidate_rounds: int | None,
) -> subprocess.CompletedProcess[str]:
    candidate_command = [
        sys.executable,
        str(root / candidate_script),
        "--depth",
        str(depth),
        "--block-size",
        str(block_size),
        "--corpus",
        str(corpus),
        "--output",
        str(output),
        "--runtime-label",
        runtime_label,
    ]
    if request_start:
        candidate_command.extend(["--request-start", str(request_start)])
    if request_limit is not None:
        candidate_command.extend(["--request-limit", str(request_limit)])
    if task_sequence:
        candidate_command.extend(["--task-sequence", *task_sequence])
    if candidate_rounds is not None:
        candidate_command.extend(["--rounds", str(candidate_rounds)])
    if candidate_disable_thinking:
        candidate_command.append("--disable-thinking")
    if candidate_thinking_token_budget is not None:
        candidate_command.extend(
            ["--thinking-token-budget", str(candidate_thinking_token_budget)]
        )
    return command(
        candidate_command,
        check=False,
        timeout=7200,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--runtime-label", required=True)
    parser.add_argument("--served-name")
    parser.add_argument("--depths", nargs="+", type=int, required=True)
    parser.add_argument(
        "--candidate-script", default="run_multiboundary_candidate.py"
    )
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument("--max-num-batched-tokens", type=int, default=8352)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--startup-attempts", type=int, default=2)
    parser.add_argument("--request-limit", type=int)
    parser.add_argument("--request-start", type=int, default=0)
    parser.add_argument("--candidate-task-sequence", nargs="+")
    parser.add_argument("--candidate-rounds", type=int)
    parser.add_argument("--spec-method", choices=("dflash", "mtp"), default="dflash")
    parser.add_argument("--candidate-disable-thinking", action="store_true")
    parser.add_argument("--candidate-thinking-token-budget", type=int)
    parser.add_argument("--triton-force-first", action="store_true")
    parser.add_argument("--disable-flashinfer-autotune", action="store_true")
    parser.add_argument("--use-v1-model-runner", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    image_metadata = inspect_image(args.image)
    (args.output / "image.json").write_text(
        json.dumps(image_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary: list[dict[str, Any]] = []

    try:
        for depth in args.depths:
            depth_root = args.output / f"depth-{depth}"
            depth_root.mkdir(parents=True, exist_ok=False)
            name = container_name(f"{args.runtime_label}-depth-{depth}")
            served = args.served_name or f"qwen27_{args.runtime_label}_multiboundary_depth{depth}"
            startup = start_fresh(
                name=name,
                image=args.image,
                root=args.root,
                depth=depth,
                served=served,
                spec_method=args.spec_method,
                kv_cache_dtype=args.kv_cache_dtype,
                max_num_batched_tokens=args.max_num_batched_tokens,
                timeout_seconds=args.startup_timeout,
                triton_force_first=args.triton_force_first,
                flashinfer_autotune=not args.disable_flashinfer_autotune,
                attempts=args.startup_attempts,
                use_v2_model_runner=not args.use_v1_model_runner,
            )
            startup_log = startup.pop("server_log", "")
            (depth_root / "startup.json").write_text(
                json.dumps(startup, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (depth_root / "server-startup.log").write_text(
                startup_log, encoding="utf-8"
            )
            if not startup["ready"]:
                summary.append(
                    {"depth": depth, "status": "startup-failed", "startup": startup}
                )
                print(f"depth {depth}: startup failed", flush=True)
                continue

            process = run_candidate(
                root=args.root,
                corpus=args.corpus,
                output=depth_root / "run",
                runtime_label=args.runtime_label,
                depth=depth,
                block_size=startup["block_size"],
                candidate_script=args.candidate_script,
                request_limit=args.request_limit,
                request_start=args.request_start,
                task_sequence=args.candidate_task_sequence,
                candidate_disable_thinking=args.candidate_disable_thinking,
                candidate_thinking_token_budget=args.candidate_thinking_token_budget,
                candidate_rounds=args.candidate_rounds,
            )
            final_log = logs(name)
            (depth_root / "server-full.log").write_text(final_log, encoding="utf-8")
            process_record = {
                "returncode": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
            (depth_root / "candidate-process.json").write_text(
                json.dumps(process_record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result_path = depth_root / "run" / "result.json"
            result = json.loads(result_path.read_text()) if result_path.is_file() else None
            summary.append(
                {
                    "depth": depth,
                    "status": "completed" if result is not None else "candidate-failed",
                    "startup": startup,
                    "candidate_returncode": process.returncode,
                    "result": result,
                }
            )
            if result is None:
                print(
                    f"depth {depth}: candidate failed rc={process.returncode}", flush=True
                )
            else:
                print(
                    f"depth {depth}: quality "
                    f"{result['quality_pass_count']}/{result['quality_total']} "
                    f"multi_boundary={result.get('all_requests_crossed_minimum_boundaries', 'n/a')}",
                    flush=True,
                )
            remove_container(name)
            time.sleep(10)
    finally:
        for depth in args.depths:
            remove_container(container_name(f"{args.runtime_label}-depth-{depth}"))

    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failed_execution = any(item["status"] != "completed" for item in summary)
    return 1 if failed_execution else 0


if __name__ == "__main__":
    raise SystemExit(main())
