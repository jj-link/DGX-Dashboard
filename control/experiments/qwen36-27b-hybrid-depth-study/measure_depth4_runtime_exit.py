#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
import traceback
from typing import Any

from run_depth_candidate import (
    GpuMemoryPoller,
    fetch_text,
    parse_spec_metrics,
    run_chat_sse,
    subtract_metrics,
)
from run_multiboundary_candidate import warmup_payload
from run_multiboundary_sweep import logs, remove_container, start_fresh


NUM_COMPUTED_RE = re.compile(r"num_computed_tokens=\[(\d+)\]")
NUM_OUTPUT_RE = re.compile(r"num_output_tokens=\[(\d+)\]")


def last_int(pattern: re.Pattern[str], text: str) -> int | None:
    matches = pattern.findall(text)
    return int(matches[-1]) if matches else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--startup-timeout", type=int, default=600)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    request_template = json.loads((args.corpus / "front.request.json").read_text())
    records: list[dict[str, Any]] = []

    for repetition in range(1, args.repetitions + 1):
        repetition_root = args.output / f"repetition-{repetition}"
        repetition_root.mkdir()
        name = f"qwen27-depth4-exit-measurement-{repetition}"
        served = f"qwen27_depth4_exit_measurement_{repetition}"
        startup = start_fresh(
            name=name,
            image=args.image,
            root=args.root,
            depth=4,
            served=served,
            timeout_seconds=args.startup_timeout,
            attempts=2,
        )
        startup_log = startup.pop("server_log", "")
        (repetition_root / "startup.json").write_text(
            json.dumps(startup, indent=2, sort_keys=True) + "\n"
        )
        (repetition_root / "server-startup.log").write_text(startup_log)
        if not startup["ready"]:
            records.append(
                {
                    "repetition": repetition,
                    "startup": startup,
                    "status": "startup-failed",
                }
            )
            remove_container(name)
            continue

        payload = dict(request_template)
        payload["model"] = served
        before = parse_spec_metrics(fetch_text("/metrics"))
        request_result: dict[str, Any] | None = None
        request_exception: str | None = None
        with GpuMemoryPoller() as memory:
            run_chat_sse(
                warmup_payload(served),
                repetition_root / "warmup.response.sse",
            )
            before = parse_spec_metrics(fetch_text("/metrics"))
            try:
                request_result = run_chat_sse(
                    payload,
                    repetition_root / "front.response.sse",
                )
            except Exception:
                request_exception = traceback.format_exc()
            time.sleep(0.2)
        try:
            after = parse_spec_metrics(fetch_text("/metrics"))
            metrics = subtract_metrics(after, before)
        except Exception:
            metrics = {
                "metrics_error": traceback.format_exc(),
                "draft_steps": None,
                "draft_tokens": None,
                "accepted_tokens": None,
                "accepted_per_position": {},
                "acceptance_rate": None,
                "accepted_tokens_per_draft_step": None,
            }

        server_log = logs(name)
        (repetition_root / "server-full.log").write_text(server_log)
        num_computed = last_int(NUM_COMPUTED_RE, server_log)
        num_output = last_int(NUM_OUTPUT_RE, server_log)
        prompt_tokens = (
            num_computed - num_output + 1
            if num_computed is not None and num_output is not None
            else None
        )
        block_size = int(startup["block_size"])
        boundaries_crossed = (
            max(
                (prompt_tokens + num_output - 1) // block_size
                - max(prompt_tokens - 1, 0) // block_size,
                0,
            )
            if prompt_tokens is not None and num_output is not None
            else None
        )
        record = {
            "repetition": repetition,
            "status": "request-failed" if request_exception else "request-completed",
            "startup": startup,
            "request_exception": request_exception,
            "request_result": request_result,
            "speculative_metrics": metrics,
            "peak_gpu_memory_mib": memory.peak_mib,
            "scheduler_dump_num_computed_tokens": num_computed,
            "scheduler_dump_num_output_tokens": num_output,
            "inferred_prompt_tokens": prompt_tokens,
            "inferred_decode_boundaries_crossed": boundaries_crossed,
            "device_assert_observed": "device-side assert triggered" in server_log,
        }
        (repetition_root / "result.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        records.append(record)
        remove_container(name)
        time.sleep(10)

    summary = {
        "schema_version": 1,
        "depth": 4,
        "image": args.image,
        "corpus": str(args.corpus),
        "records": records,
        "runtime_exit_count": sum(
            int(record.get("device_assert_observed", False)) for record in records
        ),
        "three_boundary_measurement_count": sum(
            int((record.get("inferred_decode_boundaries_crossed") or 0) >= 3)
            for record in records
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(args.output / "summary.json")
    return 0 if (
        summary["runtime_exit_count"] == args.repetitions
        and summary["three_boundary_measurement_count"] == args.repetitions
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
