#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
SPEC_METRIC_RE = re.compile(
    r'^(vllm:spec_decode_(?:num_drafts|num_draft_tokens|num_accepted_tokens)(?:_per_pos)?_total)'
    r'\{([^}]*)\}\s+([0-9.eE+-]+)$'
)
POSITION_RE = re.compile(r'position="(\d+)"')


class GpuMemoryPoller:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.samples: list[int] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                values = [int(line.strip()) for line in output.splitlines() if line.strip()]
                if values:
                    self.samples.append(max(values))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop_event.wait(0.1)

    def __enter__(self) -> GpuMemoryPoller:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    @property
    def peak_mib(self) -> int | None:
        return max(self.samples) if self.samples else None


def fetch_json(path: str) -> dict[str, Any]:
    with urllib.request.urlopen(BASE_URL + path, timeout=30) as response:
        return json.loads(response.read())


def fetch_text(path: str) -> str:
    with urllib.request.urlopen(BASE_URL + path, timeout=30) as response:
        return response.read().decode("utf-8")


def parse_spec_metrics(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "draft_steps": 0,
        "draft_tokens": 0,
        "accepted_tokens": 0,
        "accepted_per_position": {},
    }
    for line in text.splitlines():
        match = SPEC_METRIC_RE.match(line)
        if match is None:
            continue
        name, labels, raw_value = match.groups()
        value = int(float(raw_value))
        if name.endswith("num_drafts_total"):
            result["draft_steps"] += value
        elif name.endswith("num_draft_tokens_total"):
            result["draft_tokens"] += value
        elif name.endswith("num_accepted_tokens_total"):
            result["accepted_tokens"] += value
        elif name.endswith("num_accepted_tokens_per_pos_total"):
            position_match = POSITION_RE.search(labels)
            if position_match is not None:
                position = position_match.group(1)
                result["accepted_per_position"][position] = (
                    result["accepted_per_position"].get(position, 0) + value
                )
    return result


def subtract_metrics(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    positions = set(after["accepted_per_position"]) | set(
        before["accepted_per_position"]
    )
    delta = {
        key: after[key] - before[key]
        for key in ("draft_steps", "draft_tokens", "accepted_tokens")
    }
    delta["accepted_per_position"] = {
        position: after["accepted_per_position"].get(position, 0)
        - before["accepted_per_position"].get(position, 0)
        for position in sorted(positions, key=int)
    }
    drafts = delta["draft_tokens"]
    steps = delta["draft_steps"]
    accepted = delta["accepted_tokens"]
    delta["acceptance_rate"] = accepted / drafts if drafts else None
    delta["accepted_tokens_per_draft_step"] = accepted / steps if steps else None
    return delta


def run_chat_sse(payload: dict[str, Any], raw_path: Path) -> dict[str, Any]:
    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    first_token_at: float | None = None
    raw_lines: list[bytes] = []
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        status = response.status
        for raw_line in response:
            raw_lines.append(raw_line)
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                if first_token_at is None and (content or reasoning):
                    first_token_at = time.perf_counter()
                text_parts.append(content)
                reasoning_parts.append(reasoning)
                finish_reason = choice.get("finish_reason") or finish_reason
    ended = time.perf_counter()
    raw_path.write_bytes(b"".join(raw_lines))
    text = "".join(text_parts)
    completion_tokens = usage.get("completion_tokens", 0) if usage else 0
    decode_duration = ended - first_token_at if first_token_at is not None else None
    return {
        "http_status": status,
        "finish_reason": finish_reason,
        "usage": usage,
        "ttft_seconds": first_token_at - started if first_token_at is not None else None,
        "decode_duration_seconds": decode_duration,
        "decode_tokens_per_second": (
            max(completion_tokens - 1, 0) / decode_duration
            if decode_duration and completion_tokens
            else None
        ),
        "e2e_latency_seconds": ended - started,
        "e2e_tokens_per_second": (
            completion_tokens / (ended - started) if completion_tokens else None
        ),
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "output_bytes": len(text.encode("utf-8")),
        "text": text,
        "reasoning": "".join(reasoning_parts),
    }


def validate_structured_warmup(text: str) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"passed": False, "failure_reasons": [f"invalid JSON: {exc}"]}
    if set(value) != {"bug", "line", "fix", "regression_test"}:
        reasons.append(f"unexpected keys: {sorted(value)}")
    if value.get("line") != 4:
        reasons.append(f"line={value.get('line')!r}, expected 4")
    if "parts[1]" not in str(value.get("fix")):
        reasons.append("fix does not read hard limit from parts[1]")
    regression = str(value.get("regression_test"))
    for expected in ("10:20", "10", "20"):
        if expected not in regression:
            reasons.append(f"regression_test omits {expected!r}")
    return {"passed": not reasons, "failure_reasons": reasons}


def extract_python_source(text: str) -> tuple[str | None, list[str]]:
    stripped = text.strip()
    if stripped.startswith("```python"):
        if not stripped.endswith("```"):
            return None, ["Python fence is not closed"]
        source = stripped[len("```python") : -3].lstrip("\r\n")
        if "```" in source:
            return None, ["response contains multiple Markdown fences"]
        return source, []
    if "```" in stripped:
        return None, ["response contains malformed or multiple Markdown fences"]
    return stripped, []


def external_contract_source(generated_path: Path) -> str:
    return f'''import asyncio
import importlib.util

spec = importlib.util.spec_from_file_location("generated", {str(generated_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

async def main():
    active = 0
    peak = 0
    async def ordered_worker(value):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep((5 - value) * 0.002)
            return value * 10
        finally:
            active -= 1
    result = await module.map_ordered([1, 2, 3, 4], ordered_worker, 2)
    assert result == [10, 20, 30, 40], result
    assert peak == 2, peak

    for args in ((None, ordered_worker, 1), ([], None, 1), ([], ordered_worker, True), ([], ordered_worker, 0)):
        try:
            await module.map_ordered(*args)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"validation accepted {{args!r}}")

    sibling_started = [asyncio.Event(), asyncio.Event()]
    sibling_finished = [asyncio.Event(), asyncio.Event()]
    sentinel = RuntimeError("sentinel-first-error")
    async def failing_worker(value):
        if value == 0:
            await sibling_started[0].wait()
            await sibling_started[1].wait()
            raise sentinel
        slot = value - 1
        sibling_started[slot].set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_finished[slot].set()

    try:
        await module.map_ordered([0, 1, 2], failing_worker, 3)
    except RuntimeError as exc:
        assert exc is sentinel, (exc, sentinel)
    else:
        raise AssertionError("worker exception did not propagate")
    assert all(event.is_set() for event in sibling_finished), "cancelled siblings were not awaited"
    await asyncio.sleep(0)
    leaked = [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]
    assert not leaked, leaked

asyncio.run(main())
'''


def run_subprocess(
    command: list[str], *, cwd: Path, timeout: float
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
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


def validate_agent_implementation(text: str, validation_dir: Path) -> dict[str, Any]:
    validation_dir.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    source, extraction_reasons = extract_python_source(text)
    reasons.extend(extraction_reasons)
    if source is None:
        return {"passed": False, "failure_reasons": reasons}

    generated_path = validation_dir / "generated.py"
    generated_path.write_text(source)
    try:
        tree = ast.parse(source, filename=str(generated_path))
    except SyntaxError as exc:
        reasons.append(f"generated source syntax error at line {exc.lineno}: {exc.msg}")
        return {"passed": False, "failure_reasons": reasons}

    functions = [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)]
    if not any(node.name == "map_ordered" for node in functions):
        reasons.append("map_ordered async function is missing")
    test_methods = [
        node
        for node in functions
        if node.name.startswith("test_") and any(
            isinstance(parent, ast.ClassDef)
            and node in parent.body
            and any(
                isinstance(base, ast.Attribute)
                and base.attr == "IsolatedAsyncioTestCase"
                for base in parent.bases
            )
            for parent in ast.walk(tree)
        )
    ]
    if len(test_methods) < 5:
        reasons.append(f"found {len(test_methods)} isolated async tests, expected at least 5")

    generated_test = run_subprocess(
        [sys.executable, str(generated_path)], cwd=validation_dir, timeout=35
    )
    (validation_dir / "generated-test.json").write_text(
        json.dumps(generated_test, indent=2, sort_keys=True) + "\n"
    )
    if generated_test["timed_out"]:
        reasons.append("generated tests timed out after 35 seconds")
    elif generated_test["exit_code"] != 0:
        detail = (generated_test["stderr"] or generated_test["stdout"]).strip()
        reasons.append(f"generated tests exited {generated_test['exit_code']}: {detail[-1000:]}")

    external_path = validation_dir / "external_contract.py"
    external_path.write_text(external_contract_source(generated_path))
    external_test = run_subprocess(
        [sys.executable, str(external_path)], cwd=validation_dir, timeout=20
    )
    (validation_dir / "external-contract.json").write_text(
        json.dumps(external_test, indent=2, sort_keys=True) + "\n"
    )
    if external_test["timed_out"]:
        reasons.append("external contract test timed out after 20 seconds")
    elif external_test["exit_code"] != 0:
        detail = (external_test["stderr"] or external_test["stdout"]).strip()
        reasons.append(f"external contract exited {external_test['exit_code']}: {detail[-1000:]}")

    return {
        "passed": not reasons,
        "failure_reasons": reasons,
        "test_method_count": len(test_methods),
        "generated_test": generated_test,
        "external_contract": external_test,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = fetch_json("/v1/models")["data"][0]
    model_id = model["id"]
    warmup_payload = json.loads(
        (args.root / "depth11-original-repro/raw/depth11-repro-1.request.json").read_text()
    )
    measured_payload = json.loads(
        (
            args.root
            / "depth11-original-repro/raw/depth11-agent-implementation-1.request.json"
        ).read_text()
    )
    warmup_payload["model"] = model_id
    measured_payload["model"] = model_id

    output_dir = args.output / f"depth-{args.depth}"
    raw_dir = output_dir / "raw"
    outputs_dir = output_dir / "outputs"
    validation_root = output_dir / "validation"
    for directory in (raw_dir, outputs_dir, validation_root):
        directory.mkdir(parents=True, exist_ok=True)

    metrics_before = parse_spec_metrics(fetch_text("/metrics"))
    with GpuMemoryPoller() as memory:
        warmup = run_chat_sse(warmup_payload, raw_dir / "warmup.response.sse")
        (outputs_dir / "warmup.txt").write_text(warmup["text"])
        warmup["quality"] = validate_structured_warmup(warmup["text"])
        warmup.pop("text")
        warmup.pop("reasoning")

        requests: list[dict[str, Any]] = []
        workload_started = time.perf_counter()
        for repetition in range(1, 4):
            result = run_chat_sse(
                measured_payload,
                raw_dir / f"agent-{repetition}.response.sse",
            )
            text = result.pop("text")
            reasoning = result.pop("reasoning")
            (outputs_dir / f"agent-{repetition}.txt").write_text(text)
            (outputs_dir / f"agent-{repetition}.reasoning.txt").write_text(reasoning)
            result["quality"] = validate_agent_implementation(
                text, validation_root / f"agent-{repetition}"
            )
            result["repetition"] = repetition
            requests.append(result)
        workload_wall_seconds = time.perf_counter() - workload_started
    metrics_after = parse_spec_metrics(fetch_text("/metrics"))
    metrics_delta = subtract_metrics(metrics_after, metrics_before)

    completion_tokens = sum(
        request["usage"]["completion_tokens"] for request in requests
    )
    decode_seconds = sum(
        request["decode_duration_seconds"] or 0.0 for request in requests
    )
    e2e_seconds = sum(request["e2e_latency_seconds"] for request in requests)
    result = {
        "schema_version": 1,
        "depth": args.depth,
        "served_model": model_id,
        "max_model_len": model.get("max_model_len"),
        "warmup": warmup,
        "requests": requests,
        "quality_pass_count": sum(
            request["quality"]["passed"] for request in requests
        ),
        "quality_total": len(requests),
        "all_quality_passed": all(
            request["quality"]["passed"] for request in requests
        ),
        "workload_wall_seconds": workload_wall_seconds,
        "aggregate_decode_tokens_per_second": (
            max(completion_tokens - len(requests), 0) / decode_seconds
            if decode_seconds
            else None
        ),
        "aggregate_e2e_tokens_per_second": (
            completion_tokens / e2e_seconds if e2e_seconds else None
        ),
        "speculative_metrics": metrics_delta,
        "peak_gpu_memory_mib": memory.peak_mib,
        "artifacts": {
            "output_dir": str(output_dir),
            "fixed_payload_sha256": hashlib.sha256(
                json.dumps(
                    {key: value for key, value in measured_payload.items() if key != "model"},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result_path)
    print(
        json.dumps(
            {
                "depth": args.depth,
                "quality": f"{result['quality_pass_count']}/{result['quality_total']}",
                "wall_seconds": workload_wall_seconds,
                "decode_tps": result["aggregate_decode_tokens_per_second"],
                "e2e_tps": result["aggregate_e2e_tokens_per_second"],
                "speculative_metrics": metrics_delta,
                "peak_gpu_memory_mib": memory.peak_mib,
            },
            sort_keys=True,
        )
    )
    return 0 if result["all_quality_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
