#!/usr/bin/env python3
"""Quality-gated concurrency-1 speculative decoding benchmark for Qwen3.6 35B A3B."""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

EXPERIMENT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
WORKLOAD_MANIFEST = EXPERIMENT_ROOT / "workload-manifest.json"
IMAGE = "vllm/vllm-openai@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f"
IMAGE_ID = "sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f"
TARGET = "/models/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/11fb1afff493ad0d94c808ceecdf73b8215ccc7a"
TARGET_REVISION = "11fb1afff493ad0d94c808ceecdf73b8215ccc7a"
DRAFTER = "/models/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/42d3b34d588423cdae7ba8f53a8cf7789346a719"
DRAFTER_REVISION = "42d3b34d588423cdae7ba8f53a8cf7789346a719"
SERVED = "benchmark_qwen36_35b_a3b_nvfp4_fast"
BASE_URL = "http://127.0.0.1:8000"
PORT = 8000
MAX_MODEL_LEN = 262_144
MAX_NUM_BATCHED_TOKENS = 8192
MAX_NUM_SEQS = 4
GPU_MEMORY_UTILIZATION = 0.92
MAMBA_SSM_CACHE_DTYPE = "float32"
MAMBA_CACHE_DTYPE = "float16"
SEED = 424242
TEMPERATURE = 0.0
TOP_P = 1.0
WARMUP_ROUNDS = 1
HTTP_TIMEOUT_SECONDS = 1800
STARTUP_TIMEOUT_SECONDS = 1200
GPU_TOTAL_MIB = 97_887
GPU_POLL_SECONDS = 0.25

ACCEPTED_COUNTER = "vllm:spec_decode_num_accepted_tokens_total"
DRAFT_COUNTER = "vllm:spec_decode_num_draft_tokens_total"
DRAFT_STEP_COUNTER = "vllm:spec_decode_num_drafts_total"
PER_POSITION_COUNTER = "vllm:spec_decode_num_accepted_tokens_per_pos_total"

COMMON_SYSTEM = (
    "You are a production coding agent. Follow the requested output contract exactly. "
    "Complete the task; do not stop mid-answer."
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def build_review_source() -> tuple[str, dict[str, int]]:
    lines: list[str] = [
        "from __future__ import annotations",
        "",
        "import asyncio",
        "from typing import Any, Awaitable, Callable",
        "",
    ]

    def benign(start: int, stop: int) -> None:
        for index in range(start, stop):
            lines.extend(
                [
                    f"def normalize_{index:03d}(value: str) -> str:",
                    f"    \"\"\"Normalize field {index:03d} without shared state.\"\"\"",
                    "    return value.strip()",
                    "",
                ]
            )

    benign(0, 80)
    lines.extend(
        [
            "async def load_order(conn: Any, tenant_id: str, order_id: int) -> Any:",
            "    query = f\"SELECT id, total FROM orders WHERE tenant_id = '{tenant_id}' AND id = {order_id}\"",
            "    return await conn.fetchrow(query)",
            "",
        ]
    )
    sql_line = len(lines) - 3
    benign(80, 160)
    lines.extend(
        [
            "async def cached_order(cache: Any, tenant_id: str, order_id: int, loader: Callable[..., Awaitable[Any]]) -> Any:",
            "    key = f\"order:{order_id}\"",
            "    cached = await cache.get(key)",
            "    if cached is not None:",
            "        return cached",
            "    value = await loader(tenant_id, order_id)",
            "    await cache.set(key, value, ttl=30)",
            "    return value",
            "",
        ]
    )
    cache_line = len(lines) - 8
    benign(160, 240)
    lines.extend(
        [
            "async def first_backend(backends: list[Callable[[Any], Awaitable[Any]]], request: Any) -> Any:",
            "    tasks = [asyncio.create_task(backend(request)) for backend in backends]",
            "    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)",
            "    for task in done:",
            "        try:",
            "            return task.result()",
            "        except Exception:",
            "            continue",
            "    return None",
            "",
        ]
    )
    sql_line = next(index for index, line in enumerate(lines, 1) if "query = f" in line)
    cache_line = next(index for index, line in enumerate(lines, 1) if "key = f" in line)
    async_line = next(index for index, line in enumerate(lines, 1) if "done, pending = await" in line)
    source = "\n".join(f"{number:04d}: {line}" for number, line in enumerate(lines, 1))
    return source, {"SQL_INJECTION": sql_line, "CACHE_TENANT_LEAK": cache_line, "ASYNC_TASK_LEAK": async_line}


def build_edit_source() -> tuple[str, str, int]:
    lines: list[str] = [
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "from enum import Enum",
        "from typing import Final",
        "",
        "DEFAULT_RETRIES: Final = 3",
        "",
        "class Mode(str, Enum):",
        "    SAFE = \"safe\"",
        "    FAST = \"fast\"",
        "",
        "@dataclass(frozen=True)",
        "class ClientConfig:",
        "    endpoint: str",
        "    timeout_ms: int",
        "    retries: int = DEFAULT_RETRIES",
        "    mode: Mode = Mode.SAFE",
        "",
        "    def validate(self) -> None:",
        "        if not self.endpoint.startswith((\"http://\", \"https://\")):",
        "            raise ValueError(\"endpoint must be HTTP(S)\")",
        "        if self.timeout_ms <= 0:",
        "            raise ValueError(\"timeout_ms must be positive\")",
        "        if self.retries < 0:",
        "            raise ValueError(\"retries cannot be negative\")",
        "",
        "def normalized_endpoint(endpoint: str) -> str:",
        "    return endpoint.rstrip(\"/\")",
        "",
    ]
    for index in range(24):
        lines.extend(
            [
                f"def header_{index:02d}(value: str) -> tuple[str, str]:",
                f"    return (\"x-client-field-{index:02d}\", value.strip())",
                "",
            ]
        )
    lines.extend(
        [
            "def request_timeout(config: ClientConfig) -> float:",
            "    config.validate()",
            "    timeout = max(config.timeout_ms / 100, 0.001)",
            "    return timeout",
            "",
            "def retry_delays(config: ClientConfig) -> list[float]:",
            "    config.validate()",
            "    return [0.05 * (2 ** attempt) for attempt in range(config.retries)]",
            "",
            "def describe(config: ClientConfig) -> str:",
            "    config.validate()",
            "    return f\"{config.mode.value}:{normalized_endpoint(config.endpoint)}:{config.retries}\"",
            "",
        ]
    )
    original = "\n".join(lines).rstrip("\n") + "\n"
    target = "    timeout = max(config.timeout_ms / 100, 0.001)"
    replacement = "    timeout = max(config.timeout_ms / 1000, 0.001)"
    assert original.count(target) == 1
    target_line = lines.index(target) + 1
    expected = original.replace(target, replacement)
    return original, expected, target_line


REVIEW_SOURCE, REVIEW_LINES = build_review_source()
EDIT_SOURCE, EDIT_EXPECTED, EDIT_TARGET_LINE = build_edit_source()


def json_schema(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


def common_payload(messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    return {
        "model": SERVED,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def task_specs() -> list[dict[str, Any]]:
    tool = {
        "type": "function",
        "function": {
            "name": "fetch_issue",
            "description": "Fetch one repository issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "number": {"type": "integer"},
                    "include_comments": {"type": "boolean"},
                },
                "required": ["repo", "number", "include_comments"],
                "additionalProperties": False,
            },
        },
    }
    tool_payload = common_payload(
        [
            {"role": "system", "content": COMMON_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Call fetch_issue exactly once for repository acme/widgets, issue 317, "
                    "with include_comments false. Return only the tool call."
                ),
            },
        ],
        256,
    )
    tool_payload.update(
        {
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": "fetch_issue"}},
            "parallel_tool_calls": False,
        }
    )
    tool_payload["chat_template_kwargs"] = {"enable_thinking": False}

    debug_schema = {
        "type": "object",
        "properties": {
            "bug": {"type": "string", "minLength": 20},
            "line": {"type": "integer"},
            "fix": {"type": "string", "minLength": 10},
            "regression_test": {"type": "string", "minLength": 20},
        },
        "required": ["bug", "line", "fix", "regression_test"],
        "additionalProperties": False,
    }
    debug_payload = common_payload(
        [
            {"role": "system", "content": COMMON_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Debug this numbered function and return one JSON object with exactly the keys "
                    "bug, line, fix, regression_test. line must be the defective line number; fix must "
                    "contain the corrected Python statement; regression_test must state concrete input "
                    "and expected soft/hard values.\n\n"
                    "1 def parse_limits(raw: str) -> dict[str, int]:\n"
                    "2     parts = raw.split(\":\")\n"
                    "3     soft = int(parts[0])\n"
                    "4     hard = int(parts[0])\n"
                    "5     return {\"soft\": soft, \"hard\": hard}"
                ),
            },
        ],
        512,
    )
    debug_payload["response_format"] = json_schema("debug_result", debug_schema)
    debug_payload["chat_template_kwargs"] = {"enable_thinking": False}

    implementation_prompt = """Return one complete Python 3.12 implementation as raw source or one Python fenced block:

    async def map_ordered(items, worker, limit)

Requirements:
- Validate items, worker, and limit before any early return. Reject bool/non-int limits and limits below 1.
- Run worker(item) with real concurrency bounded by limit and return results in input order.
- Propagate the first observed worker exception unchanged.
- On failure, explicitly cancel every unfinished sibling task and await every cancelled task before returning control. Merely setting an Event while already-running siblings continue is invalid.
- Include at least five deterministic unittest.IsolatedAsyncioTestCase tests covering validation, concurrency bound, ordering, first-error propagation, sibling cancellation, and completed cancellation cleanup.
- Include `if __name__ == "__main__": unittest.main()` so the tests execute.
- Use only the Python standard library.
"""
    implementation_payload = common_payload(
        [
            {"role": "system", "content": COMMON_SYSTEM},
            {"role": "user", "content": implementation_prompt},
        ],
        3000,
    )
    implementation_payload["chat_template_kwargs"] = {"enable_thinking": False}

    review_schema = {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "enum": ["SQL_INJECTION", "CACHE_TENANT_LEAK", "ASYNC_TASK_LEAK"],
                        },
                        "line": {"type": "integer"},
                        "failure_mode": {"type": "string", "minLength": 30},
                        "fix": {"type": "string", "minLength": 30},
                    },
                    "required": ["id", "line", "failure_mode", "fix"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["findings"],
        "additionalProperties": False,
    }
    review_payload = common_payload(
        [
            {"role": "system", "content": COMMON_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Review the entire numbered Python module below. It has exactly three independent "
                    "defect classes: SQL construction, tenant-aware caching, and asynchronous control "
                    "flow. Return JSON with exactly one finding for each required id: SQL_INJECTION, "
                    "CACHE_TENANT_LEAK, ASYNC_TASK_LEAK. For each, report the defective line and explain "
                    "the concrete failure mode and a complete fix. The async fix must cancel and await "
                    "every pending sibling and must not suppress backend failures. Do not report benign helpers.\n\n"
                    + REVIEW_SOURCE
                ),
            },
        ],
        1400,
    )
    review_payload["response_format"] = json_schema("review_findings", review_schema)
    review_payload["chat_template_kwargs"] = {"enable_thinking": False}

    reasoning_schema = {
        "type": "object",
        "properties": {
            "case_1": {
                "type": "object",
                "properties": {
                    "events": {"type": "array", "items": {"type": "string"}},
                    "cancelled_tasks": {"type": "array", "items": {"type": "string"}},
                    "exception": {"type": "string"},
                    "timeout_fired": {"type": "boolean"},
                    "explanation": {"type": "string", "minLength": 80},
                },
                "required": ["events", "cancelled_tasks", "exception", "timeout_fired", "explanation"],
                "additionalProperties": False,
            },
            "case_2": {
                "type": "object",
                "properties": {
                    "events": {"type": "array", "items": {"type": "string"}},
                    "cancelled_tasks": {"type": "array", "items": {"type": "string"}},
                    "exception": {"type": "string"},
                    "timeout_fired": {"type": "boolean"},
                    "explanation": {"type": "string", "minLength": 80},
                },
                "required": ["events", "cancelled_tasks", "exception", "timeout_fired", "explanation"],
                "additionalProperties": False,
            },
            "conclusion": {"type": "string", "minLength": 120},
        },
        "required": ["case_1", "case_2", "conclusion"],
        "additionalProperties": False,
    }
    reasoning_prompt = """Analyze this Python 3.12 program without running it. Return the requested JSON only.

async def child(name, delay, fail, events):
    try:
        await asyncio.sleep(delay)
        if fail:
            raise ValueError(name)
        events.append(f"{name}:ok")
    finally:
        events.append(f"{name}:finally")

async def scenario(b_delay):
    events = []
    caught = []
    try:
        async with asyncio.timeout(0.20):
            async with asyncio.TaskGroup() as group:
                group.create_task(child("A", 0.05, False, events))
                group.create_task(child("B", b_delay, True, events))
                group.create_task(child("C", 1.00, False, events))
    except* ValueError as errors:
        caught.extend(str(error) for error in errors.exceptions)
    return events, caught

Case 1 uses b_delay=0.10. Case 2 uses b_delay=0.30. Reason concisely without restating the prompt and complete the answer within the token limit. For each case, derive the observable event list, the exact task names cancelled before completion, the exception surfaced outside the context, whether the timeout wins, and why TaskGroup cleanup is complete before scenario exits. In case 2, account for the fact that TimeoutError cannot be caught by the shown except* ValueError. End with a substantive conclusion contrasting failure-driven sibling cancellation with deadline-driven cancellation.
"""
    reasoning_payload = common_payload(
        [
            {"role": "system", "content": COMMON_SYSTEM},
            {"role": "user", "content": reasoning_prompt},
        ],
        2000,
    )
    reasoning_payload["response_format"] = json_schema("reasoning_result", reasoning_schema)
    reasoning_payload["chat_template_kwargs"] = {"enable_thinking": False}

    edit_prompt = (
        "Return the complete Python file below with exactly one targeted fix and no other changes. "
        f"On line {EDIT_TARGET_LINE}, change the millisecond conversion divisor from 100 to 1000. "
        "Preserve every unrelated byte, include the entire file, and emit no Markdown fences or commentary.\n\n"
        + EDIT_SOURCE
    )
    edit_payload = common_payload(
        [
            {"role": "system", "content": COMMON_SYSTEM},
            {"role": "user", "content": edit_prompt},
        ],
        4000,
    )
    edit_payload["chat_template_kwargs"] = {"enable_thinking": False}

    return [
        {"id": "tool_routing", "payload": tool_payload, "validator": "tool"},
        {"id": "structured_debugging", "payload": debug_payload, "validator": "debug"},
        {"id": "agent_implementation", "payload": implementation_payload, "validator": "implementation"},
        {"id": "long_context_review", "payload": review_payload, "validator": "review"},
        {"id": "novel_reasoning", "payload": reasoning_payload, "validator": "reasoning"},
        {"id": "full_file_editing", "payload": edit_payload, "validator": "edit"},
    ]


def workload_manifest_value() -> dict[str, Any]:
    tasks = []
    for spec in task_specs():
        payload = spec["payload"]
        tasks.append(
            {
                "id": spec["id"],
                "validator": spec["validator"],
                "max_tokens": payload["max_tokens"],
                "request_sha256": sha256_bytes(canonical_json_bytes(payload)),
            }
        )
    return {
        "schema_version": 1,
        "concurrency": 1,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "request_order": [task["id"] for task in tasks],
        "warmup_rounds": WARMUP_ROUNDS,
        "warmup_request_order": [task["id"] for task in tasks],
        "warmup_is_full_corpus": True,
        "qualification_rounds": 3,
        "natural_stopping": True,
        "ignore_eos": False,
        "review_source_sha256": sha256_bytes(REVIEW_SOURCE.encode("utf-8")),
        "edit_source_sha256": sha256_bytes(EDIT_SOURCE.encode("utf-8")),
        "edit_expected_sha256": sha256_bytes(EDIT_EXPECTED.encode("utf-8")),
        "tasks": tasks,
        "fixed_server_invariants": fixed_invariants(),
    }


def ensure_workload_manifest() -> None:
    expected = workload_manifest_value()
    if WORKLOAD_MANIFEST.exists():
        actual = json.loads(WORKLOAD_MANIFEST.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(f"fixed workload manifest drifted: {WORKLOAD_MANIFEST}")
    else:
        write_json(WORKLOAD_MANIFEST, expected)


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def quality_result(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [f"{item['name']}: {item['detail']}" for item in checks if not item["passed"]]
    return {"passed": not failures, "checks": checks, "failure_reasons": failures}


def parse_json_content(response: dict[str, Any]) -> tuple[Any | None, str | None]:
    try:
        return json.loads(response.get("content") or ""), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"


def validate_tool(response: dict[str, Any], _: Path | None = None) -> dict[str, Any]:
    calls = response.get("tool_calls") or []
    checks = [
        check("natural_finish", response.get("finish_reason") in {"tool_calls", "stop"}, f"finish_reason={response.get('finish_reason')!r}"),
        check("one_tool_call", len(calls) == 1, f"count={len(calls)}"),
        check("no_prose", not (response.get("content") or "").strip(), f"content={response.get('content')!r}"),
    ]
    if len(calls) == 1:
        function = calls[0].get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "")
            argument_error = None
        except json.JSONDecodeError as exc:
            arguments = None
            argument_error = str(exc)
        expected = {"repo": "acme/widgets", "number": 317, "include_comments": False}
        checks.extend(
            [
                check("function_name", function.get("name") == "fetch_issue", f"name={function.get('name')!r}"),
                check("arguments_parse", argument_error is None, argument_error or "parsed"),
                check("arguments_exact", arguments == expected, f"arguments={arguments!r}"),
            ]
        )
    return quality_result(checks)


def validate_debug(response: dict[str, Any], _: Path | None = None) -> dict[str, Any]:
    value, error = parse_json_content(response)
    checks = [
        check("natural_finish", response.get("finish_reason") == "stop", f"finish_reason={response.get('finish_reason')!r}"),
        check("json_parse", error is None, error or "parsed"),
    ]
    if isinstance(value, dict):
        expected_keys = {"bug", "line", "fix", "regression_test"}
        bug = value.get("bug")
        fix = value.get("fix")
        test = value.get("regression_test")
        checks.extend(
            [
                check("exact_keys", set(value) == expected_keys, f"keys={sorted(value)}"),
                check("types", isinstance(bug, str) and type(value.get("line")) is int and isinstance(fix, str) and isinstance(test, str), f"types={ {key: type(item).__name__ for key, item in value.items()} }"),
                check("defective_line", value.get("line") == 4, f"line={value.get('line')!r}"),
                check("bug_semantics", isinstance(bug, str) and "hard" in bug.lower() and ("parts[0]" in bug or "first" in bug.lower()) and ("parts[1]" in bug or "second" in bug.lower()), f"bug={bug!r}"),
                check("fix_semantics", isinstance(fix, str) and "hard" in fix and "parts[1]" in fix, f"fix={fix!r}"),
                check("test_semantics", isinstance(test, str) and "10:20" in test and "10" in test and "20" in test and "soft" in test.lower() and "hard" in test.lower(), f"regression_test={test!r}"),
            ]
        )
    else:
        checks.append(check("json_object", False, f"value_type={type(value).__name__}"))
    return quality_result(checks)


CONTRACT_RUNNER = r'''import asyncio
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("candidate", Path(__file__).with_name("candidate.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
map_ordered = module.map_ordered

class FastFailure(RuntimeError):
    pass

class SlowFailure(RuntimeError):
    pass

async def main():
    async def identity(value):
        return value

    for bad in (0, -1, 1.5, True, None):
        try:
            await map_ordered([], identity, bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid limit accepted: {bad!r}")
    try:
        await map_ordered(None, identity, 1)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("None items accepted")
    try:
        await map_ordered([], None, 1)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("non-callable worker accepted")

    active = 0
    maximum = 0
    async def ordered_worker(value):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.01 * (6 - value))
            return value * 10
        finally:
            active -= 1
    ordered = await map_ordered(list(range(6)), ordered_worker, 2)
    assert ordered == [0, 10, 20, 30, 40, 50], ordered
    assert maximum == 2, maximum

    ready = asyncio.Event()
    started = set()
    cancelled = set()
    finalized = set()
    async def failing_worker(value):
        started.add(value)
        if len(started) == 3:
            ready.set()
        try:
            await ready.wait()
            if value == "fast":
                await asyncio.sleep(0.01)
                raise FastFailure("first-observed")
            if value == "slow":
                await asyncio.sleep(0.20)
                raise SlowFailure("later")
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.add(value)
            raise
        finally:
            finalized.add(value)
    try:
        await asyncio.wait_for(map_ordered(["fast", "peer", "slow"], failing_worker, 3), 2)
    except FastFailure as exc:
        assert str(exc) == "first-observed"
    else:
        raise AssertionError("first observed exception was not propagated unchanged")
    assert {"peer", "slow"}.issubset(cancelled), cancelled
    assert started == finalized == {"fast", "peer", "slow"}, (started, finalized)
    await asyncio.sleep(0)
    leaked = [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]
    assert not leaked, leaked

asyncio.run(main())
print("CONTRACT_OK")
'''


def validate_implementation(response: dict[str, Any], artifact_dir: Path | None = None) -> dict[str, Any]:
    raw_content = response.get("content") or ""
    fenced = re.fullmatch(r"\s*```(?:python)?\s*\n(.*)\n```\s*", raw_content, flags=re.DOTALL)
    content = ((fenced.group(1) if fenced else raw_content).strip("\n") + "\n")
    checks = [
        check("natural_finish", response.get("finish_reason") == "stop", f"finish_reason={response.get('finish_reason')!r}"),
        check("extractable_source", "```" not in raw_content or fenced is not None, "single Python block" if fenced else "raw source"),
    ]
    try:
        tree = ast.parse(content)
        parse_error = None
    except SyntaxError as exc:
        tree = None
        parse_error = str(exc)
    checks.append(check("python_parse", parse_error is None, parse_error or "parsed"))
    if tree is not None:
        functions = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "map_ordered"]
        test_methods = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")]
        checks.extend(
            [
                check("map_ordered_defined", len(functions) == 1, f"definitions={len(functions)}"),
                check("generated_tests_present", len(test_methods) >= 5, f"test_methods={len(test_methods)}"),
            ]
        )
    if tree is None or not checks[-2]["passed"]:
        return quality_result(checks)

    base = artifact_dir or Path(tempfile.mkdtemp(prefix="qwen-agent-validation-"))
    cleanup = artifact_dir is None
    validation_dir = base / "implementation-validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = validation_dir / "candidate.py"
    runner_path = validation_dir / "contract_runner.py"
    candidate_path.write_text(content, encoding="utf-8")
    runner_path.write_text(CONTRACT_RUNNER, encoding="utf-8")
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "HOME": str(validation_dir)}
    try:
        generated = subprocess.run(
            [sys.executable, "-I", str(candidate_path)],
            cwd=validation_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        generated_text = generated.stdout + generated.stderr
        match = re.search(r"Ran\s+(\d+)\s+tests?", generated_text)
        generated_count = int(match.group(1)) if match else 0
        checks.extend(
            [
                check("generated_tests_exit", generated.returncode == 0, f"exit={generated.returncode}; output={generated_text[-1500:]!r}"),
                check("generated_tests_executed", generated_count >= 5 and "OK" in generated_text, f"count={generated_count}; output={generated_text[-800:]!r}"),
            ]
        )
    except subprocess.TimeoutExpired:
        checks.append(check("generated_tests_exit", False, "generated tests timed out after 30 seconds"))

    try:
        contract = subprocess.run(
            [sys.executable, "-I", str(runner_path)],
            cwd=validation_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        contract_text = contract.stdout + contract.stderr
        checks.append(
            check(
                "independent_contract_tests",
                contract.returncode == 0 and "CONTRACT_OK" in contract.stdout,
                f"exit={contract.returncode}; output={contract_text[-2000:]!r}",
            )
        )
    except subprocess.TimeoutExpired:
        checks.append(check("independent_contract_tests", False, "contract tests timed out after 30 seconds"))
    if cleanup:
        with contextlib.suppress(Exception):
            import shutil
            shutil.rmtree(base)
    return quality_result(checks)


def validate_review(response: dict[str, Any], _: Path | None = None) -> dict[str, Any]:
    value, error = parse_json_content(response)
    checks = [
        check("natural_finish", response.get("finish_reason") == "stop", f"finish_reason={response.get('finish_reason')!r}"),
        check("json_parse", error is None, error or "parsed"),
    ]
    findings = value.get("findings") if isinstance(value, dict) else None
    checks.append(check("exact_top_level", isinstance(value, dict) and set(value) == {"findings"}, f"value={value!r}"[:1000]))
    if isinstance(findings, list):
        by_id = {item.get("id"): item for item in findings if isinstance(item, dict)}
        checks.extend(
            [
                check("exact_finding_ids", len(findings) == 3 and set(by_id) == set(REVIEW_LINES), f"ids={sorted(str(key) for key in by_id)}"),
                check("exact_finding_keys", all(set(item) == {"id", "line", "failure_mode", "fix"} for item in findings if isinstance(item, dict)) and len(findings) == 3, f"findings={findings!r}"[:1500]),
            ]
        )
        for finding_id, line in REVIEW_LINES.items():
            item = by_id.get(finding_id) or {}
            failure = str(item.get("failure_mode", "")).lower()
            fix = str(item.get("fix", "")).lower()
            checks.append(check(f"{finding_id}_line", item.get("line") == line, f"line={item.get('line')!r}, expected={line}"))
            if finding_id == "SQL_INJECTION":
                semantic = ("inject" in failure or "untrusted" in failure or "attacker" in failure) and ("parameter" in fix or "bind" in fix or "placeholder" in fix)
            elif finding_id == "CACHE_TENANT_LEAK":
                semantic = "tenant" in failure and ("leak" in failure or "collision" in failure or "other" in failure) and "tenant" in fix and "key" in fix
            else:
                semantic = ("pending" in failure or "sibling" in failure or "task" in failure) and ("cancel" in fix and ("await" in fix or "gather" in fix)) and any(term in failure + " " + fix for term in ("exception", "error", "suppress", "failure"))
            checks.append(check(f"{finding_id}_semantics", semantic, f"failure_mode={failure!r}; fix={fix!r}"))
    else:
        checks.append(check("findings_array", False, f"type={type(findings).__name__}"))
    return quality_result(checks)


def validate_reasoning(response: dict[str, Any], _: Path | None = None) -> dict[str, Any]:
    value, error = parse_json_content(response)
    checks = [
        check("natural_finish", response.get("finish_reason") == "stop", f"finish_reason={response.get('finish_reason')!r}"),
        check("json_parse", error is None, error or "parsed"),
        check("reasoning_not_truncated", response.get("finish_reason") == "stop" and (not (response.get("reasoning") or "").strip() or len((response.get("reasoning") or "").strip()) >= 120), f"reasoning_chars={len((response.get('reasoning') or '').strip())}"),
    ]
    if isinstance(value, dict):
        checks.append(check("exact_keys", set(value) == {"case_1", "case_2", "conclusion"}, f"keys={sorted(value)}"))
        case1 = value.get("case_1") if isinstance(value.get("case_1"), dict) else {}
        case2 = value.get("case_2") if isinstance(value.get("case_2"), dict) else {}
        events1 = case1.get("events") if isinstance(case1.get("events"), list) else []
        events2 = case2.get("events") if isinstance(case2.get("events"), list) else []
        cancelled1 = case1.get("cancelled_tasks") if isinstance(case1.get("cancelled_tasks"), list) else []
        cancelled2 = case2.get("cancelled_tasks") if isinstance(case2.get("cancelled_tasks"), list) else []
        explanation1 = str(case1.get("explanation", "")).lower()
        explanation2 = str(case2.get("explanation", "")).lower()
        conclusion = str(value.get("conclusion", "")).lower()
        checks.extend(
            [
                check("case1_events", events1 == ["A:ok", "A:finally", "B:finally", "C:finally"], f"events={events1!r}"),
                check("case1_exception", "valueerror" in str(case1.get("exception", "")).lower() and "b" in str(case1.get("exception", "")).lower() and case1.get("timeout_fired") is False, f"case_1={case1!r}"),
                check("case1_causality", cancelled1 == ["C"] and "cancel" in explanation1 and "c" in explanation1, f"cancelled={cancelled1!r}; explanation={explanation1!r}"),
                check("case2_events", set(events2) == {"A:ok", "A:finally", "B:finally", "C:finally"} and len(events2) == 4, f"events={events2!r}"),
                check("case2_exception", "timeouterror" in str(case2.get("exception", "")).replace(" ", "").lower() and case2.get("timeout_fired") is True, f"case_2={case2!r}"),
                check("case2_causality", set(cancelled2) == {"B", "C"} and len(cancelled2) == 2 and ("deadline" in explanation2 or "timeout" in explanation2) and "cancel" in explanation2, f"cancelled={cancelled2!r}; explanation={explanation2!r}"),
                check("completed_conclusion", len(conclusion) >= 120 and "failure" in conclusion and ("deadline" in conclusion or "timeout" in conclusion) and "cancel" in conclusion, f"conclusion={conclusion!r}"),
            ]
        )
    else:
        checks.append(check("json_object", False, f"type={type(value).__name__}"))
    return quality_result(checks)


def validate_edit(response: dict[str, Any], _: Path | None = None) -> dict[str, Any]:
    content = response.get("content") or ""
    normalized = content.strip("\n") + "\n"
    expected_lines = EDIT_EXPECTED.splitlines()
    actual_lines = normalized.splitlines()
    changed = [index + 1 for index, (before, after) in enumerate(zip(EDIT_SOURCE.splitlines(), actual_lines)) if before != after]
    if len(actual_lines) != len(EDIT_SOURCE.splitlines()):
        changed.append(-1)
    checks = [
        check("natural_finish", response.get("finish_reason") == "stop", f"finish_reason={response.get('finish_reason')!r}"),
        check("no_markdown_fence", "```" not in content, "Markdown fence present" if "```" in content else "none"),
        check("complete_line_count", len(actual_lines) == len(expected_lines), f"actual={len(actual_lines)}, expected={len(expected_lines)}"),
        check("target_fix", len(actual_lines) >= EDIT_TARGET_LINE and actual_lines[EDIT_TARGET_LINE - 1] == expected_lines[EDIT_TARGET_LINE - 1], f"target_line={actual_lines[EDIT_TARGET_LINE - 1] if len(actual_lines) >= EDIT_TARGET_LINE else None!r}"),
        check("unrelated_bytes_preserved", normalized == EDIT_EXPECTED, f"changed_lines={changed[:30]!r}; actual_sha256={sha256_bytes(normalized.encode('utf-8'))}"),
    ]
    return quality_result(checks)


VALIDATORS: dict[str, Callable[[dict[str, Any], Path | None], dict[str, Any]]] = {
    "tool": validate_tool,
    "debug": validate_debug,
    "implementation": validate_implementation,
    "review": validate_review,
    "reasoning": validate_reasoning,
    "edit": validate_edit,
}


def api_headers() -> dict[str, str]:
    return {"Accept": "text/event-stream", "Content-Type": "application/json"}


def assemble_tool_calls(state: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [state[index] for index in sorted(state)]


def perform_request(task: dict[str, Any], request_id: str, artifact_dir: Path, validate: bool) -> dict[str, Any]:
    raw_dir = artifact_dir / "raw"
    output_dir = artifact_dir / "outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = raw_dir / f"{request_id}.request.json"
    response_path = raw_dir / f"{request_id}.response.sse"
    output_path = output_dir / f"{request_id}.txt"
    reasoning_path = output_dir / f"{request_id}.reasoning.txt"
    payload = task["payload"]
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    request_path.write_bytes(payload_bytes + b"\n")
    request = urllib.request.Request(BASE_URL + "/v1/chat/completions", data=payload_bytes, headers=api_headers(), method="POST")

    started_at = utc_now()
    started_perf = time.perf_counter()
    first_token_perf: float | None = None
    status: int | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_state: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    saw_done = False
    raw_response = bytearray()

    def meaningful(value: Any) -> bool:
        return isinstance(value, str) and bool(value)

    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            status = response.status
            while True:
                line = response.readline()
                if not line:
                    break
                raw_response.extend(line)
                stripped = line.rstrip(b"\r\n")
                if not stripped.startswith(b"data:"):
                    continue
                data = stripped[5:].lstrip()
                if data == b"[DONE]":
                    saw_done = True
                    continue
                try:
                    event = json.loads(data)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid SSE event: {exc}")
                    continue
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                if event.get("error") is not None:
                    errors.append("server event error: " + json.dumps(event["error"], sort_keys=True))
                choices = event.get("choices") or []
                for choice in choices:
                    if not isinstance(choice, dict):
                        errors.append("non-object choice")
                        continue
                    reason = choice.get("finish_reason")
                    if isinstance(reason, str):
                        finish_reason = reason
                    delta = choice.get("delta") or {}
                    if not isinstance(delta, dict):
                        errors.append("non-object delta")
                        continue
                    content = delta.get("content")
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if meaningful(content):
                        content_parts.append(content)
                    if meaningful(reasoning):
                        reasoning_parts.append(reasoning)
                    calls = delta.get("tool_calls") or []
                    for call in calls:
                        if not isinstance(call, dict):
                            errors.append("non-object tool call delta")
                            continue
                        index = call.get("index", 0)
                        if type(index) is not int:
                            errors.append("invalid tool call index")
                            continue
                        current = tool_state.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if isinstance(call.get("id"), str):
                            current["id"] += call["id"]
                        if isinstance(call.get("type"), str):
                            current["type"] = call["type"]
                        function = call.get("function") or {}
                        if isinstance(function.get("name"), str):
                            current["function"]["name"] += function["name"]
                        if isinstance(function.get("arguments"), str):
                            current["function"]["arguments"] += function["arguments"]
                    if first_token_perf is None and (meaningful(content) or meaningful(reasoning) or bool(calls)):
                        first_token_perf = time.perf_counter()
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read()
        raw_response.extend(body)
        errors.append(f"HTTP {exc.code}: {exc.reason}; body={body[:2000]!r}")
    except Exception as exc:
        errors.append(f"HTTP/stream error: {type(exc).__name__}: {exc}")

    ended_perf = time.perf_counter()
    ended_at = utc_now()
    response_path.write_bytes(bytes(raw_response))
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    tool_calls = assemble_tool_calls(tool_state)
    output_path.write_text(content, encoding="utf-8")
    reasoning_path.write_text(reasoning, encoding="utf-8")
    answer_value: Any = tool_calls if tool_calls else content
    answer_bytes = canonical_json_bytes(answer_value) if tool_calls else content.encode("utf-8")

    def usage_int(key: str) -> int | None:
        value = usage.get(key) if isinstance(usage, dict) else None
        return value if type(value) is int and value >= 0 else None

    prompt_tokens = usage_int("prompt_tokens")
    completion_tokens = usage_int("completion_tokens")
    total_tokens = usage_int("total_tokens")
    e2e_latency = ended_perf - started_perf
    ttft = first_token_perf - started_perf if first_token_perf is not None else None
    decode_duration = ended_perf - first_token_perf if first_token_perf is not None else None
    common_checks = list(errors)
    if status != 200:
        common_checks.append(f"expected HTTP 200, got {status!r}")
    if not saw_done:
        common_checks.append("missing [DONE] event")
    if first_token_perf is None:
        common_checks.append("no content, reasoning, or tool-call token observed")
    if prompt_tokens is None or completion_tokens is None:
        common_checks.append(f"invalid usage: {usage!r}")
    if total_tokens is not None and prompt_tokens is not None and completion_tokens is not None and total_tokens != prompt_tokens + completion_tokens:
        common_checks.append("total token count does not equal prompt plus completion")

    response_value = {
        "content": content,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
    }
    quality = VALIDATORS[task["validator"]](response_value, artifact_dir / "validation" / request_id) if validate and not common_checks else quality_result([check("transport", False, "; ".join(common_checks))])
    decode_tokens = max((completion_tokens or 0) - 1, 0)
    result = {
        "request_id": request_id,
        "task_id": task["id"],
        "started_at": started_at,
        "finished_at": ended_at,
        "http_status": status,
        "transport_valid": not common_checks,
        "transport_errors": common_checks,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "ttft_seconds": ttft,
        "decode_duration_seconds": decode_duration,
        "decode_tokens_per_second": decode_tokens / decode_duration if decode_duration and decode_tokens else None,
        "e2e_latency_seconds": e2e_latency,
        "e2e_tokens_per_second": completion_tokens / e2e_latency if completion_tokens else None,
        "finish_reason": finish_reason,
        "output_sha256": sha256_bytes(answer_bytes),
        "reasoning_sha256": sha256_bytes(reasoning.encode("utf-8")),
        "output_bytes": len(answer_bytes),
        "reasoning_bytes": len(reasoning.encode("utf-8")),
        "quality": quality,
        "artifacts": {
            "request": str(request_path.relative_to(artifact_dir)),
            "response_sse": str(response_path.relative_to(artifact_dir)),
            "output": str(output_path.relative_to(artifact_dir)),
            "reasoning": str(reasoning_path.relative_to(artifact_dir)),
        },
        "_started_perf": started_perf,
        "_ended_perf": ended_perf,
    }
    return result


def fetch_json(url: str, timeout: float = 10) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def fetch_text(url: str, timeout: float = 10) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_prometheus(text: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    sample_re = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{.*\})?\s+([-+0-9.eE]+)(?:\s+\d+)?$")
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = sample_re.match(line)
        if not match:
            continue
        with contextlib.suppress(ValueError):
            totals[match.group(1)] = totals.get(match.group(1), 0.0) + float(match.group(2))
    return totals


def speculative_delta(before_text: str, after_text: str) -> dict[str, Any]:
    before = parse_prometheus(before_text)
    after = parse_prometheus(after_text)
    accepted = after.get(ACCEPTED_COUNTER, 0.0) - before.get(ACCEPTED_COUNTER, 0.0)
    drafted = after.get(DRAFT_COUNTER, 0.0) - before.get(DRAFT_COUNTER, 0.0)
    steps = after.get(DRAFT_STEP_COUNTER, 0.0) - before.get(DRAFT_STEP_COUNTER, 0.0)
    per_position = {}
    for line in after_text.splitlines():
        if line.startswith(PER_POSITION_COUNTER):
            position_match = re.search(r'position="([^"]+)"', line)
            value_match = re.search(r"\s([-+0-9.eE]+)$", line)
            if position_match and value_match:
                per_position[position_match.group(1)] = float(value_match.group(1))
    return {
        "accepted_tokens": accepted,
        "draft_tokens": drafted,
        "draft_steps": steps,
        "acceptance_rate": accepted / drafted if drafted > 0 else None,
        "accepted_tokens_per_draft_step": accepted / steps if steps > 0 else None,
        "accepted_tokens_per_position_after": per_position,
    }


class GpuMemoryPoller:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                completed = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
                    text=True,
                    capture_output=True,
                    timeout=5,
                    check=True,
                )
                used = int(completed.stdout.strip().splitlines()[0])
                self.samples.append({"at": utc_now(), "used_mib": used})
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            self.stop_event.wait(GPU_POLL_SECONDS)

    def summary(self) -> dict[str, Any]:
        values = [sample["used_mib"] for sample in self.samples]
        return {
            "gpu_index": 0,
            "hardware_limit_mib": GPU_TOTAL_MIB,
            "peak_used_mib": max(values) if values else None,
            "sample_count": len(values),
            "poll_interval_seconds": GPU_POLL_SECONDS,
            "errors": self.errors,
            "samples": self.samples,
        }


def fixed_invariants() -> dict[str, Any]:
    return {
        "image": IMAGE,
        "target_checkpoint": TARGET,
        "target_revision": TARGET_REVISION,
        "target_quantization": "checkpoint compressed-tensors mixed NVFP4/FP8",
        "kv_cache_dtype": "checkpoint-default FP8 kv_cache_scheme (no CLI override)",
        "mamba_ssm_cache_dtype": MAMBA_SSM_CACHE_DTYPE,
        "mamba_secondary_cache_dtype": MAMBA_CACHE_DTYPE,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": MAX_NUM_SEQS,
        "prefix_caching": True,
        "chunked_prefill": True,
        "concurrency": 1,
        "served_model_during_benchmark": SERVED,
    }


def candidate_label(method: str, depth: int) -> str:
    return "off" if method == "off" else f"{method}{depth}"


def docker_command(method: str, depth: int, container: str) -> list[str]:
    command = [
        "docker", "run", "-d",
        "--device", "nvidia.com/gpu=all",
        "--name", container,
        "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL",
        "--pids-limit", "4096",
        "--shm-size", "16g",
        "-p", f"127.0.0.1:{PORT}:{PORT}",
        "--tmpfs", "/tmp:rw,nosuid,nodev,exec,size=8g",
        "-e", "HOME=/cache",
        "-e", "HF_HOME=/models",
        "-e", "HF_HUB_OFFLINE=1",
        "-e", "TRANSFORMERS_OFFLINE=1",
        "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "-e", "VLLM_WORKER_MULTIPROC_METHOD=spawn",
        "-e", "NCCL_CUMEM_ENABLE=0",
        "-e", "TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor",
        "-e", "TRITON_CACHE_DIR=/cache/triton",
        "-v", "vllm-compile-cache:/cache",
        "-v", "/home/workbench/.cache/huggingface:/models:ro",
        "--entrypoint", "vllm",
        IMAGE,
        "serve", TARGET,
        "--served-model-name", SERVED,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
        "--max-num-seqs", str(MAX_NUM_SEQS),
        "--mamba-ssm-cache-dtype", MAMBA_SSM_CACHE_DTYPE,
        "--mamba-cache-dtype", MAMBA_CACHE_DTYPE,
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--disable-custom-all-reduce",
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
        "--generation-config", "vllm",
        "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
    ]
    if method == "mtp":
        command.extend(["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": depth}, separators=(",", ":"))])
    elif method == "dflash":
        command.extend(["--speculative-config", json.dumps({"method": "dflash", "model": DRAFTER, "num_speculative_tokens": depth}, separators=(",", ":"))])
    elif method != "off":
        raise ValueError(f"unsupported method: {method}")
    return command


def docker(*args: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], text=True, capture_output=True, check=check, timeout=timeout)


def wait_for_server(container: str) -> tuple[dict[str, Any], str]:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last_error = "not attempted"
    while time.monotonic() < deadline:
        inspect = docker("inspect", "--format", "{{.State.Status}}|{{.State.ExitCode}}", container, check=False)
        state = inspect.stdout.strip()
        if state.startswith("exited") or state.startswith("dead"):
            logs = docker("logs", container, check=False, timeout=60).stdout + docker("logs", container, check=False, timeout=60).stderr
            raise RuntimeError(f"container exited during startup ({state}):\n{logs[-12000:]}")
        try:
            with urllib.request.urlopen(BASE_URL + "/health", timeout=3) as response:
                if response.status == 200:
                    models = fetch_json(BASE_URL + "/v1/models", timeout=10)
                    return models, last_error
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    logs_result = docker("logs", container, check=False, timeout=60)
    raise TimeoutError(f"server did not become healthy: {last_error}\n{(logs_result.stdout + logs_result.stderr)[-12000:]}")


def capture_logs(container: str) -> str:
    completed = docker("logs", container, check=False, timeout=120)
    return completed.stdout + completed.stderr


def analyze_logs(logs: str) -> dict[str, Any]:
    lines = logs.splitlines()
    relevant = [line for line in lines if re.search(r"warning|speculat|dflash|mtp|draft|scheduler|capacity", line, re.IGNORECASE)]
    errors = [line for line in lines if re.search(r"traceback|\berror\b|runtimeerror|valueerror|scheduler-capacity|no available slot", line, re.IGNORECASE)]
    capacity = [line for line in lines if re.search(r"no available slot|scheduler[- ]capacity|insufficient.*(?:draft|spec)|cannot reserve.*(?:draft|spec)|(?:draft|spec).*(?:exceeds|must be at least)", line, re.IGNORECASE) and re.search(r"warning|error|exception|failed|invalid", line, re.IGNORECASE)]
    return {
        "relevant_warnings": relevant,
        "error_lines": errors,
        "scheduler_capacity_errors": capacity,
        "contains_traceback": "Traceback (most recent call last)" in logs,
    }


def aggregate_requests(requests: list[dict[str, Any]], measured_wall_seconds: float) -> dict[str, Any]:
    completion = sum(item.get("completion_tokens") or 0 for item in requests)
    prompt = sum(item.get("prompt_tokens") or 0 for item in requests)
    decode_tokens = sum(max((item.get("completion_tokens") or 0) - 1, 0) for item in requests)
    decode_seconds = sum(item.get("decode_duration_seconds") or 0 for item in requests)
    e2e_seconds = sum(item.get("e2e_latency_seconds") or 0 for item in requests)
    ttfts = [item["ttft_seconds"] for item in requests if item.get("ttft_seconds") is not None]
    by_task: dict[str, Any] = {}
    for task_id in [spec["id"] for spec in task_specs()]:
        members = [item for item in requests if item["task_id"] == task_id]
        by_task[task_id] = {
            "requests": len(members),
            "quality_passes": sum(1 for item in members if item["quality"]["passed"]),
            "prompt_tokens": sum(item.get("prompt_tokens") or 0 for item in members),
            "completion_tokens": sum(item.get("completion_tokens") or 0 for item in members),
            "mean_ttft_seconds": statistics.mean([item["ttft_seconds"] for item in members if item.get("ttft_seconds") is not None]) if any(item.get("ttft_seconds") is not None for item in members) else None,
            "mean_e2e_latency_seconds": statistics.mean([item["e2e_latency_seconds"] for item in members]),
            "aggregate_e2e_tokens_per_second": sum(item.get("completion_tokens") or 0 for item in members) / sum(item.get("e2e_latency_seconds") or 0 for item in members),
            "aggregate_decode_tokens_per_second": sum(max((item.get("completion_tokens") or 0) - 1, 0) for item in members) / sum(item.get("decode_duration_seconds") or 0 for item in members),
            "output_sha256": [item["output_sha256"] for item in members],
            "failure_reasons": [reason for item in members for reason in item["quality"]["failure_reasons"]],
        }
    return {
        "request_count": len(requests),
        "total_prompt_tokens": prompt,
        "total_completion_tokens": completion,
        "sum_request_e2e_seconds": e2e_seconds,
        "total_corpus_wall_seconds": measured_wall_seconds,
        "aggregate_e2e_tokens_per_second": completion / e2e_seconds if e2e_seconds else None,
        "aggregate_decode_tokens_per_second": decode_tokens / decode_seconds if decode_seconds else None,
        "mean_ttft_seconds": statistics.mean(ttfts) if ttfts else None,
        "max_ttft_seconds": max(ttfts) if ttfts else None,
        "quality_passes": sum(1 for item in requests if item["quality"]["passed"]),
        "quality_failures": sum(1 for item in requests if not item["quality"]["passed"]),
        "per_task": by_task,
    }


def run_candidate(run_id: str, phase: str, method: str, depth: int, rounds: int) -> Path:
    ensure_workload_manifest()
    if phase == "qualification" and rounds != 3:
        raise ValueError("qualification requires exactly three rounds")
    run_dir = RESULTS_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    container = f"qwen36-35b-agent-{run_id}".replace("_", "-")[:63]
    command = docker_command(method, depth, container)
    config = {
        "run_id": run_id,
        "phase": phase,
        "method": method,
        "draft_depth": 0 if method == "off" else depth,
        "rounds": rounds,
        "warmup_rounds": WARMUP_ROUNDS,
        "container": container,
        "docker_command": command,
        "docker_command_sha256": sha256_bytes(canonical_json_bytes(command)),
        "invariants": fixed_invariants(),
        "workload_manifest_sha256": sha256_bytes(WORKLOAD_MANIFEST.read_bytes()),
    }
    write_json(run_dir / "configuration.json", config)
    poller = GpuMemoryPoller()
    poller.start()
    started_at = utc_now()
    models: Any = None
    startup_error: str | None = None
    warmup_results: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    metrics_before = ""
    metrics_after = ""
    logs = ""
    measured_wall_seconds = 0.0
    try:
        docker("rm", "-f", container, check=False)
        launched = subprocess.run(command, text=True, capture_output=True, timeout=120)
        if launched.returncode != 0:
            raise RuntimeError(f"docker run failed: stdout={launched.stdout!r}; stderr={launched.stderr!r}")
        print(f"[{utc_now()}] {run_id}: container {container} started", flush=True)
        models, _ = wait_for_server(container)
        print(f"[{utc_now()}] {run_id}: server healthy", flush=True)
        for index, task in enumerate(task_specs(), 1):
            warmup_id = f"warmup-01-{index:02d}-{task['id']}"
            warmup = perform_request(task, warmup_id, run_dir, validate=False)
            warmup_results.append(warmup)
            if not warmup["transport_valid"]:
                raise RuntimeError(f"warmup transport failed for {task['id']}: {warmup['transport_errors']}")
            print(f"[{utc_now()}] {run_id}: warmup {task['id']} complete", flush=True)
        metrics_before = fetch_text(BASE_URL + "/metrics", timeout=30)
        (run_dir / "metrics-before.prom").write_text(metrics_before, encoding="utf-8")
        measured_started = time.perf_counter()
        for round_number in range(1, rounds + 1):
            for index, task in enumerate(task_specs(), 1):
                request_id = f"round-{round_number:02d}-{index:02d}-{task['id']}"
                result = perform_request(task, request_id, run_dir, validate=True)
                result["round"] = round_number
                requests.append(result)
                print(
                    f"[{utc_now()}] {run_id}: r{round_number} {task['id']} "
                    f"quality={'PASS' if result['quality']['passed'] else 'FAIL'} "
                    f"e2e={result['e2e_latency_seconds']:.3f}s",
                    flush=True,
                )
        measured_wall_seconds = time.perf_counter() - measured_started
        metrics_after = fetch_text(BASE_URL + "/metrics", timeout=30)
        (run_dir / "metrics-after.prom").write_text(metrics_after, encoding="utf-8")
        logs = capture_logs(container)
    except Exception as exc:
        startup_error = f"{type(exc).__name__}: {exc}"
        print(f"[{utc_now()}] {run_id}: FAILED {startup_error}", flush=True)
        with contextlib.suppress(Exception):
            logs = capture_logs(container)
    finally:
        poller.stop()
        (run_dir / "server.log").write_text(logs, encoding="utf-8")
        gpu = poller.summary()
        samples = gpu.pop("samples")
        write_json(run_dir / "gpu-memory-summary.json", gpu)
        with (run_dir / "gpu-memory-samples.jsonl").open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
        inspect = docker("inspect", container, check=False, timeout=60)
        if inspect.returncode == 0:
            (run_dir / "container-inspect.json").write_text(inspect.stdout, encoding="utf-8")
        docker("rm", "-f", container, check=False, timeout=120)

    for item in requests:
        item.pop("_started_perf", None)
        item.pop("_ended_perf", None)
    for item in warmup_results:
        item.pop("_started_perf", None)
        item.pop("_ended_perf", None)
    log_analysis = analyze_logs(logs)
    spec = speculative_delta(metrics_before, metrics_after) if metrics_before and metrics_after else {
        "accepted_tokens": None,
        "draft_tokens": None,
        "draft_steps": None,
        "acceptance_rate": None,
        "accepted_tokens_per_draft_step": None,
        "accepted_tokens_per_position_after": {},
    }
    aggregate = aggregate_requests(requests, measured_wall_seconds) if requests else None
    invalid_reasons: list[str] = []
    if startup_error:
        invalid_reasons.append(startup_error)
    if len(requests) != rounds * 6:
        invalid_reasons.append(f"measured request count {len(requests)}, expected {rounds * 6}")
    if any(not item["transport_valid"] for item in requests):
        invalid_reasons.append("one or more measured requests had transport failures")
    if log_analysis["scheduler_capacity_errors"]:
        invalid_reasons.append("scheduler-capacity error appeared in server log")
    if log_analysis["contains_traceback"]:
        invalid_reasons.append("traceback appeared in pre-shutdown server log")
    if method != "off" and not ((spec.get("draft_tokens") or 0) > 0 and (spec.get("draft_steps") or 0) > 0):
        invalid_reasons.append("speculative counters did not increase")
    peak = json.loads((run_dir / "gpu-memory-summary.json").read_text(encoding="utf-8")).get("peak_used_mib")
    if peak is None or peak > GPU_TOTAL_MIB:
        invalid_reasons.append(f"peak GPU memory invalid or over limit: {peak!r}")
    quality_passes = aggregate["quality_passes"] if aggregate else 0
    expected_quality = rounds * 6
    qualification_eligible = phase == "qualification" and not invalid_reasons and quality_passes == 18
    result = {
        "schema_version": 1,
        "created_at": started_at,
        "finished_at": utc_now(),
        "configuration": config,
        "models_response": models,
        "startup_error": startup_error,
        "status": "valid" if not invalid_reasons else "invalid",
        "invalid_reasons": invalid_reasons,
        "quality_decisions": {"passed": quality_passes, "expected": expected_quality},
        "qualification_eligible": qualification_eligible,
        "warmup": warmup_results,
        "requests": requests,
        "aggregate": aggregate,
        "speculative_metrics": spec,
        "gpu_memory": json.loads((run_dir / "gpu-memory-summary.json").read_text(encoding="utf-8")),
        "log_analysis": log_analysis,
        "artifacts": {
            "configuration": "configuration.json",
            "server_log": "server.log",
            "metrics_before": "metrics-before.prom" if metrics_before else None,
            "metrics_after": "metrics-after.prom" if metrics_after else None,
            "gpu_summary": "gpu-memory-summary.json",
            "gpu_samples": "gpu-memory-samples.jsonl",
        },
    }
    result_path = run_dir / "result.json"
    write_json(result_path, result)
    print(f"[{utc_now()}] {run_id}: result {result_path} status={result['status']} quality={quality_passes}/{expected_quality}", flush=True)
    return result_path


def valid_implementation_fixture() -> str:
    return '''import asyncio
import unittest

async def map_ordered(items, worker, limit):
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if limit < 1:
        raise ValueError("limit must be positive")
    if not callable(worker):
        raise TypeError("worker must be callable")
    try:
        values = list(items)
    except TypeError:
        raise TypeError("items must be iterable") from None
    semaphore = asyncio.Semaphore(limit)
    async def invoke(item):
        async with semaphore:
            return await worker(item)
    tasks = [asyncio.create_task(invoke(item)) for item in values]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

class Tests(unittest.IsolatedAsyncioTestCase):
    async def test_validation_zero(self):
        with self.assertRaises(ValueError):
            await map_ordered([], lambda x: x, 0)
    async def test_validation_bool(self):
        with self.assertRaises(TypeError):
            await map_ordered([], lambda x: x, True)
    async def test_order(self):
        async def worker(x):
            await asyncio.sleep(0.001 * (3 - x))
            return x
        self.assertEqual(await map_ordered([0, 1, 2], worker, 2), [0, 1, 2])
    async def test_error(self):
        async def worker(x):
            if x == 1: raise RuntimeError("x")
            await asyncio.sleep(1)
        with self.assertRaises(RuntimeError):
            await map_ordered([1, 2], worker, 2)
    async def test_empty(self):
        async def worker(x): return x
        self.assertEqual(await map_ordered([], worker, 1), [])

if __name__ == "__main__":
    unittest.main()
'''


def self_test() -> None:
    ensure_workload_manifest()
    good_tool = {"content": "", "reasoning": "done", "finish_reason": "tool_calls", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "fetch_issue", "arguments": '{"repo":"acme/widgets","number":317,"include_comments":false}'}}]}
    good_debug_value = {"bug": "hard incorrectly reuses the first parts[0] instead of second parts[1]", "line": 4, "fix": "hard = int(parts[1])", "regression_test": "parse_limits('10:20') returns soft 10 and hard 20"}
    good_debug = {"content": json.dumps(good_debug_value), "reasoning": "done", "finish_reason": "stop", "tool_calls": []}
    good_impl = {"content": valid_implementation_fixture(), "reasoning": "done", "finish_reason": "stop", "tool_calls": []}
    findings = [
        {"id": "SQL_INJECTION", "line": REVIEW_LINES["SQL_INJECTION"], "failure_mode": "Attacker-controlled tenant input is interpolated into SQL and permits injection.", "fix": "Use bound parameters or placeholders for tenant and order values instead of interpolation."},
        {"id": "CACHE_TENANT_LEAK", "line": REVIEW_LINES["CACHE_TENANT_LEAK"], "failure_mode": "The key omits tenant identity, causing collisions and cross-tenant data leaks.", "fix": "Include the tenant identifier and order identifier in the cache key."},
        {"id": "ASYNC_TASK_LEAK", "line": REVIEW_LINES["ASYNC_TASK_LEAK"], "failure_mode": "Pending sibling tasks keep running and an exception may be suppressed after a completed task errors.", "fix": "Cancel every pending task and await them with gather before propagating the selected exception."},
    ]
    good_review = {"content": json.dumps({"findings": findings}), "reasoning": "done", "finish_reason": "stop", "tool_calls": []}
    case1_explanation = "B raises before the deadline, TaskGroup cancels C, and C finalizes before the context exits and the ValueError group is handled."
    case2_explanation = "The timeout deadline cancels the parent wait, TaskGroup cancels B and C and waits for their finalizers, then TimeoutError escapes."
    conclusion = "Failure-driven cancellation starts when B raises and cancels its sibling, while deadline-driven cancellation starts at the timeout and cancels all unfinished children; in both cases TaskGroup awaits cleanup and every finally block completes before control leaves the context."
    good_reasoning = {"content": json.dumps({"case_1": {"events": ["A:ok", "A:finally", "B:finally", "C:finally"], "cancelled_tasks": ["C"], "exception": "ExceptionGroup(ValueError:B)", "timeout_fired": False, "explanation": case1_explanation}, "case_2": {"events": ["A:ok", "A:finally", "B:finally", "C:finally"], "cancelled_tasks": ["B", "C"], "exception": "TimeoutError", "timeout_fired": True, "explanation": case2_explanation}, "conclusion": conclusion}), "reasoning": "x" * 150, "finish_reason": "stop", "tool_calls": []}
    good_edit = {"content": EDIT_EXPECTED, "reasoning": "done", "finish_reason": "stop", "tool_calls": []}
    fixtures = [(validate_tool, good_tool), (validate_debug, good_debug), (validate_implementation, good_impl), (validate_review, good_review), (validate_reasoning, good_reasoning), (validate_edit, good_edit)]
    for validator, fixture in fixtures:
        result = validator(fixture, None)
        if not result["passed"]:
            raise AssertionError(f"valid fixture failed {validator.__name__}: {result}")
        broken = dict(fixture)
        broken["finish_reason"] = "length"
        result = validator(broken, None)
        if result["passed"]:
            raise AssertionError(f"invalid fixture passed {validator.__name__}")
    print(f"SELF_TEST_OK validators={len(fixtures)} workload={WORKLOAD_MANIFEST}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test")
    run = subparsers.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--phase", choices=["screening", "qualification", "production-verification"], required=True)
    run.add_argument("--method", choices=["off", "mtp", "dflash"], required=True)
    run.add_argument("--depth", type=int, default=0)
    run.add_argument("--rounds", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "self-test":
        self_test()
        return 0
    if args.method == "off" and args.depth != 0:
        raise SystemExit("off method requires depth 0")
    if args.method != "off" and args.depth < 1:
        raise SystemExit("speculative method requires positive depth")
    path = run_candidate(args.run_id, args.phase, args.method, args.depth, args.rounds)
    result = json.loads(path.read_text(encoding="utf-8"))
    return 0 if result["status"] == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
