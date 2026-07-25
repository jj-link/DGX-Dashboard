#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from run_depth_candidate import (
    BASE_URL,
    GpuMemoryPoller,
    fetch_json,
    fetch_text,
    parse_spec_metrics,
    subtract_metrics,
)

SEED = 424242
SYSTEM = (
    "You are a coding agent operating in a software repository. Be precise, "
    "obey output contracts exactly, use tools when requested, and complete the "
    "task rather than merely describing what should be done."
)

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read an exact line range from a repository file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["path", "offset", "limit"],
            "additionalProperties": False,
        },
    },
}

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": "Replace one exact source fragment in a repository file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
    },
}

WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write",
        "description": "Create or overwrite one complete repository file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def response_schema(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def base_payload(
    model: str,
    user: str,
    *,
    max_tokens: int,
    response_format: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": SEED,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if response_format is not None:
        payload["response_format"] = response_format
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = False
    return payload


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
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    first_token_at: float | None = None
    raw_lines: list[bytes] = []
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1200) as response:
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
                reasoning = (
                    delta.get("reasoning") or delta.get("reasoning_content") or ""
                )
                fragments = delta.get("tool_calls") or []
                if first_token_at is None and (content or reasoning or fragments):
                    first_token_at = time.perf_counter()
                text_parts.append(content)
                reasoning_parts.append(reasoning)
                for fragment in fragments:
                    index = int(fragment.get("index", 0))
                    call = tool_calls.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    call["id"] += fragment.get("id") or ""
                    call["type"] = fragment.get("type") or call["type"]
                    function = fragment.get("function") or {}
                    call["function"]["name"] += function.get("name") or ""
                    call["function"]["arguments"] += function.get("arguments") or ""
                finish_reason = choice.get("finish_reason") or finish_reason
    ended = time.perf_counter()
    raw_path.write_bytes(b"".join(raw_lines))
    text = "".join(text_parts)
    reasoning = "".join(reasoning_parts)
    calls = [tool_calls[index] for index in sorted(tool_calls)]
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    decode_duration = ended - first_token_at if first_token_at is not None else None
    output_identity = json.dumps(
        {"text": text, "reasoning": reasoning, "tool_calls": calls},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
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
        "output_sha256": sha256_bytes(output_identity),
        "output_bytes": len(output_identity),
        "text": text,
        "reasoning": reasoning,
        "tool_calls": calls,
    }


def parse_json_text(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(value, dict):
        return None, "response is not a JSON object"
    return value, None


def quality(reasons: list[str], **evidence: Any) -> dict[str, Any]:
    return {"passed": not reasons, "failure_reasons": reasons, **evidence}


def validate_read(result: dict[str, Any], _: Path) -> dict[str, Any]:
    reasons: list[str] = []
    calls = result["tool_calls"]
    args: dict[str, Any] | None = None
    if len(calls) != 1:
        reasons.append(f"expected exactly one tool call, got {len(calls)}")
    elif calls[0]["function"]["name"] != "read":
        reasons.append(f"wrong tool: {calls[0]['function']['name']!r}")
    else:
        try:
            args = json.loads(calls[0]["function"]["arguments"])
        except json.JSONDecodeError as exc:
            reasons.append(f"invalid tool arguments: {exc}")
    expected = {"path": "src/cache.py", "offset": 120, "limit": 61}
    if args is not None and args != expected:
        reasons.append(f"tool arguments differ: {args!r}")
    if result["finish_reason"] != "tool_calls":
        reasons.append(f"finish_reason={result['finish_reason']!r}, expected 'tool_calls'")
    if result["text"].strip():
        reasons.append("assistant emitted prose before the required tool call")
    return quality(reasons, parsed_arguments=args)


def validate_debug(result: dict[str, Any], _: Path) -> dict[str, Any]:
    reasons: list[str] = []
    value, error = parse_json_text(result["text"])
    if error:
        reasons.append(error)
    expected_keys = {"root_cause", "failing_input", "replacement"}
    if value is not None:
        if set(value) != expected_keys:
            reasons.append(f"keys={sorted(value)}, expected={sorted(expected_keys)}")
        if value.get("failing_input") != 0:
            reasons.append("failing_input must be numeric zero")
        cause = str(value.get("root_cause", "")).lower()
        replacement = str(value.get("replacement", "")).lower()
        if not any(term in cause for term in ("truth", "falsy", "falsey", "zero")):
            reasons.append("root_cause does not explain the truthiness/zero defect")
        if "is not none" not in replacement:
            reasons.append("replacement does not preserve zero while filtering None")
    if result["finish_reason"] != "stop":
        reasons.append(f"finish_reason={result['finish_reason']!r}, expected 'stop'")
    return quality(reasons, parsed=value)


def implementation_test_source() -> str:
    return '''import asyncio
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("generated", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ordered_map = module.ordered_map

class MarkerError(RuntimeError):
    pass

async def main():
    active = 0
    maximum = 0

    async def worker(item):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep((6 - item) * 0.003)
            return item * 10
        finally:
            active -= 1

    values = await ordered_map(list(range(6)), worker, 2)
    assert values == [0, 10, 20, 30, 40, 50], values
    assert maximum <= 2, maximum

    for bad in (0, -1, 1.5, True):
        try:
            await ordered_map([], worker, bad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid limit accepted: {bad!r}")

    started = set()
    finished = set()

    async def failing(item):
        started.add(item)
        try:
            if item == 1:
                await asyncio.sleep(0.02)
                raise MarkerError("first failure")
            await asyncio.sleep(5)
            return item
        finally:
            finished.add(item)

    before = set(asyncio.all_tasks())
    started_at = asyncio.get_running_loop().time()
    try:
        await asyncio.wait_for(ordered_map([0, 1, 2, 3], failing, 4), 1)
    except MarkerError as exc:
        assert str(exc) == "first failure"
    else:
        raise AssertionError("first worker error was not propagated")
    elapsed = asyncio.get_running_loop().time() - started_at
    await asyncio.sleep(0)
    leaked = [task for task in asyncio.all_tasks() if task not in before and not task.done()]
    assert elapsed < 0.5, elapsed
    assert started, "no workers started"
    assert finished == started, (started, finished)
    assert not leaked, leaked

asyncio.run(main())
print("PASS")
'''


def validate_implementation(result: dict[str, Any], root: Path) -> dict[str, Any]:
    reasons: list[str] = []
    calls = result["tool_calls"]
    args: dict[str, Any] | None = None
    source = ""
    if len(calls) != 1:
        reasons.append(f"expected exactly one write call, got {len(calls)}")
    elif calls[0]["function"]["name"] != "write":
        reasons.append(f"wrong tool: {calls[0]['function']['name']!r}")
    else:
        try:
            args = json.loads(calls[0]["function"]["arguments"])
        except json.JSONDecodeError as exc:
            reasons.append(f"invalid write arguments: {exc}")
    if args is not None:
        if set(args) != {"path", "content"}:
            reasons.append("write contains unexpected arguments")
        if args.get("path") != "src/ordered_map.py":
            reasons.append("write targets the wrong file")
        source = str(args.get("content", ""))
    if result["finish_reason"] != "tool_calls":
        reasons.append(
            f"finish_reason={result['finish_reason']!r}, expected 'tool_calls'"
        )
    if result["text"].strip():
        reasons.append("assistant emitted prose before the write call")
    if "```" in source:
        reasons.append("implementation contains Markdown fences")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        tree = None
        reasons.append(f"invalid Python: {exc}")
    if tree is not None and not any(
        isinstance(node, ast.AsyncFunctionDef) and node.name == "ordered_map"
        for node in tree.body
    ):
        reasons.append("missing async ordered_map")
    external: dict[str, Any] | None = None
    if not reasons:
        root.mkdir(parents=True, exist_ok=True)
        generated = root / "generated.py"
        test_file = root / "test_generated.py"
        generated.write_text(source, encoding="utf-8")
        test_file.write_text(implementation_test_source(), encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(test_file), str(generated)],
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
            external = {
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "timed_out": False,
            }
            if completed.returncode != 0:
                reasons.append(
                    "generated implementation failed behavioral tests: "
                    + (completed.stderr or completed.stdout)[-1500:]
                )
        except subprocess.TimeoutExpired:
            external = {
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "timed_out": True,
            }
            reasons.append("generated implementation tests timed out")
    return quality(
        reasons,
        external_semantic_test=external,
        generated_source_sha256=sha256_bytes(source.encode("utf-8")),
    )


def validate_review(result: dict[str, Any], _: Path) -> dict[str, Any]:
    reasons: list[str] = []
    value, error = parse_json_text(result["text"])
    if error:
        reasons.append(error)
    if value is not None:
        if set(value) != {"sql", "cache", "async_control"}:
            reasons.append("review must contain exactly sql, cache, and async_control")
        sql = value.get("sql") if isinstance(value.get("sql"), dict) else {}
        cache = value.get("cache") if isinstance(value.get("cache"), dict) else {}
        async_control = (
            value.get("async_control")
            if isinstance(value.get("async_control"), dict)
            else {}
        )
        sql_text = " ".join(map(str, sql.values())).lower()
        if not any(term in sql_text for term in ("parameter", "placeholder", "bind")):
            reasons.append("SQL fix is not parameterized")
        falsey = str(cache.get("falsey_hit", "")).lower()
        expiry = str(cache.get("expiry", "")).lower()
        stampede = str(cache.get("stampede", "")).lower()
        if not any(term in falsey for term in ("none", "falsy", "falsey", "truth")):
            reasons.append("cache review misses falsey cached values")
        if not any(term in expiry for term in ("ttl", "expir", "stale")):
            reasons.append("cache review misses expiration")
        if not any(term in stampede for term in ("lock", "singleflight", "coalesc", "deduplic", "concurrent", "simultaneous", "redundant")):
            reasons.append("cache review misses concurrent stampede prevention")
        async_text = " ".join(map(str, async_control.values())).lower()
        if "cancel" not in async_text or not any(
            term in async_text for term in ("gather", "await")
        ):
            reasons.append("async fix does not cancel and await pending siblings")
    if result["finish_reason"] != "stop":
        reasons.append(f"finish_reason={result['finish_reason']!r}, expected 'stop'")
    return quality(reasons, parsed=value)


def validate_reasoning(result: dict[str, Any], _: Path) -> dict[str, Any]:
    reasons: list[str] = []
    value, error = parse_json_text(result["text"])
    if error:
        reasons.append(error)
    if value is not None:
        if set(value) != {"completion_time_seconds", "b_finally_runs", "why"}:
            reasons.append("reasoning response has the wrong key set")
        if value.get("completion_time_seconds") != 2:
            reasons.append("completion time must be two seconds")
        if value.get("b_finally_runs") is not True:
            reasons.append("B's finally block must run")
        why = str(value.get("why", "")).lower()
        for required in ("cancel", "finally", "gather"):
            if required not in why:
                reasons.append(f"why field omits {required}")
    if result["finish_reason"] != "stop":
        reasons.append(f"finish_reason={result['finish_reason']!r}, expected 'stop'")
    return quality(reasons, parsed=value)


def validate_edit(result: dict[str, Any], _: Path) -> dict[str, Any]:
    reasons: list[str] = []
    calls = result["tool_calls"]
    if len(calls) != 1:
        reasons.append(f"expected exactly one edit call, got {len(calls)}")
        args: dict[str, Any] | None = None
    elif calls[0]["function"]["name"] != "edit":
        reasons.append(f"wrong tool: {calls[0]['function']['name']!r}")
        args = None
    else:
        try:
            args = json.loads(calls[0]["function"]["arguments"])
        except json.JSONDecodeError as exc:
            reasons.append(f"invalid edit arguments: {exc}")
            args = None
    if args is not None:
        if args.get("path") != "src/cache.py":
            reasons.append("edit targets the wrong file")
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")
        if old_text != "    if value:\n":
            reasons.append("edit old_text is not the targeted condition")
        if new_text not in {"    if value is not None:\n", "    if key in cache:\n"}:
            reasons.append("edit does not preserve cached falsey values")
    return quality(reasons, parsed_arguments=args)


def review_source() -> str:
    filler_a = "\n".join(
        f"def helper_before_{index}(value):\n    return value + {index}\n"
        for index in range(35)
    )
    filler_b = "\n".join(
        f"def helper_after_{index}(value):\n    return value * {index + 1}\n"
        for index in range(35)
    )
    return f'''import asyncio
import time

{filler_a}

def get_user(db, user_id):
    return db.execute(f"SELECT id, name FROM users WHERE id = {{user_id}}")

cache = {{}}

async def cached_profile(key, load_profile):
    value = cache.get(key)
    if value:
        return value
    value = await load_profile(key)
    cache[key] = value
    return value

{filler_b}

async def run_jobs(coros):
    tasks = [asyncio.create_task(coro) for coro in coros]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    for task in done:
        if task.exception() is not None:
            raise task.exception()
    return [await task for task in tasks]
'''


def build_tasks(model: str) -> list[dict[str, Any]]:
    debug_schema = {
        "type": "object",
        "properties": {
            "root_cause": {"type": "string"},
            "failing_input": {"type": "integer"},
            "replacement": {"type": "string"},
        },
        "required": ["root_cause", "failing_input", "replacement"],
        "additionalProperties": False,
    }
    review_schema = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "object",
                "properties": {
                    "defect": {"type": "string"},
                    "fix": {"type": "string"},
                },
                "required": ["defect", "fix"],
                "additionalProperties": False,
            },
            "cache": {
                "type": "object",
                "properties": {
                    "falsey_hit": {"type": "string"},
                    "expiry": {"type": "string"},
                    "stampede": {"type": "string"},
                },
                "required": ["falsey_hit", "expiry", "stampede"],
                "additionalProperties": False,
            },
            "async_control": {
                "type": "object",
                "properties": {
                    "defect": {"type": "string"},
                    "fix": {"type": "string"},
                },
                "required": ["defect", "fix"],
                "additionalProperties": False,
            },
        },
        "required": ["sql", "cache", "async_control"],
        "additionalProperties": False,
    }
    reasoning_schema = {
        "type": "object",
        "properties": {
            "completion_time_seconds": {"type": "number"},
            "b_finally_runs": {"type": "boolean"},
            "why": {"type": "string"},
        },
        "required": ["completion_time_seconds", "b_finally_runs", "why"],
        "additionalProperties": False,
    }
    return [
        {
            "name": "tool_route_read",
            "payload": base_payload(
                model,
                "Before doing anything else, call the read tool exactly once with path "
                "src/cache.py, offset 120, and limit 61. Provide no prose yet.",
                max_tokens=1024,
                tools=[READ_TOOL],
            ),
            "validator": validate_read,
        },
        {
            "name": "structured_debug",
            "payload": base_payload(
                model,
                "Debug this function:\n\n"
                "def normalize(ids):\n"
                "    return [str(item).strip() for item in ids if item]\n\n"
                "For input [0, None, ' 7 '], zero must be preserved and only None removed. "
                "Return only one JSON object with exactly the keys root_cause, failing_input, "
                "and replacement; failing_input must be the numeric value 0. Do not use "
                "Markdown fences.",
                max_tokens=4096,
            ),
            "validator": validate_debug,
        },
        {
            "name": "agent_implementation",
            "payload": base_payload(
                model,
                "Use the write tool exactly once to create src/ordered_map.py as a complete "
                "Python module. Implement async def ordered_map(items, worker, limit). Treat "
                "items as an ordinary synchronous iterable: materialize it before creating "
                "tasks and do not add AsyncIterable support. Validate that limit is an integer "
                "greater than zero (bool is invalid); run at most limit workers concurrently; "
                "preserve input order; and propagate the first worker exception. After creating "
                "the tasks, use asyncio.wait(..., return_when=asyncio.FIRST_EXCEPTION), inspect "
                "the completed tasks for an exception, immediately cancel every pending sibling, "
                "and await asyncio.gather(*pending, return_exceptions=True) before re-raising. "
                "Do not use an Event and do not await the tasks one at a time. Provide no prose.",
                max_tokens=8192,
                tools=[WRITE_TOOL],
            ),
            "validator": validate_implementation,
        },
        {
            "name": "long_code_review",
            "payload": base_payload(
                model,
                "Review the following module. Return only one JSON object, without Markdown, "
                "with exactly the top-level keys sql, cache, and async_control. sql and "
                "async_control must each contain defect and fix. cache must contain falsey_hit, "
                "expiry, and stampede. Identify every independent defect: cache analysis must "
                "cover falsey values, expiration, and concurrent stampedes; async analysis must "
                "cover pending sibling cancellation and awaiting cleanup.\n\n"
                + review_source(),
                max_tokens=4096,
            ),
            "validator": validate_review,
        },
        {
            "name": "cancellation_reasoning",
            "payload": base_payload(
                model,
                "In Python asyncio, tasks A and B start at t=0. A sleeps for 2 seconds and "
                "then raises ValueError. B sleeps for 5 seconds inside try/finally. A supervisor "
                "uses asyncio.wait({A, B}, return_when=asyncio.FIRST_EXCEPTION), then cancels "
                "every pending task and awaits asyncio.gather(*pending, return_exceptions=True). "
                "Return only one JSON object, without Markdown, with exactly the keys "
                "completion_time_seconds, b_finally_runs, and why. The why field must provide "
                "a completed conclusion covering cancellation, the finally block, and gather.",
                max_tokens=4096,
            ),
            "validator": validate_reasoning,
        },
        {
            "name": "targeted_edit",
            "payload": base_payload(
                model,
                "The file src/cache.py contains:\n\n"
                "def get_cached(cache, key, load):\n"
                "    value = cache.get(key)\n"
                "    if value:\n"
                "        return value\n"
                "    value = load(key)\n"
                "    cache[key] = value\n"
                "    return value\n\n"
                "Use the edit tool exactly once to replace only the condition so cached falsey "
                "values are preserved. Include the existing four-space indentation and trailing "
                "newline in old_text and new_text. Provide no prose.",
                max_tokens=1024,
                tools=[EDIT_TOOL],
            ),
            "validator": validate_edit,
        },
    ]


CORPUS_REFERENCE_RUN = Path("omp-v6-mtp3-bf16-control/depth-3/run")


def bind_corpus_payloads(
    tasks: list[dict[str, Any]], corpus: Path, model: str
) -> dict[str, Any]:
    raw_dir = corpus / CORPUS_REFERENCE_RUN / "raw"
    if not raw_dir.is_dir():
        raise RuntimeError(f"missing OMP-v6 reference requests: {raw_dir}")

    source_hashes: dict[str, str] = {}
    for index, task in enumerate(tasks, start=1):
        source_path = raw_dir / f"round-1-{index}-{task['name']}.request.json"
        if not source_path.is_file():
            raise RuntimeError(f"missing OMP-v6 request: {source_path}")
        source_bytes = source_path.read_bytes()
        source_payload = json.loads(source_bytes)
        if source_payload.get("seed") != SEED:
            raise RuntimeError(
                f"{source_path} has seed {source_payload.get('seed')!r}, expected {SEED}"
            )

        generated_payload = dict(task["payload"])
        generated_payload.pop("model", None)
        reference_payload = dict(source_payload)
        reference_payload.pop("model", None)
        if generated_payload != reference_payload:
            raise RuntimeError(
                f"embedded task {task['name']!r} differs from OMP-v6 reference"
            )

        source_payload["model"] = model
        task["payload"] = source_payload
        source_hashes[task["name"]] = sha256_bytes(source_bytes)

    return {
        "reference_run": str(corpus / CORPUS_REFERENCE_RUN),
        "request_file_sha256_by_task": source_hashes,
    }


def warmup_payload(model: str) -> dict[str, Any]:
    payload = base_payload(
        model,
        "Return exactly: WARMUP_READY",
        max_tokens=32,
    )
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-label", required=True)
    parser.add_argument("--request-limit", type=int)
    parser.add_argument("--request-start", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--thinking-token-budget", type=int)
    parser.add_argument("--task-sequence", nargs="+")
    args = parser.parse_args()
    if args.disable_thinking and args.thinking_token_budget is not None:
        parser.error("--disable-thinking conflicts with --thinking-token-budget")

    args.output.mkdir(parents=True, exist_ok=False)
    raw_dir = args.output / "raw"
    output_dir = args.output / "outputs"
    validation_dir = args.output / "validation"
    for directory in (raw_dir, output_dir, validation_dir):
        directory.mkdir(parents=True, exist_ok=False)

    model_info = fetch_json("/v1/models")["data"][0]
    model = model_info["id"]
    tasks = build_tasks(model)
    if args.disable_thinking:
        for task in tasks:
            task["payload"]["chat_template_kwargs"] = {"enable_thinking": False}
    elif args.thinking_token_budget is not None:
        for task in tasks:
            task["payload"]["thinking_token_budget"] = args.thinking_token_budget
    corpus_metadata = bind_corpus_payloads(tasks, args.corpus, model)
    if args.task_sequence:
        tasks_by_name = {task["name"]: task for task in tasks}
        unknown = set(args.task_sequence) - set(tasks_by_name)
        if unknown:
            parser.error(f"unknown task names: {sorted(unknown)}")
        tasks = [tasks_by_name[name] for name in args.task_sequence]
    else:
        tasks = tasks[args.request_start :]
        if args.request_limit is not None:
            tasks = tasks[: args.request_limit]

    warmup = run_chat_sse(warmup_payload(model), raw_dir / "warmup.response.sse")
    warmup_text = warmup.pop("text")
    warmup_reasoning = warmup.pop("reasoning")
    warmup_calls = warmup.pop("tool_calls")
    (output_dir / "warmup.txt").write_text(warmup_text, encoding="utf-8")
    (output_dir / "warmup.reasoning.txt").write_text(
        warmup_reasoning, encoding="utf-8"
    )
    warmup["passed"] = (
        warmup["finish_reason"] == "stop"
        and warmup_text.strip() == "WARMUP_READY"
        and not warmup_calls
    )

    prepared: list[dict[str, Any]] = []
    for task in tasks:
        request_bytes = json.dumps(
            task["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        prepared.append(
            {
                **task,
                "request_sha256": sha256_bytes(request_bytes),
                "request_bytes": len(request_bytes),
            }
        )

    requests: list[dict[str, Any]] = []
    metrics_before_all = parse_spec_metrics(fetch_text("/metrics"))
    workload_started = time.perf_counter()
    with GpuMemoryPoller() as memory:
        for round_index in range(1, args.rounds + 1):
            for task_index, task in enumerate(prepared, start=1):
                request_id = f"round-{round_index}-{task_index}-{task['name']}"
                request_path = raw_dir / f"{request_id}.request.json"
                request_path.write_text(
                    json.dumps(task["payload"], indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                metrics_before = parse_spec_metrics(fetch_text("/metrics"))
                generated = run_chat_sse(
                    task["payload"], raw_dir / f"{request_id}.response.sse"
                )
                metrics_after = parse_spec_metrics(fetch_text("/metrics"))
                text = generated.pop("text")
                reasoning = generated.pop("reasoning")
                tool_calls = generated.pop("tool_calls")
                (output_dir / f"{request_id}.txt").write_text(text, encoding="utf-8")
                (output_dir / f"{request_id}.reasoning.txt").write_text(
                    reasoning, encoding="utf-8"
                )
                (output_dir / f"{request_id}.tool_calls.json").write_text(
                    json.dumps(tool_calls, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                validator_input = {
                    **generated,
                    "text": text,
                    "reasoning": reasoning,
                    "tool_calls": tool_calls,
                }
                task_quality = task["validator"](
                    validator_input, validation_dir / request_id
                )
                generated.update(
                    {
                        "request_id": request_id,
                        "round": round_index,
                        "task": task["name"],
                        "request_sha256": task["request_sha256"],
                        "request_bytes": task["request_bytes"],
                        "reasoning_bytes": len(reasoning.encode("utf-8")),
                        "text_bytes": len(text.encode("utf-8")),
                        "tool_calls": tool_calls,
                        "quality": task_quality,
                        "speculative_metrics": subtract_metrics(
                            metrics_after, metrics_before
                        ),
                    }
                )
                requests.append(generated)
    workload_seconds = time.perf_counter() - workload_started
    metrics_after_all = parse_spec_metrics(fetch_text("/metrics"))

    completion_tokens = sum(
        int((request.get("usage") or {}).get("completion_tokens") or 0)
        for request in requests
    )
    decode_seconds = sum(
        float(request.get("decode_duration_seconds") or 0.0) for request in requests
    )
    e2e_seconds = sum(float(request["e2e_latency_seconds"]) for request in requests)
    result = {
        "schema_version": 1,
        "workload": "omp-six-task-v6",
        "runtime_label": args.runtime_label,
        "depth": args.depth,
        "block_size": args.block_size,
        "served_model": model,
        "max_model_len": model_info.get("max_model_len"),
        "seed": SEED,
        "rounds": args.rounds,
        "thinking_enabled": not args.disable_thinking,
        "thinking_token_budget": args.thinking_token_budget,
        "task_order": [task["name"] for task in prepared],
        "corpus": corpus_metadata,
        "warmup": warmup,
        "requests": requests,
        "quality_pass_count": sum(request["quality"]["passed"] for request in requests),
        "quality_total": len(requests),
        "all_quality_passed": all(request["quality"]["passed"] for request in requests),
        "total_prompt_tokens": sum(
            int((request.get("usage") or {}).get("prompt_tokens") or 0)
            for request in requests
        ),
        "total_completion_tokens": completion_tokens,
        "workload_wall_seconds": workload_seconds,
        "aggregate_decode_tokens_per_second": (
            max(completion_tokens - len(requests), 0) / decode_seconds
            if decode_seconds
            else None
        ),
        "aggregate_e2e_tokens_per_second": (
            completion_tokens / e2e_seconds if e2e_seconds else None
        ),
        "speculative_metrics": subtract_metrics(metrics_after_all, metrics_before_all),
        "peak_gpu_memory_mib": memory.peak_mib,
    }
    result_path = args.output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result_path)
    print(
        json.dumps(
            {
                "runtime_label": args.runtime_label,
                "depth": args.depth,
                "quality": f"{result['quality_pass_count']}/{result['quality_total']}",
                "wall_seconds": workload_seconds,
                "decode_tps": result["aggregate_decode_tokens_per_second"],
                "e2e_tps": result["aggregate_e2e_tokens_per_second"],
                "speculative_metrics": result["speculative_metrics"],
                "peak_gpu_memory_mib": result["peak_gpu_memory_mib"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["all_quality_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
