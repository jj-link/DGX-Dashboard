#!/usr/bin/env python3
"""Record a fixed OpenAI-completions benchmark with auditable raw evidence."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request


EXPECTED_WORKLOAD: dict[str, Any] = {
    "prompt": (
        "Write a long coherent technical essay of at least 5,000 tokens about GPU "
        "inference optimization, kernel fusion, quantization, and speculative "
        "decoding. Develop every section fully and do not conclude early."
    ),
    "max_tokens": 5000,
    "temperature": 0.3,
    "top_p": 0.95,
    "seed": 42,
}
EXPECTED_COMPLETION_TOKENS = 5000
WARMUP_ROUNDS = 1
MEASURED_ROUNDS = 3
HTTP_TIMEOUT_SECONDS = 3600
GPU_POLL_INTERVAL_SECONDS = 0.1
SPECULATIVE_CONFIG_RE = re.compile(
    r"(?:speculat|spec[-_ ]?decod|draft|eagle|mtp|medusa|ngram|dflash)", re.IGNORECASE
)
COMPILATION_RE = re.compile(
    r"(?:\bJIT\b|just[- ]in[- ]time|torch(?:\.|_)?compile|torch\._dynamo|"
    r"torch\._inductor|\b(?:pre)?compil(?:e|ed|er|ing|ation)\b|"
    r"triton[^\n]{0,80}\b(?:autotun|compile)|"
    r"cuda[ _-]?graph[^\n]{0,80}\bcaptur|"
    r"\bcaptur[^\n]{0,80}cuda[ _-]?graph)",
    re.IGNORECASE,
)
PROM_SAMPLE_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{.*\})?\s+"
    r"([-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|Inf))"
    r"(?:\s+\d+)?$"
)
PROM_TYPE_RE = re.compile(r"^#\s+TYPE\s+([A-Za-z_:][A-Za-z0-9_:]*)\s+(\w+)\s*$")
KNOWN_VLLM_SPEC_COUNTERS = {
    "vllm:spec_decode_num_accepted_tokens": "accepted_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "accepted_tokens",
    "vllm:spec_decode_num_accepted_tokens_per_pos": "accepted_tokens_per_position",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total": "accepted_tokens_per_position",
    "vllm:spec_decode_num_draft_tokens": "draft_tokens",
    "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
    "vllm:spec_decode_num_drafts": "drafts",
    "vllm:spec_decode_num_drafts_total": "drafts",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_bytes_new(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)


def write_json_new(path: Path, value: Any) -> None:
    write_bytes_new(path, json_bytes(value))


def relative_path(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def api_headers() -> dict[str, str]:
    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def evidence_headers(headers: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() != "authorization"}


def endpoints(base_url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--base-url must be an absolute http:// or https:// URL")
    if parsed.query or parsed.fragment:
        raise ValueError("--base-url must not contain a query string or fragment")

    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        service_path = path[:-3]
        completions_path = f"{path}/completions"
    else:
        service_path = path
        completions_path = f"{path}/v1/completions"
    metrics_path = f"{service_path}/metrics"
    completions_url = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, completions_path, "", "")
    )
    metrics_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, metrics_path, "", ""))
    return completions_url, metrics_url


def load_workload(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        workload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid benchmark request JSON: {exc}") from exc
    if workload != EXPECTED_WORKLOAD:
        raise ValueError(
            "benchmark-request.json does not exactly match the fixed prompt, max_tokens, "
            "temperature, top_p, and seed"
        )
    if not isinstance(workload, dict):
        raise ValueError("benchmark-request.json must contain one JSON object")
    return workload, raw


def request_payload(workload: dict[str, Any], model: str) -> dict[str, Any]:
    payload = dict(workload)
    payload["model"] = model
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    return payload


def _usage_integer(usage: dict[str, Any] | None, key: str) -> int | None:
    if not usage:
        return None
    value = usage.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def perform_request(
    *,
    request_id: str,
    phase: str,
    round_number: int,
    request_index: int,
    completions_url: str,
    payload: dict[str, Any],
    raw_dir: Path,
    outputs_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    request_path = raw_dir / f"{request_id}.request.json"
    response_path = raw_dir / f"{request_id}.response.sse"
    trace_path = raw_dir / f"{request_id}.stream-trace.jsonl"
    http_path = raw_dir / f"{request_id}.http.json"
    output_path = outputs_dir / f"{request_id}.txt"
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    write_bytes_new(request_path, payload_bytes + b"\n")

    headers = api_headers()
    request = urllib.request.Request(
        completions_url, data=payload_bytes, headers=headers, method="POST"
    )
    started_wall = utc_now()
    started_perf = time.perf_counter()
    first_token_perf: float | None = None
    status: int | None = None
    response_headers: list[list[str]] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    text_parts: list[str] = []
    stream_errors: list[str] = []
    saw_done = False
    choice_index: int | None = None
    event_data_lines: list[bytes] = []
    event_elapsed = 0.0

    def process_event() -> None:
        nonlocal first_token_perf, finish_reason, usage, saw_done, choice_index
        if not event_data_lines:
            return
        raw_data = b"\n".join(event_data_lines)
        event_data_lines.clear()
        if saw_done:
            stream_errors.append("received SSE data after the [DONE] event")
            return
        if raw_data.strip() == b"[DONE]":
            saw_done = True
            return
        try:
            decoded = raw_data.decode("utf-8")
            event = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            stream_errors.append(f"invalid SSE JSON event: {exc}")
            return
        if not isinstance(event, dict):
            stream_errors.append("SSE data event was not a JSON object")
            return
        if event.get("error") is not None:
            stream_errors.append(
                "server error event: "
                + json.dumps(event["error"], sort_keys=True, ensure_ascii=False)
            )
        event_usage = event.get("usage")
        if event_usage is not None:
            if isinstance(event_usage, dict):
                usage = event_usage
            else:
                stream_errors.append("stream usage was not a JSON object")
        choices = event.get("choices", [])
        if not isinstance(choices, list):
            stream_errors.append("stream choices was not a JSON array")
            return
        if len(choices) > 1:
            stream_errors.append("stream returned more than one completion choice")
        for choice in choices:
            if not isinstance(choice, dict):
                stream_errors.append("stream choice was not a JSON object")
                continue
            index = choice.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                stream_errors.append("completion choice index was absent or invalid")
            elif index != 0:
                stream_errors.append(f"completion choice index was {index}, expected 0")
            elif choice_index is None:
                choice_index = index
            elif choice_index != index:
                stream_errors.append(
                    f"completion choice index changed from {choice_index} to {index}"
                )
            text = choice.get("text", "")
            if not isinstance(text, str):
                stream_errors.append("completion choice text was not a string")
            elif text:
                if first_token_perf is None:
                    first_token_perf = started_perf + event_elapsed
                text_parts.append(text)
            reason = choice.get("finish_reason")
            if reason is not None:
                if not isinstance(reason, str):
                    stream_errors.append("finish_reason was not a string")
                elif finish_reason is not None and finish_reason != reason:
                    stream_errors.append(
                        f"conflicting finish reasons {finish_reason!r} and {reason!r}"
                    )
                else:
                    finish_reason = reason

    try:
        with response_path.open("xb") as response_file, trace_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as trace_file:
            try:
                with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                    status = response.status
                    response_headers = [[key, value] for key, value in response.headers.items()]
                    while True:
                        line = response.readline()
                        if not line:
                            break
                        event_elapsed = time.perf_counter() - started_perf
                        response_file.write(line)
                        trace_file.write(
                            json.dumps(
                                {
                                    "elapsed_seconds": event_elapsed,
                                    "line_base64": base64.b64encode(line).decode("ascii"),
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        stripped = line.rstrip(b"\r\n")
                        if not stripped:
                            process_event()
                        elif stripped.startswith(b"data:"):
                            data = stripped[5:]
                            if data.startswith(b" "):
                                data = data[1:]
                            event_data_lines.append(data)
                    process_event()
            except urllib.error.HTTPError as exc:
                status = exc.code
                response_headers = [[key, value] for key, value in exc.headers.items()]
                body = exc.read()
                response_file.write(body)
                trace_file.write(
                    json.dumps(
                        {
                            "elapsed_seconds": time.perf_counter() - started_perf,
                            "line_base64": base64.b64encode(body).decode("ascii"),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                stream_errors.append(f"HTTP {exc.code}: {exc.reason}")
            except Exception as exc:  # Network, timeout, TLS, and stream read failures are evidence.
                stream_errors.append(f"HTTP/stream error: {type(exc).__name__}: {exc}")
    except Exception:
        # Artifact write failures are fatal: a request without raw evidence is not recordable.
        raise

    ended_perf = time.perf_counter()
    ended_wall = utc_now()
    output_bytes = "".join(text_parts).encode("utf-8")
    write_bytes_new(output_path, output_bytes)
    prompt_tokens = _usage_integer(usage, "prompt_tokens")
    completion_tokens = _usage_integer(usage, "completion_tokens")
    total_tokens = _usage_integer(usage, "total_tokens")
    e2e_latency = ended_perf - started_perf
    ttft = None if first_token_perf is None else first_token_perf - started_perf
    decode_duration = None if first_token_perf is None else ended_perf - first_token_perf
    decode_token_count = None if completion_tokens is None else max(completion_tokens - 1, 0)

    invalid_reasons = list(stream_errors)
    if status != 200:
        invalid_reasons.append(f"expected HTTP 200, received {status!r}")
    if not saw_done:
        invalid_reasons.append("stream ended without a [DONE] event")
    if first_token_perf is None:
        invalid_reasons.append("stream contained no non-empty completion text")
    if prompt_tokens is None:
        invalid_reasons.append("prompt_tokens was absent or invalid")
    if completion_tokens != EXPECTED_COMPLETION_TOKENS:
        invalid_reasons.append(
            f"completion_tokens was {completion_tokens!r}, expected {EXPECTED_COMPLETION_TOKENS}"
        )
    if (
        total_tokens is not None
        and prompt_tokens is not None
        and completion_tokens is not None
        and total_tokens != prompt_tokens + completion_tokens
    ):
        invalid_reasons.append(
            f"total_tokens was {total_tokens}, expected "
            f"{prompt_tokens + completion_tokens} from prompt_tokens + completion_tokens"
        )
    if finish_reason != "length":
        invalid_reasons.append(f"finish_reason was {finish_reason!r}, expected 'length'")

    result: dict[str, Any] = {
        "request_id": request_id,
        "phase": phase,
        "round": round_number,
        "request_index": request_index,
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "started_at": started_wall,
        "finished_at": ended_wall,
        "http_status": status,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "ttft_seconds": ttft,
        "e2e_latency_seconds": e2e_latency,
        "decode_duration_seconds": decode_duration,
        "e2e_tokens_per_second": (
            None if completion_tokens is None or e2e_latency <= 0 else completion_tokens / e2e_latency
        ),
        "decode_tokens_per_second": (
            None
            if decode_token_count is None or decode_duration is None or decode_duration <= 0
            else decode_token_count / decode_duration
        ),
        "output_bytes": len(output_bytes),
        "output_sha256": sha256_bytes(output_bytes),
        "artifacts": {
            "request": relative_path(request_path, output_dir),
            "response_sse": relative_path(response_path, output_dir),
            "stream_trace": relative_path(trace_path, output_dir),
            "http_metadata": relative_path(http_path, output_dir),
            "output_text": relative_path(output_path, output_dir),
        },
        "_started_perf": started_perf,
        "_first_token_perf": first_token_perf,
        "_ended_perf": ended_perf,
    }
    write_json_new(
        http_path,
        {
            "request_id": request_id,
            "url": completions_url,
            "method": "POST",
            "request_headers": evidence_headers(headers),
            "status": status,
            "response_headers": response_headers,
            "started_at": started_wall,
            "finished_at": ended_wall,
            "saw_done": saw_done,
            "stream_errors": stream_errors,
            "finish_reason": finish_reason,
            "usage": usage,
            "output_sha256": result["output_sha256"],
        },
    )
    return result


def run_round(
    *,
    phase: str,
    round_number: int,
    concurrency: int,
    completions_url: str,
    payload: dict[str, Any],
    raw_dir: Path,
    outputs_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    barrier = threading.Barrier(concurrency + 1)

    def worker(request_index: int) -> dict[str, Any]:
        barrier.wait()
        request_id = f"{phase}-r{round_number}-q{request_index:02d}"
        return perform_request(
            request_id=request_id,
            phase=phase,
            round_number=round_number,
            request_index=request_index,
            completions_url=completions_url,
            payload=payload,
            raw_dir=raw_dir,
            outputs_dir=outputs_dir,
            output_dir=output_dir,
        )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrency, thread_name_prefix=f"benchmark-{phase}-{round_number}"
    ) as executor:
        futures = [executor.submit(worker, index) for index in range(1, concurrency + 1)]
        round_started_wall = utc_now()
        round_started_perf = time.perf_counter()
        barrier.wait()
        results = [future.result() for future in futures]
        round_ended_perf = time.perf_counter()
        round_ended_wall = utc_now()

    completion_total = sum(
        result["completion_tokens"]
        for result in results
        if isinstance(result["completion_tokens"], int)
    )
    decode_total = sum(
        max(result["completion_tokens"] - 1, 0)
        for result in results
        if isinstance(result["completion_tokens"], int)
    )
    first_token_times = [
        result["_first_token_perf"]
        for result in results
        if result["_first_token_perf"] is not None
    ]
    wall_seconds = round_ended_perf - round_started_perf
    aggregate_decode_seconds = (
        None if not first_token_times else round_ended_perf - min(first_token_times)
    )
    for result in results:
        result["start_offset_seconds"] = result["_started_perf"] - round_started_perf
        result["first_token_offset_seconds"] = (
            None
            if result["_first_token_perf"] is None
            else result["_first_token_perf"] - round_started_perf
        )
        del result["_started_perf"]
        del result["_first_token_perf"]
        del result["_ended_perf"]

    return {
        "phase": phase,
        "round": round_number,
        "started_at": round_started_wall,
        "finished_at": round_ended_wall,
        "wall_seconds": wall_seconds,
        "valid": all(result["valid"] for result in results),
        "completion_tokens": completion_total,
        "aggregate_e2e_tokens_per_second": (
            None if wall_seconds <= 0 else completion_total / wall_seconds
        ),
        "aggregate_decode_seconds": aggregate_decode_seconds,
        "aggregate_decode_tokens_per_second": (
            None
            if aggregate_decode_seconds is None or aggregate_decode_seconds <= 0
            else decode_total / aggregate_decode_seconds
        ),
        "requests": results,
    }


def fetch_raw_metrics(
    *, url: str, label: str, raw_dir: Path, output_dir: Path
) -> dict[str, Any]:
    body_path = raw_dir / f"metrics-{label}.prom"
    metadata_path = raw_dir / f"metrics-{label}.http.json"
    headers = api_headers()
    headers["Accept"] = "text/plain"
    headers.pop("Content-Type", None)
    request = urllib.request.Request(url, headers=headers, method="GET")
    started_at = utc_now()
    started_perf = time.perf_counter()
    status: int | None = None
    response_headers: list[list[str]] = []
    error: str | None = None
    body = b""
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            response_headers = [[key, value] for key, value in response.headers.items()]
            body = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_headers = [[key, value] for key, value in exc.headers.items()]
        body = exc.read()
        error = f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    duration = time.perf_counter() - started_perf
    write_bytes_new(body_path, body)
    metadata = {
        "url": url,
        "method": "GET",
        "request_headers": evidence_headers(headers),
        "status": status,
        "response_headers": response_headers,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": duration,
        "error": error,
        "body_bytes": len(body),
        "body_sha256": sha256_bytes(body),
    }
    write_json_new(metadata_path, metadata)
    return {
        "label": label,
        "success": status == 200 and error is None,
        "status": status,
        "error": error,
        "body": body,
        "artifacts": {
            "body": relative_path(body_path, output_dir),
            "http_metadata": relative_path(metadata_path, output_dir),
        },
    }


def parse_prometheus(body: bytes) -> dict[str, Any]:
    text = body.decode("utf-8", errors="replace")
    metric_types: dict[str, str] = {}
    samples: dict[str, dict[str, Any]] = {}
    unparsed_sample_lines = 0
    non_finite_samples = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        type_match = PROM_TYPE_RE.match(stripped)
        if type_match:
            metric_types[type_match.group(1)] = type_match.group(2).lower()
            continue
        if stripped.startswith("#"):
            continue
        sample_match = PROM_SAMPLE_RE.match(stripped)
        if not sample_match:
            unparsed_sample_lines += 1
            continue
        name = sample_match.group(1)
        labels = sample_match.group(2) or ""
        try:
            value = float(sample_match.group(3))
        except ValueError:
            unparsed_sample_lines += 1
            continue
        if not math.isfinite(value):
            non_finite_samples += 1
            continue
        metric_type = metric_types.get(name)
        if metric_type is None and name.endswith("_total"):
            metric_type = metric_types.get(name[:-6])
        key = name + labels
        samples[key] = {
            "series": key,
            "name": name,
            "labels": labels or None,
            "value": value,
            "type": metric_type,
        }
    return {
        "samples": samples,
        "metric_types": metric_types,
        "sample_count": len(samples),
        "unparsed_sample_lines": unparsed_sample_lines,
        "non_finite_samples": non_finite_samples,
    }


def is_counter_sample(sample: dict[str, Any]) -> bool:
    return (
        sample.get("type") == "counter"
        or sample["name"].endswith("_total")
        or sample["name"] in KNOWN_VLLM_SPEC_COUNTERS
    )


def speculative_category(sample: dict[str, Any]) -> str | None:
    name = sample["name"].lower()
    if name in KNOWN_VLLM_SPEC_COUNTERS:
        return KNOWN_VLLM_SPEC_COUNTERS[name]
    if name in {"sglang:spec_accept_rate", "sglang_spec_accept_rate"}:
        return "accept_rate"
    if name in {"sglang:spec_accept_length", "sglang_spec_accept_length"}:
        return "accept_length"
    if name in {"sglang:spec_num_steps", "sglang_spec_num_steps"}:
        return "configured_steps"
    if name in {"sglang:spec_num_draft_tokens", "sglang_spec_num_draft_tokens"} and not is_counter_sample(sample):
        return "configured_draft_tokens"
    if not is_counter_sample(sample):
        return None
    has_spec_marker = any(
        marker in name for marker in ("spec", "draft", "eagle", "mtp", "medusa", "dflash")
    )
    if not has_spec_marker and not name.startswith("sglang"):
        return None
    if "accepted" in name and "token" in name:
        return "accepted_tokens"
    if ("draft" in name or "proposed" in name) and "token" in name:
        return "draft_tokens"
    if "draft" in name and "token" not in name:
        return "drafts"
    return None


def analyze_metrics(
    before_fetch: dict[str, Any],
    after_fetch: dict[str, Any],
    *,
    engine: str,
    speculative_configuration: bool,
) -> dict[str, Any]:
    before = parse_prometheus(before_fetch["body"])
    after = parse_prometheus(after_fetch["body"])
    before_samples = before["samples"]
    after_samples = after["samples"]
    counter_deltas: list[dict[str, Any]] = []
    counter_resets: list[str] = []
    categories: dict[str, list[dict[str, Any]]] = {}

    for series in sorted(set(before_samples) & set(after_samples)):
        before_sample = before_samples[series]
        after_sample = after_samples[series]
        if before_sample["name"] != after_sample["name"]:
            continue
        counter = is_counter_sample(before_sample) or is_counter_sample(after_sample)
        delta = after_sample["value"] - before_sample["value"] if counter else None
        if counter:
            entry = {
                "series": series,
                "name": after_sample["name"],
                "labels": after_sample["labels"],
                "before": before_sample["value"],
                "after": after_sample["value"],
                "delta": delta,
                "reset_detected": bool(delta is not None and delta < 0),
            }
            counter_deltas.append(entry)
            if entry["reset_detected"]:
                counter_resets.append(series)
        category = speculative_category(after_sample) or speculative_category(before_sample)
        if category:
            categories.setdefault(category, []).append(
                {
                    "series": series,
                    "name": after_sample["name"],
                    "labels": after_sample["labels"],
                    "kind": "counter" if counter else "gauge",
                    "before": before_sample["value"],
                    "after": after_sample["value"],
                    "delta": delta,
                    "available": True,
                }
            )

    all_category_names = {
        "accepted_tokens",
        "accepted_tokens_per_position",
        "draft_tokens",
        "drafts",
        "accept_rate",
        "accept_length",
        "configured_steps",
        "configured_draft_tokens",
    }
    availability = {
        category: {
            "available": bool(categories.get(category)),
            "series": categories.get(category, []),
        }
        for category in sorted(all_category_names | set(categories))
    }

    if engine == "vllm":
        required_alternatives = [["accepted_tokens", "draft_tokens"]]
    else:
        required_alternatives = [
            ["accepted_tokens", "draft_tokens"],
            ["accept_rate", "accept_length"],
        ]
    satisfied_by: list[str] | None = None
    inactive_required: list[str] = []
    for alternative in required_alternatives:
        if not all(availability[category]["available"] for category in alternative):
            continue
        if engine == "vllm" and "draft_tokens" in alternative:
            draft_delta = sum(
                series["delta"]
                for series in availability["draft_tokens"]["series"]
                if series["delta"] is not None and series["delta"] >= 0
            )
            if draft_delta <= 0:
                inactive_required = ["draft_tokens counter did not advance"]
                continue
        satisfied_by = alternative
        inactive_required = []
        break
    missing_required = []
    if speculative_configuration and satisfied_by is None and not inactive_required:
        missing_required = [" or ".join(" + ".join(group) for group in required_alternatives)]

    return {
        "before": {
            "success": before_fetch["success"],
            "status": before_fetch["status"],
            "error": before_fetch["error"],
            "sample_count": before["sample_count"],
            "unparsed_sample_lines": before["unparsed_sample_lines"],
            "non_finite_samples": before["non_finite_samples"],
            "artifacts": before_fetch["artifacts"],
        },
        "after": {
            "success": after_fetch["success"],
            "status": after_fetch["status"],
            "error": after_fetch["error"],
            "sample_count": after["sample_count"],
            "unparsed_sample_lines": after["unparsed_sample_lines"],
            "non_finite_samples": after["non_finite_samples"],
            "artifacts": after_fetch["artifacts"],
        },
        "counter_deltas": counter_deltas,
        "counter_resets": counter_resets,
        "speculative": {
            "configuration_detected": speculative_configuration,
            "required_alternatives": required_alternatives,
            "satisfied_by": satisfied_by,
            "missing_required": missing_required,
            "inactive_required": inactive_required if speculative_configuration else [],
            "signals": availability,
        },
    }


class GpuMemoryPoller:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-memory-poller", daemon=True)
        self.samples: list[dict[str, Any]] = []
        self.started_at: str | None = None
        self.finished_at: str | None = None

    def start(self) -> None:
        self.started_at = utc_now()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self.finished_at = utc_now()
        if self._thread.is_alive():
            self.samples.append(
                {
                    "observed_at": utc_now(),
                    "returncode": None,
                    "stdout": "",
                    "stderr": "GPU poller did not stop within 10 seconds",
                    "gpu_index": None,
                    "memory_used_mib": None,
                    "memory_total_mib": None,
                }
            )

    def _run(self) -> None:
        command = [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        while True:
            observed_at = utc_now()
            sample: dict[str, Any] = {
                "observed_at": observed_at,
                "command": command,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "gpu_index": None,
                "memory_used_mib": None,
                "memory_total_mib": None,
            }
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=5,
                    check=False,
                )
                sample["returncode"] = completed.returncode
                sample["stdout"] = completed.stdout
                sample["stderr"] = completed.stderr
                if completed.returncode == 0:
                    first_line = completed.stdout.strip().splitlines()[0]
                    fields = [field.strip() for field in first_line.split(",")]
                    if len(fields) != 3:
                        raise ValueError(f"unexpected nvidia-smi field count: {first_line!r}")
                    gpu_index, used_mib, total_mib = (int(field) for field in fields)
                    if gpu_index != 0:
                        raise ValueError(f"nvidia-smi returned GPU index {gpu_index}, expected 0")
                    sample["gpu_index"] = gpu_index
                    sample["memory_used_mib"] = used_mib
                    sample["memory_total_mib"] = total_mib
            except Exception as exc:
                sample["stderr"] = (
                    sample["stderr"] + ("\n" if sample["stderr"] else "")
                    + f"{type(exc).__name__}: {exc}"
                )
            self.samples.append(sample)
            if self._stop.wait(GPU_POLL_INTERVAL_SECONDS):
                return

    def save(self, path: Path, output_dir: Path) -> dict[str, Any]:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            for sample in self.samples:
                handle.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")
        valid_samples = [
            sample
            for sample in self.samples
            if sample["gpu_index"] == 0 and isinstance(sample["memory_used_mib"], int)
        ]
        return {
            "gpu_index": 0,
            "poll_interval_seconds": GPU_POLL_INTERVAL_SECONDS,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "sample_count": len(self.samples),
            "valid_sample_count": len(valid_samples),
            "peak_memory_used_mib": (
                None
                if not valid_samples
                else max(sample["memory_used_mib"] for sample in valid_samples)
            ),
            "memory_total_mib_values": sorted(
                {sample["memory_total_mib"] for sample in valid_samples}
            ),
            "artifact": relative_path(path, output_dir),
        }


def open_server_log_window(server_log: Path) -> dict[str, Any]:
    handle = server_log.open("rb")
    stat = os.fstat(handle.fileno())
    handle.seek(0, os.SEEK_END)
    start_offset = handle.tell()
    guard_offset = max(0, start_offset - 4096)
    handle.seek(guard_offset)
    start_guard = handle.read(start_offset - guard_offset)
    return {
        "handle": handle,
        "path": server_log,
        "start_offset": start_offset,
        "start_device": stat.st_dev,
        "start_inode": stat.st_ino,
        "start_guard_offset": guard_offset,
        "start_guard": start_guard,
        "started_at": utc_now(),
    }


def close_server_log_window(
    window: dict[str, Any] | None,
    *,
    server_log: Path,
    raw_dir: Path,
    output_dir: Path,
    open_error: str | None,
) -> dict[str, Any]:
    interval_path = raw_dir / "server-log-measured-interval.log"
    metadata_path = raw_dir / "server-log-measured-interval.json"
    data = b""
    errors: list[str] = []
    start_offset: int | None = None
    end_offset: int | None = None
    identity_changed = False
    started_at: str | None = None
    if window is None:
        errors.append(open_error or "server log interval was not opened")
    else:
        handle = window["handle"]
        started_at = window["started_at"]
        start_offset = window["start_offset"]
        try:
            handle.seek(0, os.SEEK_END)
            end_offset = handle.tell()
            guard_offset = window["start_guard_offset"]
            handle.seek(guard_offset)
            current_guard = handle.read(start_offset - guard_offset)
            if current_guard != window["start_guard"]:
                errors.append(
                    "server log contents before the measured interval changed; "
                    "copy-truncate or in-place overwrite detected"
                )
            if end_offset < start_offset:
                errors.append("server log was truncated during the measured interval")
            else:
                handle.seek(start_offset)
                data = handle.read(end_offset - start_offset)
            try:
                current_stat = server_log.stat()
                identity_changed = (
                    current_stat.st_dev != window["start_device"]
                    or current_stat.st_ino != window["start_inode"]
                )
                if identity_changed:
                    errors.append("server log was rotated or replaced during the measured interval")
            except OSError as exc:
                errors.append(f"could not stat server log at interval end: {exc}")
        except OSError as exc:
            errors.append(f"could not read measured server log interval: {exc}")
        finally:
            handle.close()

    write_bytes_new(interval_path, data)
    decoded = data.decode("utf-8", errors="replace")
    matches: list[dict[str, Any]] = []
    match_count = 0
    for line_number, line in enumerate(decoded.splitlines(), start=1):
        # SGLang logs the complete generated essay on its request-completion line.
        # The fixed benchmark topic legitimately discusses compilers and JIT, so
        # scanning echoed model text would falsely mark every such run invalid.
        if "Finish: obj=GenerateReqInput" in line:
            continue
        for match in COMPILATION_RE.finditer(line):
            match_count += 1
            if len(matches) < 200:
                matches.append(
                    {
                        "interval_line": line_number,
                        "matched_text": match.group(0),
                        "line": line,
                    }
                )
    metadata = {
        "source_path": str(server_log.resolve()),
        "started_at": started_at,
        "finished_at": utc_now(),
        "start_offset": start_offset,
        "end_offset": end_offset,
        "start_guard_offset": (
            None if window is None else window["start_guard_offset"]
        ),
        "start_guard_bytes": (
            None if window is None else len(window["start_guard"])
        ),
        "start_guard_sha256": (
            None if window is None else sha256_bytes(window["start_guard"])
        ),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "identity_changed": identity_changed,
        "errors": errors,
        "compilation_or_jit_detected": match_count > 0,
        "compilation_or_jit_match_count": match_count,
        "matches": matches,
        "matches_truncated": match_count > len(matches),
        "artifacts": {
            "interval": relative_path(interval_path, output_dir),
            "metadata": relative_path(metadata_path, output_dir),
        },
    }
    write_json_new(metadata_path, metadata)
    return metadata


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "mean": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def aggregate_measured(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [request for round_result in rounds for request in round_result["requests"]]
    total_wall = sum(round_result["wall_seconds"] for round_result in rounds)
    total_completion_tokens = sum(
        request["completion_tokens"]
        for request in requests
        if isinstance(request["completion_tokens"], int)
    )
    decode_tokens = sum(
        max(request["completion_tokens"] - 1, 0)
        for request in requests
        if isinstance(request["completion_tokens"], int)
    )
    decode_seconds = sum(
        round_result["aggregate_decode_seconds"]
        for round_result in rounds
        if isinstance(round_result["aggregate_decode_seconds"], (int, float))
    )
    return {
        "round_count": len(rounds),
        "request_count": len(requests),
        "total_completion_tokens": total_completion_tokens,
        "total_round_wall_seconds": total_wall,
        "aggregate_e2e_tokens_per_second": (
            None if total_wall <= 0 else total_completion_tokens / total_wall
        ),
        "aggregate_decode_tokens_per_second": (
            None if decode_seconds <= 0 else decode_tokens / decode_seconds
        ),
        "per_request_ttft_seconds": latency_summary(
            [request["ttft_seconds"] for request in requests if request["ttft_seconds"] is not None]
        ),
        "per_request_e2e_latency_seconds": latency_summary(
            [request["e2e_latency_seconds"] for request in requests]
        ),
        "per_request_decode_tokens_per_second": latency_summary(
            [
                request["decode_tokens_per_second"]
                for request in requests
                if request["decode_tokens_per_second"] is not None
            ]
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fixed warmup and exactly three measured streaming /v1/completions "
            "rounds, preserving outputs and raw benchmark evidence."
        )
    )
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible server base URL")
    parser.add_argument("--model", required=True, help="served model name sent in each request")
    parser.add_argument(
        "--engine", required=True, choices=("vllm", "sglang"), help="server engine"
    )
    parser.add_argument("--configuration", required=True, help="configuration label")
    parser.add_argument("--image", required=True, help="container image reference")
    parser.add_argument("--revision", required=True, help="image/model revision identifier")
    parser.add_argument(
        "--launch-command-file",
        required=True,
        type=Path,
        help="file containing the exact server launch command",
    )
    parser.add_argument(
        "--server-log",
        required=True,
        type=Path,
        help="server log to inspect over the measured interval",
    )
    parser.add_argument(
        "--concurrency",
        required=True,
        type=int,
        choices=(1, 4),
        help="simultaneous requests in each round",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new or empty result directory",
    )
    return parser


def execute(args: argparse.Namespace) -> tuple[Path, bool]:
    benchmark_request_path = Path(__file__).with_name("benchmark-request.json")
    workload, workload_raw = load_workload(benchmark_request_path)
    completions_url, metrics_url = endpoints(args.base_url)
    if not args.model.strip():
        raise ValueError("--model must not be empty")
    if not args.configuration.strip():
        raise ValueError("--configuration must not be empty")
    if not args.image.strip() or not args.revision.strip():
        raise ValueError("--image and --revision must not be empty")
    if not args.launch_command_file.is_file():
        raise ValueError(f"launch command file not found: {args.launch_command_file}")
    if not args.server_log.is_file():
        raise ValueError(f"server log not found: {args.server_log}")
    launch_command_raw = args.launch_command_file.read_bytes()
    if not launch_command_raw.strip():
        raise ValueError("launch command file is empty")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    outputs_dir = output_dir / "outputs"
    raw_dir.mkdir()
    outputs_dir.mkdir()
    write_bytes_new(raw_dir / "benchmark-request.json", workload_raw)
    write_bytes_new(raw_dir / "launch-command.txt", launch_command_raw)

    run_started_at = utc_now()
    payload = request_payload(workload, args.model)
    launch_command_text = launch_command_raw.decode("utf-8", errors="replace")
    speculative_configuration = bool(
        SPECULATIVE_CONFIG_RE.search(args.configuration)
        or SPECULATIVE_CONFIG_RE.search(launch_command_text)
    )

    warmup_rounds = [
        run_round(
            phase="warmup",
            round_number=1,
            concurrency=args.concurrency,
            completions_url=completions_url,
            payload=payload,
            raw_dir=raw_dir,
            outputs_dir=outputs_dir,
            output_dir=output_dir,
        )
    ]

    metrics_before = fetch_raw_metrics(
        url=metrics_url, label="before", raw_dir=raw_dir, output_dir=output_dir
    )
    log_window: dict[str, Any] | None = None
    log_open_error: str | None = None
    try:
        log_window = open_server_log_window(args.server_log)
    except OSError as exc:
        log_open_error = f"could not open server log interval: {exc}"

    gpu_poller = GpuMemoryPoller()
    gpu_poller.start()
    measured_rounds: list[dict[str, Any]] = []
    try:
        for round_number in range(1, MEASURED_ROUNDS + 1):
            measured_rounds.append(
                run_round(
                    phase="measured",
                    round_number=round_number,
                    concurrency=args.concurrency,
                    completions_url=completions_url,
                    payload=payload,
                    raw_dir=raw_dir,
                    outputs_dir=outputs_dir,
                    output_dir=output_dir,
                )
            )
    finally:
        gpu_poller.stop()

    server_log = close_server_log_window(
        log_window,
        server_log=args.server_log,
        raw_dir=raw_dir,
        output_dir=output_dir,
        open_error=log_open_error,
    )
    gpu_memory = gpu_poller.save(raw_dir / "gpu0-memory.jsonl", output_dir)
    metrics_after = fetch_raw_metrics(
        url=metrics_url, label="after", raw_dir=raw_dir, output_dir=output_dir
    )
    metrics = analyze_metrics(
        metrics_before,
        metrics_after,
        engine=args.engine,
        speculative_configuration=speculative_configuration,
    )

    invalid_reasons: list[str] = []
    for round_result in warmup_rounds + measured_rounds:
        for request_result in round_result["requests"]:
            for reason in request_result["invalid_reasons"]:
                invalid_reasons.append(f"{request_result['request_id']}: {reason}")
    if not metrics_before["success"]:
        invalid_reasons.append("metrics scrape before measured rounds failed")
    if not metrics_after["success"]:
        invalid_reasons.append("metrics scrape after measured rounds failed")
    if metrics_before["success"] and metrics["before"]["sample_count"] == 0:
        invalid_reasons.append("metrics scrape before measured rounds contained no parseable samples")
    if metrics_after["success"] and metrics["after"]["sample_count"] == 0:
        invalid_reasons.append("metrics scrape after measured rounds contained no parseable samples")
    if (
        metrics_before["success"]
        and metrics_after["success"]
        and not metrics["counter_deltas"]
    ):
        invalid_reasons.append("metrics scrapes contained no comparable counter series")
    if metrics["counter_resets"]:
        invalid_reasons.append("one or more metrics counters reset during measured rounds")
    if metrics["speculative"]["missing_required"]:
        invalid_reasons.append(
            "required speculative metrics were absent: "
            + ", ".join(metrics["speculative"]["missing_required"])
        )
    if metrics["speculative"]["inactive_required"]:
        invalid_reasons.append(
            "speculative decoding was not active: "
            + ", ".join(metrics["speculative"]["inactive_required"])
        )
    if gpu_memory["valid_sample_count"] == 0:
        invalid_reasons.append("no valid GPU0 memory samples were collected")
    if server_log["errors"]:
        invalid_reasons.extend(f"server log evidence: {error}" for error in server_log["errors"])
    if server_log["compilation_or_jit_detected"]:
        invalid_reasons.append("compilation/JIT activity appeared in the measured server-log interval")

    valid = not invalid_reasons
    result_path = output_dir / "benchmark-result.json"
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "valid" if valid else "invalid",
        "ranking_eligible": valid,
        "invalid_reasons": invalid_reasons,
        "run": {
            "started_at": run_started_at,
            "finished_at": utc_now(),
            "base_url": args.base_url,
            "completions_url": completions_url,
            "metrics_url": metrics_url,
            "model": args.model,
            "engine": args.engine,
            "configuration": args.configuration,
            "image": args.image,
            "revision": args.revision,
            "concurrency": args.concurrency,
            "warmup_rounds": WARMUP_ROUNDS,
            "measured_rounds": MEASURED_ROUNDS,
            "requests_per_round": args.concurrency,
            "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
            "benchmark_request_source": str(benchmark_request_path.resolve()),
            "benchmark_request_sha256": sha256_bytes(workload_raw),
            "launch_command_file": str(args.launch_command_file.resolve()),
            "launch_command_sha256": sha256_bytes(launch_command_raw),
            "server_log": str(args.server_log.resolve()),
            "speculative_configuration_detected": speculative_configuration,
        },
        "request": payload,
        "warmup": warmup_rounds,
        "measured": {
            "rounds": measured_rounds,
            "aggregate": aggregate_measured(measured_rounds),
        },
        "metrics": metrics,
        "gpu_memory": gpu_memory,
        "server_log_interval": server_log,
        "artifacts": [],
    }
    artifact_paths = sorted(
        relative_path(path, output_dir)
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    artifact_paths.append(relative_path(result_path, output_dir))
    result["artifacts"] = sorted(set(artifact_paths))
    write_json_new(result_path, result)
    return result_path, valid


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result_path, valid = execute(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {"result": str(result_path), "status": "valid" if valid else "invalid"},
            sort_keys=True,
        )
    )
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
