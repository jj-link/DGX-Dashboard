#!/usr/bin/env python3
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def parse_sse(path: Path) -> list[str]:
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        for choice in event.get("choices") or []:
            content = choice.get("delta", {}).get("content")
            if content:
                chunks.append(content)
    return chunks


def common_prefix_characters(left: str, right: str) -> int:
    count = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        count += 1
    return count


def common_prefix_tokens(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def canonical(text: str) -> str:
    return text.rstrip() + "\n"


def stream_steps(
    *,
    tokenizer: Any,
    text: str,
    chunks: list[str],
) -> list[dict[str, Any]]:
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    token_end_offsets = [end for _, end in encoding["offset_mapping"]]
    result: list[dict[str, Any]] = []
    character_end = 0
    tokens_before = 0
    for step, chunk in enumerate(chunks, 1):
        character_end += len(chunk)
        tokens_after = bisect_right(token_end_offsets, character_end)
        emitted = tokens_after - tokens_before
        result.append(
            {
                "stream_step": step,
                "character_start": character_end - len(chunk),
                "character_end": character_end,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "tokens_emitted": emitted,
                "accepted_drafts_inferred": max(emitted - 1, 0),
                "chunk": chunk,
            }
        )
        tokens_before = tokens_after
    return result


def analyze_request(
    *,
    tokenizer: Any,
    request: dict[str, Any],
    expected: str,
    text: str,
    chunks: list[str],
) -> dict[str, Any]:
    canonical_expected = canonical(expected)
    canonical_actual = canonical(text)
    expected_tokens = tokenizer.encode(canonical_expected, add_special_tokens=False)
    actual_tokens = tokenizer.encode(canonical_actual, add_special_tokens=False)
    common_chars = common_prefix_characters(canonical_expected, canonical_actual)
    common_tokens = common_prefix_tokens(expected_tokens, actual_tokens)
    steps = stream_steps(tokenizer=tokenizer, text=text, chunks=chunks)

    first_divergent_step = None
    if not request["quality"]["exact_expected_output"]:
        for step in steps:
            if step["character_end"] > common_chars:
                first_divergent_step = step
                break

    boundary_steps: list[dict[str, Any]] = []
    boundary_offsets = request["quality"]["boundary_analysis"][
        "completion_token_offsets_of_boundaries"
    ]
    for ordinal, boundary_offset in enumerate(boundary_offsets, 1):
        crossing_step = next(
            (step for step in steps if step["tokens_after"] >= boundary_offset),
            None,
        )
        boundary_steps.append(
            {
                "ordinal": ordinal,
                "completion_token_offset": boundary_offset,
                "crossing_step": crossing_step,
                "output_diverged_before_boundary": (
                    common_tokens < boundary_offset
                    and not request["quality"]["exact_expected_output"]
                ),
            }
        )

    return {
        "request_id": request["request_id"],
        "round": request["round"],
        "case": request["case"],
        "passed": request["quality"]["passed"],
        "failure_reasons": request["quality"]["failure_reasons"],
        "exact_expected_output": request["quality"]["exact_expected_output"],
        "prompt_tokens": request["usage"]["prompt_tokens"],
        "completion_tokens": request["usage"]["completion_tokens"],
        "output_sha256": request["output_sha256"],
        "common_prefix_characters": common_chars,
        "common_prefix_completion_tokens": common_tokens,
        "first_divergent_stream_step": first_divergent_step,
        "boundary_steps": boundary_steps,
        "multi_token_stream_steps": [
            step for step in steps if step["tokens_emitted"] > 1
        ],
        "speculative_metrics": request["speculative_metrics"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--runtime", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    corpus_manifest = json.loads((args.corpus / "manifest.json").read_text())
    expected_by_case = {
        case["name"]: (args.corpus / case["expected_path"]).read_text()
        for case in corpus_manifest["cases"]
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "model": str(args.model),
        "corpus": str(args.corpus),
        "runtimes": {},
    }

    for runtime_arg in args.runtime:
        if "=" not in runtime_arg:
            raise ValueError("--runtime must be LABEL=PATH")
        label, raw_root = runtime_arg.split("=", 1)
        root = Path(raw_root)
        runtime_report: dict[str, Any] = {"root": str(root), "depths": {}}
        for depth_dir in sorted(
            root.glob("depth-*"), key=lambda path: int(path.name.split("-")[-1])
        ):
            result_path = depth_dir / "run" / "result.json"
            if not result_path.is_file():
                continue
            result = json.loads(result_path.read_text())
            requests: list[dict[str, Any]] = []
            for request in result["requests"]:
                request_id = request["request_id"]
                text = (depth_dir / "run" / "outputs" / f"{request_id}.txt").read_text()
                chunks = parse_sse(
                    depth_dir / "run" / "raw" / f"{request_id}.response.sse"
                )
                requests.append(
                    analyze_request(
                        tokenizer=tokenizer,
                        request=request,
                        expected=expected_by_case[request["case"]],
                        text=text,
                        chunks=chunks,
                    )
                )
            runtime_report["depths"][str(result["depth"])] = {
                "block_size": result["block_size"],
                "quality_pass_count": result["quality_pass_count"],
                "quality_total": result["quality_total"],
                "all_quality_passed": result["all_quality_passed"],
                "all_requests_crossed_minimum_boundaries": result[
                    "all_requests_crossed_minimum_boundaries"
                ],
                "unique_output_hashes": len(
                    {request["output_sha256"] for request in result["requests"]}
                ),
                "speculative_metrics": result["speculative_metrics"],
                "requests": requests,
            }
        report["runtimes"][label] = runtime_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
