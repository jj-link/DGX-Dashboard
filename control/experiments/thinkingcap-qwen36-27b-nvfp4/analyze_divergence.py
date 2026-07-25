#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

TASKS = (
    "tool_route_read",
    "structured_debug",
    "agent_implementation",
    "long_code_review",
    "cancellation_reasoning",
    "targeted_edit",
)


def tokens(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def token_window(tokenizer: Any, values: list[int], index: int | None) -> dict[str, Any]:
    if index is None:
        return {"ids": [], "text": ""}
    start = max(index - 8, 0)
    end = min(index + 9, len(values))
    selected = values[start:end]
    return {
        "range": [start, end],
        "ids": selected,
        "pieces": [tokenizer.decode([token]) for token in selected],
        "text": tokenizer.decode(selected),
    }


def compare(tokenizer: Any, left_text: str, right_text: str) -> dict[str, Any]:
    left = tokens(tokenizer, left_text)
    right = tokens(tokenizer, right_text)
    difference = first_difference(left, right)
    common = min(len(left), len(right)) if difference is None else difference
    return {
        "left_tokens": len(left),
        "right_tokens": len(right),
        "first_difference": difference,
        "common_prefix_text": tokenizer.decode(left[:common]),
        "left_window": token_window(tokenizer, left, difference),
        "right_window": token_window(tokenizer, right, difference),
    }


def artifact(root: Path, round_number: int, task_index: int, task: str, suffix: str) -> str:
    return (root / f"round-{round_number}-{task_index}-{task}.{suffix}").read_text(
        encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--plain", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result: dict[str, Any] = {"label": args.label, "round": args.round, "tasks": []}
    for task_index, task in enumerate(TASKS, 1):
        plain_reasoning = artifact(
            args.plain, args.round, task_index, task, "reasoning.txt"
        )
        candidate_reasoning = artifact(
            args.candidate, args.round, task_index, task, "reasoning.txt"
        )
        plain_content = artifact(args.plain, args.round, task_index, task, "txt")
        candidate_content = artifact(
            args.candidate, args.round, task_index, task, "txt"
        )
        result["tasks"].append(
            {
                "task": task,
                "reasoning": compare(tokenizer, plain_reasoning, candidate_reasoning),
                "content": compare(tokenizer, plain_content, candidate_content),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for task in result["tasks"]:
        print(
            task["task"],
            "reasoning_first_diff=",
            task["reasoning"]["first_difference"],
            "content_first_diff=",
            task["content"]["first_difference"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
