#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HELPER_COUNT = 110
ROUNDS = 3
MINIMUM_DECODE_BOUNDARIES = 3
SEED = 424242
MAX_TOKENS = 7000
CASES = ("front", "middle", "end")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def helper_source(index: int) -> str:
    multiplier = 17 + (index % 19)
    increment = 31 + (index * 7 % 101)
    return (
        f"def helper_{index:04d}(value: int) -> int:\n"
        f'    """Unrelated deterministic helper {index:04d}."""\n'
        f"    return (value * {multiplier} + {increment}) % 1000003\n\n"
    )


def target_source(case: str, *, fixed: bool) -> str:
    if case == "front":
        expression = "limit_text" if fixed else "name"
        return (
            "def parse_window(spec: str) -> tuple[str, int]:\n"
            '    """Parse NAME:LIMIT with a positive integer limit."""\n'
            '    name, limit_text = spec.split(":", 1)\n'
            "    if not name or not limit_text.isdigit() or int(limit_text) < 1:\n"
            '        raise ValueError("invalid window")\n'
            f"    return name, int({expression})\n\n"
        )
    if case == "middle":
        expression = "limit" if fixed else "max(limit - 1, 0)"
        return (
            "def take_prefix(values: list[int], limit: int) -> list[int]:\n"
            '    """Return exactly the first limit values."""\n'
            "    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:\n"
            '        raise ValueError("invalid limit")\n'
            f"    return values[:{expression}]\n\n"
        )
    if case == "end":
        expression = "expected" if fixed else "expected.upper()"
        return (
            "def contains_token(tokens: list[str], expected: str) -> bool:\n"
            '    """Perform an exact, case-sensitive membership check."""\n'
            "    if not isinstance(expected, str):\n"
            '        raise TypeError("expected must be str")\n'
            f"    return {expression} in tokens\n\n"
        )
    raise ValueError(f"unknown case: {case}")


def module_source(case: str, *, fixed: bool) -> str:
    slots = {"front": 20, "middle": HELPER_COUNT // 2, "end": HELPER_COUNT - 20}
    chunks = [
        "from __future__ import annotations\n\n",
        'MODULE_SENTINEL = "qwen27-multiboundary-v1"\n\n',
    ]
    for index in range(HELPER_COUNT + 1):
        if index == slots[case]:
            chunks.append(target_source(case, fixed=fixed))
        if index < HELPER_COUNT:
            chunks.append(helper_source(index))
    return "".join(chunks)


def instructions(case: str) -> tuple[str, str]:
    if case == "front":
        return (
            "Fix parse_window so the integer limit is parsed from limit_text.",
            "The only permitted change is replacing `return name, int(name)` with "
            "`return name, int(limit_text)`.",
        )
    if case == "middle":
        return (
            "Fix take_prefix so it returns exactly the first limit values.",
            "The only permitted change is replacing "
            "`return values[:max(limit - 1, 0)]` with `return values[:limit]`.",
        )
    if case == "end":
        return (
            "Fix contains_token so membership is exact and case-sensitive.",
            "The only permitted change is replacing "
            "`return expected.upper() in tokens` with `return expected in tokens`.",
        )
    raise ValueError(f"unknown case: {case}")


def request_payload(case: str, buggy_source: str) -> dict[str, Any]:
    summary, exact_change = instructions(case)
    user = (
        f"{summary} {exact_change} Return the complete corrected file as raw Python "
        "source. Do not use Markdown fences. Do not add explanations. Preserve every "
        "other character and line exactly.\n\nBEGIN FILE\n"
        f"{buggy_source}"
        "END FILE\n"
    )
    return {
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a deterministic full-file editing agent. Obey the exact "
                    "output contract and complete the entire file."
                ),
            },
            {"role": "user", "content": user},
        ],
        "model": "__SERVED_MODEL__",
        "seed": SEED,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "top_p": 1.0,
    }


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    manifest_cases: list[dict[str, Any]] = []
    for case in CASES:
        buggy = module_source(case, fixed=False).encode("utf-8")
        expected = module_source(case, fixed=True).encode("utf-8")
        request = request_payload(case, buggy.decode("utf-8"))
        request_bytes = (
            json.dumps(request, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        write_new(args.output / f"{case}.input.py", buggy)
        write_new(args.output / f"{case}.expected.py", expected)
        write_new(args.output / f"{case}.request.json", request_bytes)
        manifest_cases.append(
            {
                "name": case,
                "input_path": f"{case}.input.py",
                "input_sha256": sha256(buggy),
                "expected_path": f"{case}.expected.py",
                "expected_sha256": sha256(expected),
                "request_path": f"{case}.request.json",
                "request_sha256": sha256(request_bytes),
            }
        )

    manifest = {
        "schema_version": 1,
        "purpose": "Equal repeated multi-boundary 27B DFlash depth diagnosis",
        "helper_count": HELPER_COUNT,
        "rounds": ROUNDS,
        "minimum_decode_boundaries": MINIMUM_DECODE_BOUNDARIES,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "case_order": list(CASES),
        "cases": manifest_cases,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    write_new(args.output / "manifest.json", manifest_bytes)
    print(args.output / "manifest.json")
    print(sha256(manifest_bytes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
