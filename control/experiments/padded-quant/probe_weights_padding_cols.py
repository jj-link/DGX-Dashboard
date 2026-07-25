#!/usr/bin/env python3
"""Statically prove that weights_padding_cols can drive padded NVFP4 output."""

from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path

EXPECTED_REVISION = "23002d3f368a5a24641301bc71e4ae15dae89a24"
BACKENDS = {
    "vllm/model_executor/kernels/linear/nvfp4/cutlass.py": 1,
    "vllm/model_executor/kernels/linear/nvfp4/flashinfer.py": 2,
}


def call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def is_padding_producer(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "layer"
            and target.attr == "weights_padding_cols"
            for target in node.targets
        )
    )


def is_padding_getter(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and call_name(node) == "getattr"
        and len(node.args) == 3
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "layer"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "weights_padding_cols"
        and isinstance(node.args[2], ast.Constant)
        and node.args[2].value == 0
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def inspect_apply_weights(path: Path, phase: str, expected_count: int) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    producers = sum(is_padding_producer(node) for node in ast.walk(tree))
    require(
        producers == expected_count,
        f"{path}: expected {expected_count} weights_padding_cols producers, found {producers}",
    )

    all_apply_methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "apply_weights"
    ]
    methods = [
        method
        for method in all_apply_methods
        if any(is_padding_getter(node) for node in ast.walk(method))
    ]
    require(
        len(methods) == expected_count,
        f"{path}: expected {expected_count} padded apply_weights paths, found {len(methods)}",
    )

    for method in methods:
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        getters = [node for node in calls if is_padding_getter(node)]
        old_padding_calls = [
            node for node in calls if call_name(node) == "pad_nvfp4_activation_for_cutlass"
        ]
        quant_calls = [node for node in calls if call_name(node) == "scaled_fp4_quant"]
        require(len(getters) == 1, f"{path}:{method.lineno}: missing unambiguous padding getter")
        require(len(quant_calls) == 1, f"{path}:{method.lineno}: expected one quant call")
        padded_keywords = [
            keyword.value
            for keyword in quant_calls[0].keywords
            if keyword.arg == "padded_n"
        ]

        if phase == "pre":
            require(
                len(old_padding_calls) == 1,
                f"{path}:{method.lineno}: deployed code no longer has the expected copy padding call",
            )
            require(
                not padded_keywords,
                f"{path}:{method.lineno}: padded_n is already wired; patch assumptions changed",
            )
        else:
            require(
                not old_padding_calls,
                f"{path}:{method.lineno}: obsolete activation padding copy remains",
            )
            require(
                len(padded_keywords) == 1,
                f"{path}:{method.lineno}: padded_n is not wired exactly once",
            )
            expression = ast.unparse(padded_keywords[0]).replace(" ", "")
            require(
                expression == "x.shape[-1]+weights_padding_bytes*2",
                f"{path}:{method.lineno}: unexpected padded_n expression {expression!r}",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pre", "post"), required=True)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()

    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(
        revision == EXPECTED_REVISION,
        f"wrong source revision: expected {EXPECTED_REVISION}, got {revision}",
    )

    for relative_path, expected_count in BACKENDS.items():
        inspect_apply_weights(source / relative_path, args.phase, expected_count)

    print(
        f"PASS: {args.phase}-patch weights_padding_cols contract at {EXPECTED_REVISION} "
        f"({sum(BACKENDS.values())} NVFP4 apply paths)"
    )


if __name__ == "__main__":
    main()
