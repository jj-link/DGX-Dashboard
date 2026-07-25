#!/usr/bin/env python3
"""Bitwise equivalence and validation probe for grouped-weight vLLM RMSNorm.

This intentionally runs only the custom RMSNorm op; it does not load a model.
A successful run proves that the grouped launch selects the same per-layer
weights as the former Python loop for the DFlash tensor layout.
"""

from __future__ import annotations

import argparse
import sys

import torch

from vllm import __version__ as vllm_version
from vllm import _custom_ops as ops


SHAPES = (
    (28, 17, 128),
    (1, 5, 2, 128),
    (28, 13, 8, 128),  # [layers, context tokens, KV heads, head dim]
    (6, 3, 4, 769),
)
DTYPES = (torch.float16, torch.bfloat16, torch.float32)
EPSILON = 1e-6


def _randn(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    # Generate on CPU from a fixed generator so every invocation has identical
    # inputs without changing process-global RNG or the default device.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42 + sum((i + 1) * size for i, size in enumerate(shape)))
    return torch.randn(shape, dtype=dtype, generator=generator).cuda()


@torch.inference_mode()
def check_grouped_equivalence(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    rows, hidden = shape[0], shape[-1]
    x = _randn(shape, dtype) * 0.1
    weight = _randn((rows, hidden), dtype) * 0.1 + 1.0

    looped = torch.empty_like(x)
    for row in range(rows):
        ops.rms_norm(looped[row], x[row], weight[row], EPSILON)

    grouped = torch.empty_like(x)
    ops.rms_norm(grouped, x, weight, EPSILON)

    if not torch.equal(grouped, looped):
        mismatch = torch.count_nonzero(grouped != looped).item()
        max_abs = (grouped.float() - looped.float()).abs().max().item()
        raise AssertionError(
            f"grouped RMSNorm differs from loop: shape={shape}, dtype={dtype}, "
            f"mismatches={mismatch}, max_abs={max_abs}"
        )


@torch.inference_mode()
def check_broadcast_compatibility() -> None:
    x = _randn((7, 3, 128), torch.float16)
    weight = _randn((128,), torch.float16) * 0.1 + 1.0
    whole = torch.empty_like(x)
    ops.rms_norm(whole, x, weight, EPSILON)
    looped = torch.empty_like(x)
    for row in range(x.shape[0]):
        ops.rms_norm(looped[row], x[row], weight, EPSILON)
    if not torch.equal(whole, looped):
        raise AssertionError("legacy 1-D broadcast-weight RMSNorm behavior changed")


def _expect_runtime_error(label: str, call) -> None:
    try:
        call()
    except RuntimeError:
        return
    raise AssertionError(f"{label} did not raise RuntimeError")


@torch.inference_mode()
def check_shape_validation() -> None:
    x = _randn((4, 8, 128), torch.float32)
    out = torch.empty_like(x)
    _expect_runtime_error(
        "group-count mismatch",
        lambda: ops.rms_norm(out, x, torch.ones((3, 128), device="cuda"), EPSILON),
    )
    _expect_runtime_error(
        "hidden-size mismatch",
        lambda: ops.rms_norm(out, x, torch.ones((4, 64), device="cuda"), EPSILON),
    )
    _expect_runtime_error(
        "invalid weight rank",
        lambda: ops.rms_norm(out, x, torch.ones((1, 4, 128), device="cuda"), EPSILON),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="check only the deployed DFlash shape in BF16",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for vLLM's RMSNorm custom op")

    cases = [((28, 13, 8, 128), torch.bfloat16)] if args.quick else [
        (shape, dtype) for shape in SHAPES for dtype in DTYPES
    ]
    for shape, dtype in cases:
        check_grouped_equivalence(shape, dtype)
        print(f"PASS grouped==loop shape={shape} dtype={dtype}")

    check_broadcast_compatibility()
    print("PASS legacy 1-D broadcast compatibility")
    check_shape_validation()
    print("PASS grouped-weight shape validation")
    print(f"PASS grouped K-normalization probe (vLLM {vllm_version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
