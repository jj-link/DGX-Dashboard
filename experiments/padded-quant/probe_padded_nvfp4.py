#!/usr/bin/env python3
"""GPU smoke probe for the compiled padded NVFP4 quantization path."""

from __future__ import annotations

import importlib.metadata

import torch

from vllm import _custom_ops as ops

BLOCK_SIZE = 16
EXPECTED_VERSION_PREFIX = "0.20.2rc1.dev206+g23002d3f3"


def round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def recover_swizzled_scales(scale: torch.Tensor, m: int, n: int) -> torch.Tensor:
    rounded_m = round_up(m, 128)
    scale_n = n // BLOCK_SIZE
    rounded_n = round_up(scale_n, 4)
    tmp = scale.reshape(1, rounded_m // 128, rounded_n // 4, 32, 4, 4)
    tmp = tmp.permute(0, 1, 4, 3, 2, 5)
    return tmp.reshape(rounded_m, rounded_n)[:m, :scale_n].to(torch.float32)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    version = importlib.metadata.version("vllm")
    require(
        version.startswith(EXPECTED_VERSION_PREFIX),
        f"unexpected vLLM revision: {version}",
    )
    require(torch.cuda.is_available(), "CUDA is unavailable")
    capability = torch.cuda.get_device_capability()
    require(capability >= (12, 0), f"NVFP4 probe requires SM120+, got SM{capability[0]}{capability[1]}")

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    m, n = 150, 48
    padded_n = round_up(n, 32)
    x = torch.randn((m, n), dtype=torch.float16, device=device)
    tensor_amax = x.abs().max().to(torch.float32)
    global_scale = torch.tensor(448.0 * 6.0, device=device) / tensor_amax

    for swizzled in (False, True):
        reference, reference_scale = ops.scaled_fp4_quant(
            x,
            global_scale,
            is_sf_swizzled_layout=swizzled,
        )
        padded, padded_scale = ops.scaled_fp4_quant(
            x,
            global_scale,
            is_sf_swizzled_layout=swizzled,
            padded_n=padded_n,
        )

        require(padded.shape == (m, padded_n // 2), f"wrong output shape: {padded.shape}")
        require(
            torch.equal(padded[:, : n // 2], reference),
            f"valid packed FP4 payload changed (swizzled={swizzled})",
        )
        require(
            torch.count_nonzero(padded[:, n // 2 :]).item() == 0,
            f"packed FP4 padding is not zero (swizzled={swizzled})",
        )

        if swizzled:
            reference_linear = recover_swizzled_scales(reference_scale, m, n)
            padded_linear = recover_swizzled_scales(padded_scale, m, padded_n)
        else:
            reference_linear = reference_scale.to(torch.float32)
            padded_linear = padded_scale.to(torch.float32)

        require(
            torch.equal(padded_linear[:, : n // BLOCK_SIZE], reference_linear),
            f"valid block scales changed (swizzled={swizzled})",
        )
        require(
            torch.count_nonzero(padded_linear[:, n // BLOCK_SIZE :]).item() == 0,
            f"block-scale padding is not zero (swizzled={swizzled})",
        )

    print(
        f"PASS: compiled padded NVFP4 output on {torch.cuda.get_device_name(0)}; "
        f"vLLM {version}, shape {(m, n)} -> {(m, padded_n)}"
    )


if __name__ == "__main__":
    main()
