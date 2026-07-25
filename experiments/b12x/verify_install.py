#!/usr/bin/env python3
"""Build-time, GPU-free verification for the b12x candidate image."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata as metadata
import os
from pathlib import Path

import torch

SITE = Path("/usr/local/lib/python3.12/dist-packages")
VLLM = SITE / "vllm"
EXPECTED = {
    "flashinfer-python": "0.6.14",
    "flashinfer-cubin": "0.6.14",
    "flashinfer-jit-cache": "0.6.14+cu130",
    "nvidia-cutlass-dsl": "4.5.1",
    "nvidia-cutlass-dsl-libs-base": "4.5.1",
    "nvidia-cutlass-dsl-libs-cu13": "4.5.1",
}

for distribution, expected in EXPECTED.items():
    actual = metadata.version(distribution)
    assert actual == expected, f"{distribution}: expected {expected}, got {actual}"
    print(f"PIN {distribution}=={actual}")

assert torch.__version__.startswith("2.11."), torch.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
print(f"PIN torch=={torch.__version__} cuda={torch.version.cuda}")

version_source = (VLLM / "_version.py").read_text()
assert "0.20.2rc1.dev206+g23002d3f3" in version_source
print("PIN vllm==0.20.2rc1.dev206+g23002d3f3")

from flashinfer import gemm

assert hasattr(gemm, "Sm120B12xBlockScaledDenseGemmKernel")
print("CAPABILITY flashinfer.gemm.Sm120B12xBlockScaledDenseGemmKernel=present")

envs_source = (VLLM / "envs.py").read_text()
registry_source = (VLLM / "model_executor/kernels/linear/__init__.py").read_text()
kernel_source = (
    VLLM / "model_executor/kernels/linear/nvfp4/flashinfer.py"
).read_text()
utils_source = (VLLM / "utils/flashinfer.py").read_text()
for source in (envs_source, registry_source, utils_source):
    ast.parse(source)
kernel_tree = ast.parse(kernel_source)

assert '"flashinfer-b12x"' in envs_source
assert '"flashinfer-b12x": FlashInferB12xNvFp4LinearKernel' in registry_source
assert "class FlashInferB12xNvFp4LinearKernel" in kernel_source
b12x_calls = [
    node
    for node in ast.walk(kernel_tree)
    if isinstance(node, ast.Call)
    and any(
        keyword.arg == "backend"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == "b12x"
        for keyword in node.keywords
    )
]
assert len(b12x_calls) == 2, len(b12x_calls)
assert "Sm120B12xBlockScaledDenseGemmKernel" in utils_source
assert os.environ["VLLM_NVFP4_GEMM_BACKEND"] == "flashinfer-b12x"
print("SELECTION VLLM_NVFP4_GEMM_BACKEND=flashinfer-b12x")
print("SELECTION flashinfer-b12x -> FlashInferB12xNvFp4LinearKernel -> mm_fp4 backend=b12x")

assert 'DFlashModelTypes = Literal["dflash"]' in (
    VLLM / "config/speculative.py"
).read_text()
assert (VLLM / "v1/spec_decode/dflash.py").is_file()
assert (VLLM / "model_executor/models/qwen3_dflash.py").is_file()
print("PRESERVED dflash speculative decoder and Qwen3 DFlash model")

model_overrides = {
    "model_executor/layers/quantization/modelopt.py": "00442ddb03e0ec291ed133160fe55f047d8dc64701d291780f49da66892084c5",
    "model_executor/models/qwen3_5.py": "b6a934cf2ab468e7164d9de5269e09c64ce2b97203b1675c70b75f25c17e2655",
    "model_executor/parameter.py": "6c1e64f299764b6c6ab5028c6a288e555b0dd5315623688417b20e7aedf45d11",
    "model_executor/layers/vocab_parallel_embedding.py": "31b9c50af2005007a17607c8ee84ce2a90a406f925b6110c176037eddb7384be",
}
for relative, expected_hash in model_overrides.items():
    actual_hash = hashlib.sha256((VLLM / relative).read_bytes()).hexdigest()
    assert actual_hash == expected_hash, (relative, actual_hash)
print("PRESERVED qwen3.6 NVFP4 model overrides=sha256-verified")
