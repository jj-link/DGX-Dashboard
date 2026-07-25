from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"fatal: {message}")


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise SystemExit(f"fatal: required package is missing: {name}") from exc


expected_commit = os.environ.get("VLLM_EXPECTED_COMMIT")
expected_flashinfer = os.environ.get("FLASHINFER_EXPECTED_VERSION")
require(bool(expected_commit), "VLLM_EXPECTED_COMMIT is unset")
require(bool(expected_flashinfer), "FLASHINFER_EXPECTED_VERSION is unset")
assert expected_commit is not None
assert expected_flashinfer is not None

vllm_version = package_version("vllm")
require(expected_commit[:9] in vllm_version, f"vLLM {vllm_version} is not commit {expected_commit}")
for distribution in ("flashinfer-python", "flashinfer-cubin"):
    actual = package_version(distribution)
    require(actual == expected_flashinfer, f"{distribution} is {actual}, expected {expected_flashinfer}")
jit_cache = package_version("flashinfer-jit-cache")
require(jit_cache == f"{expected_flashinfer}+cu130", f"flashinfer-jit-cache is {jit_cache}, expected {expected_flashinfer}+cu130")

import torch

require(torch.__version__.split("+")[0] == "2.11.0", f"torch is {torch.__version__}, expected 2.11.0")
require(torch.version.cuda == "13.0", f"torch CUDA ABI is {torch.version.cuda}, expected 13.0")

speculative = Path("/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py").read_text()
require('DFlashModelTypes = Literal["dflash"]' in speculative, "pinned vLLM lacks DFlash configuration support")
require('RejectionSampleMethod = Literal["standard", "synthetic", "block"]' in speculative, "pinned vLLM lacks block rejection support")

site = Path("/usr/local/lib/python3.12/dist-packages")
registry = (site / "vllm/model_executor/models/registry.py").read_text()
require(
    '"Qwen3_5ForConditionalGeneration"' in registry,
    "pinned vLLM does not register the Qwen3.5 architecture used by Qwen3.6",
)
require(
    '"DFlashDraftModel": ("qwen3_dflash", "DFlashQwen3ForCausalLM")' in registry,
    "pinned vLLM does not register the target DFlash drafter architecture",
)

runner = (site / "vllm/v1/worker/gpu/model_runner.py").read_text()
require("ModelCudaGraphManager" in runner, "ModelRunnerV2 full CUDA graph manager is missing")
require('("eagle3", "dflash", "dspark")' in runner, "ModelRunnerV2 DFlash path is missing")

manifest = Path("/opt/modelopt-reapply/manifest.tsv")
require(manifest.is_file(), "ModelOpt reapply manifest is missing")
rows = [line.split("\t") for line in manifest.read_text().splitlines() if line]
require(len(rows) == 4, f"ModelOpt reapply manifest has {len(rows)} entries, expected 4")
valid_statuses = {"superseded-upstream", "diagnostic-only-not-forwarded"}
require(
    all(len(row) == 3 and row[1] in valid_statuses for row in rows),
    "ModelOpt reapply manifest contains an invalid status",
)

print(
    "verified:",
    f"vllm={vllm_version}",
    f"torch={torch.__version__}",
    f"cuda={torch.version.cuda}",
    f"flashinfer={expected_flashinfer}",
    "mrv2=dflash+full-cudagraph",
    "rejection=standard+block",
)
