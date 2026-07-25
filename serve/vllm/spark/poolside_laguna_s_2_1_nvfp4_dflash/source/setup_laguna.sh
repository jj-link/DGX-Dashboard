#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export UV_LINK_MODE=copy
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

VENV="$HOME/venvs/vllm025"
mkdir -p "$HF_HOME"

printf '\n[setup] managed Python 3.12\n'
uv python install 3.12
uv venv "$VENV" -p 3.12 --managed-python --seed --allow-existing

printf '\n[setup] vLLM 0.25.1 with CUDA 13 wheels\n'
uv pip install -p "$VENV" 'vllm==0.25.1' --torch-backend=cu130

printf '\n[setup] FlashInfer CUDA 13 nightly trio\n'
uv pip install -p "$VENV" \
  'flashinfer-python==0.6.15.dev20260712' \
  'flashinfer-cubin==0.6.15.dev20260712' \
  'flashinfer-jit-cache==0.6.15.dev20260712' \
  --extra-index-url https://flashinfer.ai/whl/nightly/ \
  --extra-index-url https://flashinfer.ai/whl/nightly/cu130/ \
  --index-strategy unsafe-best-match

printf '\n[verify] runtime\n'
"$VENV/bin/python" - <<'PY'
import pathlib
import sysconfig

import flashinfer
import torch
import vllm

include = pathlib.Path(sysconfig.get_paths()["include"])
print("python_include", include, "Python.h", (include / "Python.h").is_file())
print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("flashinfer", flashinfer.__version__)
PY

printf '\n[download] poolside/Laguna-S-2.1-NVFP4\n'
hf download poolside/Laguna-S-2.1-NVFP4

printf '\n[download] poolside/Laguna-S-2.1-DFlash-NVFP4\n'
hf download poolside/Laguna-S-2.1-DFlash-NVFP4

printf '\n[setup] complete\n'
df -h /
