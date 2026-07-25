#!/usr/bin/env bash
# vLLM MTP-3 launch for unsloth/Qwen3.6-35B-A3B-NVFP4-Fast on DGX Spark GB10.
set -euo pipefail

MODEL="${MODEL:-/models/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/24ccf90e45a8f7e84e6251f4a19648104949c9f1}"
SERVED="${SERVED:-unsloth-qwen36-35b-a3b-nvfp4-fast}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-262144}"
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-float32}"
MAMBA_CACHE_DTYPE="${MAMBA_CACHE_DTYPE:-float16}"
MOE_BACKEND="${MOE_BACKEND:-flashinfer_b12x}"

if [[ -z "${SPECULATIVE_CONFIG:-}" ]]; then
  SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3,"moe_backend":"auto"}'
fi
if [[ -z "${MM_LIMITS:-}" ]]; then
  MM_LIMITS='{"image":0,"video":0}'
fi

exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-model-len "${MAXLEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --mamba-ssm-cache-dtype "${MAMBA_SSM_CACHE_DTYPE}" \
  --mamba-cache-dtype "${MAMBA_CACHE_DTYPE}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt "${MM_LIMITS}" \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --moe-backend "${MOE_BACKEND}" \
  --no-enable-flashinfer-autotune \
  --speculative-config "${SPECULATIVE_CONFIG}"
