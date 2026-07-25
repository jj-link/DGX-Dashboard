#!/usr/bin/env bash
set -euo pipefail

args=(
  vllm
  serve
  "${MODEL_PATH}"
  --served-model-name
  "${SERVED}"
  --host
  0.0.0.0
  --port
  8000
  --max-model-len
  262144
  --max-num-batched-tokens
  4096
  --gpu-memory-utilization
  0.92
  --mamba-ssm-cache-dtype
  float32
  --mamba-cache-dtype
  float16
  --enable-prefix-caching
  --enable-chunked-prefill
  --disable-custom-all-reduce
  --limit-mm-per-prompt
  '{"image":0,"video":0}'
  --generation-config
  vllm
  --reasoning-parser
  qwen3
  --enable-auto-tool-choice
  --tool-call-parser
  qwen3_coder
  --moe-backend
  flashinfer_b12x
  --no-enable-flashinfer-autotune
  --speculative-config
  '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"auto"}'
)
exec "${args[@]}" "$@"
