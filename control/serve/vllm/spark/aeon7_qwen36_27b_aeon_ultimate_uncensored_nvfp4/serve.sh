#!/usr/bin/env bash
set -euo pipefail
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_FLASHINFER_MOE_FP4=0
export VLLM_USE_FLASHINFER_SAMPLER=1

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
  --quantization
  compressed-tensors
  --mamba-cache-dtype
  float32
  --max-model-len
  256000
  --max-num-seqs
  64
  --max-num-batched-tokens
  16384
  --gpu-memory-utilization
  0.85
  --enable-chunked-prefill
  --enable-prefix-caching
  --load-format
  safetensors
  --trust-remote-code
  --enable-auto-tool-choice
  --tool-call-parser
  qwen3_coder
  --reasoning-parser
  qwen3
  --attention-backend
  flash_attn
  --limit-mm-per-prompt
  '{"image":4,"video":2}'
  --mm-encoder-tp-mode
  data
  --mm-processor-cache-type
  shm
  --speculative-config
  "{\"method\":\"dflash\",\"model\":\"${DRAFTER_PATH}\",\"num_speculative_tokens\":10}"
)
exec "${args[@]}" "$@"
