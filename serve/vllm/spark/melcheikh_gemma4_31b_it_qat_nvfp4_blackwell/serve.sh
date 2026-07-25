#!/usr/bin/env bash
set -euo pipefail

args=(
  vllm
  serve
  "${MODEL_PATH}"
  --served-model-name
  "${SERVED}"
  --quantization
  modelopt
  --dtype
  auto
  --max-model-len
  65536
  --gpu-memory-utilization
  0.90
  --max-num-batched-tokens
  32768
  --max-num-seqs
  2
  --attention-backend
  triton_attn
  --host
  0.0.0.0
  --port
  8000
  --trust-remote-code
)
exec "${args[@]}" "$@"
