#!/usr/bin/env bash
set -euo pipefail
export CUTE_DSL_ARCH=sm_121a

args=(
  vllm
  serve
  "${MODEL_PATH}"
  --served-model-name
  "${SERVED}"
  --speculative-config
  "{\"model\":\"${DRAFTER_PATH}\",\"num_speculative_tokens\":6}"
  --enable-auto-tool-choice
  --tool-call-parser
  poolside_v1
  --reasoning-parser
  poolside_v1
  --default-chat-template-kwargs
  '{"enable_thinking":true}'
  --override-generation-config
  '{"temperature":0.7,"top_p":0.95}'
  --max-num-seqs
  32
  --max-model-len
  262144
  --gpu-memory-utilization
  0.85
  --host
  0.0.0.0
  --port
  8000
)
exec "${args[@]}" "$@"
