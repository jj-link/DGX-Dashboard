#!/usr/bin/env bash
set -euo pipefail
export VLLM_USE_V2_MODEL_RUNNER=1

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
  --trust-remote-code
  --max-num-seqs
  4
  --attention-backend
  TRITON_ATTN
  --enable-auto-tool-choice
  --tool-call-parser
  gemma4
  --reasoning-parser
  gemma4
  --override-generation-config
  '{"max_new_tokens": null}'
  --default-chat-template-kwargs
  '{"enable_thinking":true}'
)
exec "${args[@]}" "$@"
