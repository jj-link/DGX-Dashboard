#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-nvidia/Qwen3.6-27B-NVFP4}"
DRAFTER="${DRAFTER:-z-lab/Qwen3.6-27B-DFlash}"
DEPTH="${DEPTH:-0}"
SERVED="${SERVED:-qwen27_hybrid_depth${DEPTH}}"
PORT="${PORT:-8000}"

export VLLM_USE_V2_MODEL_RUNNER=1

SPEC_ARGS=()
if [[ "$DEPTH" != "0" ]]; then
  SPEC_ARGS=(
    --speculative-config
    "{\"method\":\"dflash\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":${DEPTH}}"
  )
fi

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.923 \
  --kv-cache-dtype fp8_e4m3 \
  --max-num-batched-tokens 8352 \
  --max-num-seqs 16 \
  --mamba-ssm-cache-dtype float32 \
  --mamba-cache-dtype float16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --linear-backend cutlass \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "${SPEC_ARGS[@]}"
