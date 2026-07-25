#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"

VENV="${VENV:-$HOME/venvs/vllm025}"
MODEL="${MODEL:-$HF_HOME/hub/models--poolside--Laguna-S-2.1-NVFP4/snapshots/07614121b31898586430f189d27a25a0be310843}"
DRAFT_MODEL="${DRAFT_MODEL:-$HF_HOME/hub/models--poolside--Laguna-S-2.1-DFlash-NVFP4/snapshots/723794750422b3efbf3a7b3af76dffb4ba035943}"
SERVED_MODEL="${SERVED_MODEL:-poolside/Laguna-S-2.1-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-6}"

export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
export MAX_JOBS="${MAX_JOBS:-4}"
export PATH="/usr/local/cuda/bin:$VENV/bin:$HOME/.local/bin:$PATH"

SPECULATIVE_ARGS=()
if (( NUM_SPECULATIVE_TOKENS > 0 )); then
  SPECULATIVE_ARGS=(
    --speculative-config
    "{\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}"
  )
fi

exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  "${SPECULATIVE_ARGS[@]}" \
  --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --override-generation-config '{"temperature":0.7,"top_p":0.95}' \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --host "$HOST" \
  --port "$PORT"
