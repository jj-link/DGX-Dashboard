#!/usr/bin/env bash
# Qwen3.8-27B FP8 + MTP on RTX PRO 6000 Blackwell (SM_120, WSL2).
set -euo pipefail
MODEL="${MODEL_PATH:?MODEL_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-262144}"
NSPEC="${NSPEC:-3}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
PORT=8000

# WSL2 does not support the CUDA VMM path used by expandable segments.
unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export SAFETENSORS_FAST_GPU=1

MAX_BATCHED_ARG=()
[ -n "${MAX_NUM_BATCHED:-}" ] && MAX_BATCHED_ARG=(--max-num-batched-tokens "$MAX_NUM_BATCHED")

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  "${MAX_BATCHED_ARG[@]}" \
  --mamba-ssm-cache-dtype float32 \
  --mamba-cache-dtype float16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --enable-flashinfer-autotune \
  --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${NSPEC}}" \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "$@"
