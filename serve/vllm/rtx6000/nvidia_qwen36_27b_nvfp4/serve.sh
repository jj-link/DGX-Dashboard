#!/usr/bin/env bash
# vLLM launch for nvidia/Qwen3.6-27B-NVFP4 (ModelOpt MIXED_PRECISION:
# FP8 attention, W4A16 NVFP4 MLPs) on RTX PRO 6000 Blackwell (WSL2).
set -euo pipefail
MODEL="${MODEL_PATH:?MODEL_PATH is required}"
TOKENIZER="${TOKENIZER_PATH:-}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-262144}"
PORT=8000

unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MAMBA_DTYPE="${MAMBA_DTYPE:-float32}"
LINEAR_BACKEND="${LINEAR_BACKEND:-cutlass}"

TOKENIZER_ARG=()
[ -n "$TOKENIZER" ] && TOKENIZER_ARG=(--tokenizer "$TOKENIZER")

KV_CACHE_ARG=()
[ -n "${KV_CACHE_DTYPE:-}" ] && KV_CACHE_ARG=(--kv-cache-dtype "${KV_CACHE_DTYPE}")

MAX_BATCHED_ARG=()
[ -n "${MAX_NUM_BATCHED:-}" ] && MAX_BATCHED_ARG=(--max-num-batched-tokens "$MAX_NUM_BATCHED")

exec vllm serve "$MODEL" \
  "${TOKENIZER_ARG[@]}" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.92 \
  "${KV_CACHE_ARG[@]}" \
  "${MAX_BATCHED_ARG[@]}" \
  --mamba-ssm-cache-dtype "$MAMBA_DTYPE" \
  --mamba-cache-dtype float16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --linear-backend "$LINEAR_BACKEND" \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "$@"
