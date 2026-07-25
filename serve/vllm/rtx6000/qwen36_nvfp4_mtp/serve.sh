#!/usr/bin/env bash
# vLLM launch for Qwen3.6-27B NVFP4 + MTP on RTX PRO 6000 Blackwell (SM_120, WSL2).
# Goal: 120 tok/s single-stream decode at 50k context, lossless.
set -euo pipefail
MODEL="${MODEL_PATH:?MODEL_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-262144}"
NSPEC="${NSPEC:-4}"                  # MTP=4 + no-prefix-cache (prefix-cache crashes vLLM 0.20 here)
PORT=8000

# WSL2+Blackwell: cuMemSetAccess (VMM API) is unsupported â€” expandable_segments crashes.
unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export SAFETENSORS_FAST_GPU=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Disable CCCL/nvcc compatibility check (cu13 packages have version skew after
# repair-from-bad-install; the check is overly strict â€” kernels still build).
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

# SSM-state cache precision. Model config intends float32; A/B (2026-05-17 on
# 27B-FP8) showed fp32 = +1/34 polyglot, zero regressions, ~3% slower, no OOM
# at 128K. Default native fp32; MAMBA_DTYPE=float16 for max-throughput sweeps.
MAMBA_DTYPE="${MAMBA_DTYPE:-float32}"

# Leave KV cache dtype at vLLM/model default for benchmark quality. Override
# with KV_CACHE_DTYPE=fp8 only for capacity/throughput sweeps.
KV_CACHE_ARG=()
[ -n "${KV_CACHE_DTYPE:-}" ] && KV_CACHE_ARG=(--kv-cache-dtype "${KV_CACHE_DTYPE}")

MAX_BATCHED_ARG=()
[ -n "${MAX_NUM_BATCHED:-}" ] && MAX_BATCHED_ARG=(--max-num-batched-tokens "$MAX_NUM_BATCHED")

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.92 \
  "${KV_CACHE_ARG[@]}" \
  "${MAX_BATCHED_ARG[@]}" \
  --mamba-ssm-cache-dtype "$MAMBA_DTYPE" \
  --mamba-cache-dtype float16 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --enable-flashinfer-autotune \
  --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${NSPEC}}" \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "$@"
