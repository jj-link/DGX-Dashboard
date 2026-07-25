#!/usr/bin/env bash
# vLLM launch for Qwen3.6-27B NVFP4 WITHOUT MTP (baseline path).
set -euo pipefail
MODEL="${MODEL_PATH:?MODEL_PATH is required}"
TOKENIZER="${TOKENIZER_PATH:-}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-262144}"
PORT=8000

# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True on WSL2+Blackwell â€”
# cuMemSetAccess (the VMM API behind expandable_segments) is unsupported and crashes
# in the EngineCore subprocess at first GPU alloc.
unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# SSM-state cache precision. Model config intends float32; A/B (2026-05-17 on
# 27B-FP8) showed fp32 = +1/34 polyglot, zero regressions, ~3% slower, no OOM
# at 128K. Default native fp32; MAMBA_DTYPE=float16 for max-throughput sweeps.
MAMBA_DTYPE="${MAMBA_DTYPE:-float32}"

TOKENIZER_ARG=()
[ -n "$TOKENIZER" ] && TOKENIZER_ARG=(--tokenizer "$TOKENIZER")

# Leave KV cache dtype at vLLM/model default for benchmark quality. Override
# with KV_CACHE_DTYPE=fp8 only for capacity/throughput sweeps.
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
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "$@"
