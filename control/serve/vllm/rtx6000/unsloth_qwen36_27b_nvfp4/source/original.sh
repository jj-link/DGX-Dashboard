#!/usr/bin/env bash
# vLLM launch for pinned unsloth/Qwen3.6-27B-NVFP4 with native MTP.
set -u

MODEL="${MODEL:-/models/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/9c295353cf85c4047fb45c27847e6c4b58596f3f}"
SERVED="${SERVED:-unsloth-qwen36-27b-nvfp4}"
MAXLEN="${MAXLEN:-262144}"
NUM_SPEC="${NUM_SPEC:-2}"
PORT="${PORT:-8000}"

unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SPEC_ARG=()
if [ "${SPEC:-off}" != "off" ]; then
  SPEC_CFG="{\"method\": \"${MTP_METHOD:-mtp}\", \"num_speculative_tokens\": ${NUM_SPEC}}"
  SPEC_ARG=(--speculative-config "$SPEC_CFG")
fi
CACHE_ARG=()
[ "${PREFIX_CACHING:-1}" = "1" ] && CACHE_ARG+=(--enable-prefix-caching) || CACHE_ARG+=(--no-enable-prefix-caching)
KV_CACHE_ARG=()
[ -n "${KV_CACHE_DTYPE:-}" ] && KV_CACHE_ARG=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
PARSER_ARG=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()

exec vllm serve "$MODEL" \
  "${SPEC_ARG[@]}" \
  "${CACHE_ARG[@]}" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.92 \
  "${KV_CACHE_ARG[@]}" \
  --mamba-ssm-cache-dtype "${MAMBA_DTYPE:-float32}" \
  --mamba-cache-dtype float16 \
  --enable-chunked-prefill \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --generation-config vllm \
  "${PARSER_ARG[@]}"
