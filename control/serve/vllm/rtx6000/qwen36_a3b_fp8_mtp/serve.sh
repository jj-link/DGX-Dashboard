#!/usr/bin/env bash
# vLLM launch for Qwen3.6-35B-A3B FP8 with the model's OWN built-in MTP
# module as the speculative drafter (no extra model needed: the FP8
# checkpoint ships mtp.* weights, mtp_num_hidden_layers=1).
# Unlike DFlash, MTP does not require flash_attn. Leave KV cache dtype at
# vLLM/model default for benchmark quality; opt into FP8 with KV_CACHE_DTYPE=fp8.
set -euo pipefail
MODEL="${MODEL_PATH:?MODEL_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-262144}"
NUM_SPEC="${NUM_SPEC:-1}"          # 1 mtp layer -> 1 spec token is the native setting
PORT=8000

unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Self-speculative: the draft model IS the target; vLLM uses its mtp module.
SPEC_CFG="{\"method\": \"${MTP_METHOD:-mtp}\", \"num_speculative_tokens\": ${NUM_SPEC}}"

CACHE_ARG=()
[ "${PREFIX_CACHING:-1}" = "1" ] && CACHE_ARG+=(--enable-prefix-caching) || CACHE_ARG+=(--no-enable-prefix-caching)

KV_CACHE_ARG=()
[ -n "${KV_CACHE_DTYPE:-}" ] && KV_CACHE_ARG=(--kv-cache-dtype "${KV_CACHE_DTYPE}")

PARSER_ARG=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()

exec vllm serve "$MODEL" \
  --speculative-config "$SPEC_CFG" \
  "${CACHE_ARG[@]}" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.92 \
  "${KV_CACHE_ARG[@]}" \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --generation-config vllm \
  "${PARSER_ARG[@]}" \
  "$@"
