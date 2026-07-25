#!/usr/bin/env bash
# vLLM launch for rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm
# on RTX PRO 6000 Blackwell (WSL2).
# Uses built-in MTP speculative decoding (no separate drafter model).
set -u

MODEL="${MODEL:-rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm}"
SERVED="${SERVED:-prismascout}"
MAXLEN="${MAXLEN:-262144}"
NUM_SPEC="${NUM_SPEC:-3}"  # MTP k=3 from model card
PORT="${PORT:-8000}"

# WSL2+Blackwell: cuMemSetAccess (VMM API) unsupported.
unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# MTP speculative config (built-in, no drafter model)
SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC}}"

# Prefix caching ON by default
CACHE_ARG=()
[ "${PREFIX_CACHING:-1}" = "1" ] && CACHE_ARG+=(--enable-prefix-caching) || CACHE_ARG+=(--no-enable-prefix-caching)

# Parsers
PARSER_ARG=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()

exec vllm serve "$MODEL" \
  --speculative-config "$SPEC_CFG" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
  "${CACHE_ARG[@]}" \
  --enable-chunked-prefill \
  --trust-remote-code \
  "${PARSER_ARG[@]}"
