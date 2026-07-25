#!/usr/bin/env bash
# vLLM launch for AEON-7/Qwen3.6-27B-AEON-Ultimate-NVFP4 + DFlash speculative drafter
# on RTX PRO 6000 Blackwell (WSL2).
# Requires the ghcr.io/aeon-7/aeon-vllm-ultimate:latest image.
#
# Based on AEON-7 HF model card config for DGX Spark, adapted for local WSL.
set -u

# Default model/drafter — HF repo IDs, resolved from mounted HF cache
MODEL="${MODEL:-AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-NVFP4}"
DRAFTER="${DRAFTER:-z-lab/Qwen3.6-27B-DFlash}"
SERVED="${SERVED:-aeon-ultimate}"
MAXLEN="${MAXLEN:-256000}"
NUM_SPEC="${NUM_SPEC:-10}"  # from AEON-7 model card (10 is validated default)
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
PORT="${PORT:-8000}"

# WSL2+Blackwell: cuMemSetAccess (VMM API) unsupported.
unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# AEON-7 model card required env vars for GB10
# RTX PRO 6000 is SM120 (12.0+PTX), not GB10's 12.1a
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export ENABLE_NVFP4_SM100="0"
export VLLM_USE_FLASHINFER_MOE_FP4="0"
export VLLM_USE_FLASHINFER_SAMPLER="1"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
# WSL2+Blackwell: cuMemSetAccess (VMM API behind expandable_segments) is
# unsupported and crashes in the EngineCore subprocess. Do NOT set it.
export PYTORCH_CUDA_ALLOC_CONF=""

# DFlash speculative config
SPEC_CFG="{\"method\":\"dflash\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":${NUM_SPEC}}"

# Prefix caching ON by default
CACHE_ARG=()
[ "${PREFIX_CACHING:-1}" = "1" ] && CACHE_ARG+=(--enable-prefix-caching) || CACHE_ARG+=(--no-enable-prefix-caching)

# AEON-7 model card aliases
SERVED_NAMES="${SERVED_NAMES:-aeon-ultimate}"

# Parsers
PARSER_ARG=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()

exec vllm serve "$MODEL" \
  --speculative-config "$SPEC_CFG" \
  --served-model-name $SERVED_NAMES \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --quantization compressed-tensors \
  --mamba-cache-dtype float32 \
  --max-num-seqs 64 \
  --max-num-batched-tokens "${MAX_NUM_BATCHED:-16384}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.87}" \
  "${CACHE_ARG[@]}" \
  --enable-chunked-prefill \
  --trust-remote-code \
  --attention-backend flash_attn \
  --limit-mm-per-prompt '{"image":4,"video":2}' \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  "${PARSER_ARG[@]}"
