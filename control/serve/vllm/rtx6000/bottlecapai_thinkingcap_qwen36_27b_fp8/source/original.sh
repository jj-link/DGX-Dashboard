#!/usr/bin/env bash
# Native-FP8 ThinkingCap study and production profile for port 8000.
set -euo pipefail

MODEL="${MODEL:-/models/hub/models--bottlecapai--ThinkingCap-Qwen3.6-27B-FP8/snapshots/f25072f4e15775ca0d017afd754c7b8da85a9ff5}"
DRAFTER="${DRAFTER:-/models/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106}"
SPEC="${SPEC:-dflash}"
EAGER="${EAGER:-0}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAXLEN="${MAXLEN:-262144}"
MTP_NUM_SPEC="${MTP_NUM_SPEC:-6}"
DFLASH_NUM_SPEC="${DFLASH_NUM_SPEC:-9}"
DFLASH_LONG_CONTEXT_THRESHOLD="${DFLASH_LONG_CONTEXT_THRESHOLD:-16384}"
DFLASH_LONG_CONTEXT_NUM_SPEC="${DFLASH_LONG_CONTEXT_NUM_SPEC:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.923}"
MAMBA_DTYPE="${MAMBA_DTYPE:-float32}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
LINEAR_BACKEND="${LINEAR_BACKEND-cutlass}"
ATTENTION_BACKEND="${ATTENTION_BACKEND-}"
BLOCK_SIZE="${BLOCK_SIZE-848}"
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-8688}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
COMPILATION_CONFIG="${COMPILATION_CONFIG-}"
VLLM_BIN="${VLLM_BIN:-vllm}"
DISABLE_HYBRID_KV_CACHE_MANAGER="${DISABLE_HYBRID_KV_CACHE_MANAGER:-0}"

if [ "$EAGER" = "1" ] && [ "$SPEC" != "off" ]; then
  echo "EAGER=1 is restricted to SPEC=off plain oracle/recovery runs" >&2
  exit 2
fi

if [ "$EAGER" = "1" ]; then
  EXECUTION_PROFILE="eager_oracle"
else
  EXECUTION_PROFILE="compiled"
  [ -n "$COMPILATION_CONFIG" ] || COMPILATION_CONFIG='{"custom_ops":["all"],"cudagraph_mode":"FULL"}'
fi

case "$SPEC" in
  dflash)
    NUM_SPEC="${NUM_SPEC:-$DFLASH_NUM_SPEC}"
    export VLLM_SPECULATIVE_LONG_CONTEXT_THRESHOLD="${VLLM_SPECULATIVE_LONG_CONTEXT_THRESHOLD:-$DFLASH_LONG_CONTEXT_THRESHOLD}"
    export VLLM_SPECULATIVE_LONG_CONTEXT_TOKENS="${VLLM_SPECULATIVE_LONG_CONTEXT_TOKENS:-$DFLASH_LONG_CONTEXT_NUM_SPEC}"
    SERVED="${SERVED:-bottlecapai_thinkingcap_qwen36_27b_fp8_dflash${NUM_SPEC}_adaptive${VLLM_SPECULATIVE_LONG_CONTEXT_TOKENS}at${VLLM_SPECULATIVE_LONG_CONTEXT_THRESHOLD}_${EXECUTION_PROFILE}_bf16kv}"
    SPEC_ARG=(--speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":${NUM_SPEC}}")
    ;;
  mtp)
    NUM_SPEC="${NUM_SPEC:-$MTP_NUM_SPEC}"
    SERVED="${SERVED:-bottlecapai_thinkingcap_qwen36_27b_fp8_mtp${NUM_SPEC}_${EXECUTION_PROFILE}_bf16kv}"
    SPEC_ARG=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC}}")
    ;;
  off)
    NUM_SPEC=0
    SERVED="${SERVED:-bottlecapai_thinkingcap_qwen36_27b_fp8_plain_${EXECUTION_PROFILE}_bf16kv}"
    SPEC_ARG=()
    ;;
  *)
    echo "unsupported SPEC '$SPEC' (expected dflash, mtp, or off)" >&2
    exit 2
    ;;
esac

unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export VLLM_FORCE_UVA="${VLLM_FORCE_UVA:-1}"
export VLLM_ENFORCE_STRICT_TOOL_CALLING="${VLLM_ENFORCE_STRICT_TOOL_CALLING:-0}"
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"

EAGER_ARG=()
[ "$EAGER" = "1" ] && EAGER_ARG=(--enforce-eager)
COMPILE_ARG=()
[ -n "$COMPILATION_CONFIG" ] && COMPILE_ARG=(--compilation-config "$COMPILATION_CONFIG")
BLOCK_SIZE_ARG=()
[ -n "$BLOCK_SIZE" ] && BLOCK_SIZE_ARG=(--block-size "$BLOCK_SIZE")
LINEAR_ARG=()
[ -n "$LINEAR_BACKEND" ] && LINEAR_ARG=(--linear-backend "$LINEAR_BACKEND")
ATTENTION_ARG=()
[ -n "$ATTENTION_BACKEND" ] && ATTENTION_ARG=(--attention-backend "$ATTENTION_BACKEND")
HYBRID_KV_ARG=()
[ "$DISABLE_HYBRID_KV_CACHE_MANAGER" = "1" ] && HYBRID_KV_ARG=(--disable-hybrid-kv-cache-manager)

exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name "$SERVED" \
  --host "$HOST" --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  "${BLOCK_SIZE_ARG[@]}" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --mamba-ssm-cache-dtype "$MAMBA_DTYPE" \
  --mamba-cache-dtype float16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  "${HYBRID_KV_ARG[@]}" \
  "${LINEAR_ARG[@]}" \
  "${ATTENTION_ARG[@]}" \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --generation-config vllm \
  "${SPEC_ARG[@]}" \
  "${EAGER_ARG[@]}" \
  "${COMPILE_ARG[@]}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
