#!/usr/bin/env bash
# Quality-qualified production profile for morosystems/ThinkingCap-Qwen3.6-27B-NVFP4.
# Equal OMP-v6 final qualification: plain 18/18, MTP6 18/18, DFlash9 18/18.
# DFlash9 wins at 65.429 aggregate decode tok/s versus MTP6's 62.656 and plain's 25.630.
# SPEC=off disables speculation; EAGER=1 restores the qualified eager DFlash15 profile.
set -u

MODEL="${MODEL:-/models/hub/models--morosystems--ThinkingCap-Qwen3.6-27B-NVFP4/snapshots/656627c8f7ea4785413ab1e06f6ccd20bba6622f}"
DRAFTER="${DRAFTER:-/models/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/0919688658996800f86b895034249700e9481106}"
SPEC="${SPEC:-dflash}"
SERVED_ALIASES="${SERVED_ALIASES:-}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAXLEN="${MAXLEN:-262144}"
MTP_NUM_SPEC="${MTP_NUM_SPEC:-6}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.923}"
MAMBA_DTYPE="${MAMBA_DTYPE:-float32}"
LINEAR_BACKEND="${LINEAR_BACKEND:-cutlass}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
VLLM_BIN="${VLLM_BIN:-vllm}"
EAGER="${EAGER:-0}"
COMPILATION_CONFIG="${COMPILATION_CONFIG-}"
BLOCK_SIZE="${BLOCK_SIZE-}"
if [ "$EAGER" = "1" ]; then
  EXECUTION_PROFILE=eager
  DFLASH_NUM_SPEC="${DFLASH_NUM_SPEC:-15}"
else
  EXECUTION_PROFILE=compiled
  DFLASH_NUM_SPEC="${DFLASH_NUM_SPEC:-9}"
  [ -n "$COMPILATION_CONFIG" ] || COMPILATION_CONFIG='{"custom_ops":["all"],"cudagraph_mode":"FULL"}'
  [ -n "$BLOCK_SIZE" ] || BLOCK_SIZE=848
fi

case "$SPEC" in
  dflash)
    NUM_SPEC="${NUM_SPEC:-$DFLASH_NUM_SPEC}"
    SERVED="${SERVED:-morosystems_thinkingcap_qwen36_27b_nvfp4_dflash${NUM_SPEC}_${EXECUTION_PROFILE}_bf16kv}"
    if [ "$EAGER" = "1" ]; then
      MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-8192}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
    else
      MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-8688}"
      MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    fi
    ;;
  mtp)
    NUM_SPEC="${NUM_SPEC:-$MTP_NUM_SPEC}"
    SERVED="${SERVED:-morosystems_thinkingcap_qwen36_27b_nvfp4_mtp${NUM_SPEC}_${EXECUTION_PROFILE}_bf16kv}"
    MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-8688}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    ;;
  off)
    NUM_SPEC="${NUM_SPEC:-0}"
    SERVED="${SERVED:-morosystems_thinkingcap_qwen36_27b_nvfp4_plain_${EXECUTION_PROFILE}_bf16kv}"
    MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-8688}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    ;;
esac

unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export VLLM_FORCE_UVA="${VLLM_FORCE_UVA:-1}"
# vLLM issue #44006: speculative tokens can violate xgrammar's structural-tag
# FSM while qwen3_xml still parses the resulting tool call correctly.
export VLLM_ENFORCE_STRICT_TOOL_CALLING="${VLLM_ENFORCE_STRICT_TOOL_CALLING:-0}"
# The invariant whole-model boundary is the qualified compiled path. EAGER=1
# retains the previous eager DFlash15 settings as an independent recovery path.
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"

PARSER_ARG=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()

KV_CACHE_ARG=()
[ -n "$KV_CACHE_DTYPE" ] && KV_CACHE_ARG=(--kv-cache-dtype "$KV_CACHE_DTYPE")
SERVED_NAMES=("$SERVED")
if [ -n "$SERVED_ALIASES" ]; then
  IFS=',' read -r -a EXTRA_SERVED_NAMES <<< "$SERVED_ALIASES"
  SERVED_NAMES+=("${EXTRA_SERVED_NAMES[@]}")
fi

DRAFTER_BATCH_ARG=
[ "${DISABLE_PADDED_DRAFTER_BATCH:-0}" = "1" ] && DRAFTER_BATCH_ARG=',"disable_padded_drafter_batch":true'

SPEC_ARG=()
case "$SPEC" in
  dflash)
    SPEC_ARG=(--speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":${NUM_SPEC}${DRAFTER_BATCH_ARG}}")
    ;;
  mtp)
    SPEC_ARG=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC}${DRAFTER_BATCH_ARG}}")
    ;;
  off)
    ;;
  *)
    echo "unsupported SPEC '$SPEC' (expected dflash, mtp, or off)" >&2
    exit 2
    ;;
esac

EAGER_ARG=()
[ "${EAGER:-0}" = "1" ] && EAGER_ARG=(--enforce-eager)
COMPILE_ARG=()
[ -n "$COMPILATION_CONFIG" ] && COMPILE_ARG=(--compilation-config "$COMPILATION_CONFIG")
BLOCK_SIZE_ARG=()
[ -n "$BLOCK_SIZE" ] && BLOCK_SIZE_ARG=(--block-size "$BLOCK_SIZE")

exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name "${SERVED_NAMES[@]}" \
  --host "$HOST" --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  "${KV_CACHE_ARG[@]}" \
  "${BLOCK_SIZE_ARG[@]}" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --mamba-ssm-cache-dtype "$MAMBA_DTYPE" \
  --mamba-cache-dtype float16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --linear-backend "$LINEAR_BACKEND" \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --generation-config vllm \
  "${SPEC_ARG[@]}" \
  "${EAGER_ARG[@]}" \
  "${COMPILE_ARG[@]}" \
  "${PARSER_ARG[@]}"
