#!/usr/bin/env bash
# Qualified production profile for nvidia/Qwen3.6-27B-NVFP4.
# Native MTP3 + BF16 attention KV: 18/18 OMP-v6 quality checks,
# 127.79 decode tok/s, 115.91 end-to-end tok/s, 30.39 s corpus wall time.
set -u

MODEL="${MODEL:-nvidia/Qwen3.6-27B-NVFP4}"
DRAFTER="${DRAFTER:-z-lab/Qwen3.6-27B-DFlash}"
SPEC="${SPEC:-mtp}"
SERVED="${SERVED:-nvidia_qwen36_27b_nvfp4_mtp3_bf16kv}"
SERVED_ALIASES="${SERVED_ALIASES:-}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAXLEN="${MAXLEN:-262144}"
NUM_SPEC="${NUM_SPEC:-3}"
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-8688}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.923}"
MAMBA_DTYPE="${MAMBA_DTYPE:-float32}"
LINEAR_BACKEND="${LINEAR_BACKEND:-cutlass}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
VLLM_BIN="${VLLM_BIN:-vllm}"

unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
# WSL2's pinned-memory probe reports false even though CUDA UVA is available.
export VLLM_FORCE_UVA="${VLLM_FORCE_UVA:-1}"

PARSER_ARG=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()

KV_CACHE_ARG=()
[ -n "$KV_CACHE_DTYPE" ] && KV_CACHE_ARG=(--kv-cache-dtype "$KV_CACHE_DTYPE")
SERVED_NAMES=("$SERVED")
if [ -n "$SERVED_ALIASES" ]; then
  IFS=',' read -r -a EXTRA_SERVED_NAMES <<< "$SERVED_ALIASES"
  SERVED_NAMES+=("${EXTRA_SERVED_NAMES[@]}")
fi

SPEC_ARG=()
case "$SPEC" in
  dflash)
    SPEC_ARG=(--speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":${NUM_SPEC}}")
    ;;
  mtp)
    SPEC_ARG=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC}}")
    ;;
  off)
    ;;
  *)
    echo "unsupported SPEC '$SPEC' (expected dflash, mtp, or off)" >&2
    exit 2
    ;;
esac

exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name "${SERVED_NAMES[@]}" \
  --host "$HOST" --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  "${KV_CACHE_ARG[@]}" \
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
  "${PARSER_ARG[@]}"
