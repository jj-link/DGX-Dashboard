#!/usr/bin/env bash
# Plain-decoding vLLM launch for pinned unsloth/Qwen3.6-35B-A3B-NVFP4-Fast.
set -euo pipefail

MODEL="${MODEL_PATH:?MODEL_PATH is required}"
SERVED="${SERVED:?SERVED is required}"
MAXLEN="${MAXLEN:-262144}"
PORT=8000
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
HOST=0.0.0.0
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAMBA_DTYPE="${MAMBA_DTYPE:-float32}"
VLLM_BIN="${VLLM_BIN:-vllm}"
SPEC="${SPEC:-off}"
if [ "$SPEC" != "off" ]; then
  printf 'Plain fallback selected; only SPEC=off is supported.\n' >&2
  exit 2
fi

unset PYTORCH_CUDA_ALLOC_CONF
unset VLLM_BUILD_URL VLLM_IMAGE_TAG VLLM_BUILD_PIPELINE VLLM_BUILD_COMMIT
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"

CACHE_ARG=()
[ "${PREFIX_CACHING:-1}" = "1" ] && CACHE_ARG+=(--enable-prefix-caching) || CACHE_ARG+=(--no-enable-prefix-caching)
KV_CACHE_ARG=()
[ -n "${KV_CACHE_DTYPE:-}" ] && KV_CACHE_ARG=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
PARSER_ARG=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
[ "${PARSERS:-1}" = "1" ] || PARSER_ARG=()

exec "$VLLM_BIN" serve "$MODEL" \
  "${CACHE_ARG[@]}" \
  --served-model-name "$SERVED" \
  --host "$HOST" --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  "${KV_CACHE_ARG[@]}" \
  --mamba-ssm-cache-dtype "$MAMBA_DTYPE" \
  --mamba-cache-dtype float16 \
  --enable-chunked-prefill \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --generation-config vllm \
  "${PARSER_ARG[@]}" \
  "$@"
