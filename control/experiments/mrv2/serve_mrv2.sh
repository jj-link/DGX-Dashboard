#!/usr/bin/env bash
set -euo pipefail

: "${REJECTION_SAMPLE_METHOD:?set REJECTION_SAMPLE_METHOD to standard or block}"
case "$REJECTION_SAMPLE_METHOD" in
  standard|block) ;;
  *) echo "fatal: unsupported REJECTION_SAMPLE_METHOD=$REJECTION_SAMPLE_METHOD" >&2; exit 64 ;;
esac

MODEL="${MODEL:-nvidia/Qwen3.6-27B-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-0893e1606ff3d5f97a441f405d5fc541a6bdf404}"
DRAFTER="${DRAFTER:-z-lab/Qwen3.6-27B-DFlash}"
DRAFTER_REVISION="${DRAFTER_REVISION:-0919688658996800f86b895034249700e9481106}"
SERVED="${SERVED:-nvidia-qwen36-27b-nvfp4-mrv2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-262144}"
NUM_SPEC="${NUM_SPEC:-12}"
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"

export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND:-flashinfer-cutlass}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER="${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-0}"
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

speculative_config=$(printf '{"method":"dflash","model":"%s","revision":"%s","num_speculative_tokens":%s,"attention_backend":"FLASH_ATTN","draft_sample_method":"probabilistic","rejection_sample_method":"%s"}' \
  "$DRAFTER" "$DRAFTER_REVISION" "$NUM_SPEC" "$REJECTION_SAMPLE_METHOD")

exec vllm serve "$MODEL" \
  --revision "$MODEL_REVISION" \
  --served-model-name "$SERVED" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --max-model-len "$MAXLEN" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --attention-config '{"backend":"FLASH_ATTN"}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --speculative-config "$speculative_config" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --generation-config vllm \
  --enable-log-requests
