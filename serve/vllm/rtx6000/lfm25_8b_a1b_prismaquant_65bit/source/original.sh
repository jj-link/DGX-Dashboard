#!/usr/bin/env bash
# vLLM launch for LFM2.5-8B-A1B PrismaQuant 6.5-bit.
#
# Normal entry point from this repo:
#   HF_CACHE_MODE=rw ./serve.sh lfm25
#
# That goes through ./run_vllm_docker.sh and runs this script inside Docker.
#
# Model card serving requirements:
#   vllm serve <this-dir> --quantization compressed-tensors --trust-remote-code
#
# The image must include vLLM support for Lfm2MoeForCausalLM plus the LFM2
# short-conv / linear-attention kernels: causal-conv1d and flash-linear-attention.
set -u

MODEL="${MODEL:-rdtand/LFM2.5-8B-A1B-PrismaQuant-6.5bit-vllm}"
SERVED="${SERVED:-LFM2.5-8B-A1B-PrismaQuant-6.5bit}"
MAXLEN="${MAXLEN:-128000}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

# WSL2+Blackwell: cuMemSetAccess (VMM API) is unsupported — expandable_segments crashes.
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

exec vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --quantization compressed-tensors \
  --trust-remote-code \
  --disable-custom-all-reduce \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser lfm2
