#!/usr/bin/env bash
set -euo pipefail

export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export GLOO_SOCKET_IFNAME=enp1s0f1np1
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_HCA=rocep1s0f1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_MERGE_NICS=0
export NCCL_IB_SUBNET_AWARE_ROUTING=1
export NCCL_CROSS_NIC=1
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_NET_GDR_LEVEL=LOC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_memory_monitor_refresh_ms=0
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0
export VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_FLASHINFER_MOE_BACKEND=throughput
export VLLM_NVFP4_INLINE=1
export VLLM_WMMA_DECODE=1
export VLLM_WMMA_INSPECT=0
export VLLM_WMMA_COMPARE=0
export VLLM_WMMA_AUTHCOMPARE=***
export VLLM_MIMO_MTP1_GREEDY_FAST=1
export VLLM_HOST_IP=10.0.0.1
export RAY_TMPDIR=/dev/shm/ray

MODEL_PATH=/root/.cache/huggingface/hub/models--lukealonso--MiMo-V2.5-NVFP4

vllm serve "${MODEL_PATH}" \
  --served-model-name MiMo-V2.5-NVFP4 \
  --trust-remote-code \
  --dtype auto \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend ray \
  --load-format safetensors \
  --hf-overrides '{"architectures":["MiMoV2OmniForCausalLM"]}' \
  --limit-mm-per-prompt '{"image":4,"video":1,"audio":1}' \
  --mm-encoder-tp-mode data \
  --attention-backend triton_attn_diffkv \
  --kv-cache-dtype nvfp4 \
  --gpu-memory-utilization 0.84 \
  --max-model-len 1000000 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --block-size 64 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --no-async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser mimo \
  --reasoning-parser mimo \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --generation-config vllm \
  --override-generation-config '{"temperature":0,"top_p":0.95,"repetition_penalty":1.08}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8888
