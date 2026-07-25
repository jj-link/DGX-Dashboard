#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
IMAGE_TAG=${IMAGE_TAG:-qwen-vllm:padded-quant}

exec docker build \
    --progress=plain \
    --tag "$IMAGE_TAG" \
    --build-arg RUNTIME_IMAGE=qwen-vllm:dflash \
    --build-arg VLLM_REVISION=23002d3f368a5a24641301bc71e4ae15dae89a24 \
    --build-arg TORCH_CUDA_ARCH_LIST=12.0 \
    --build-arg MAX_JOBS="${MAX_JOBS:-2}" \
    --build-arg NVCC_THREADS="${NVCC_THREADS:-8}" \
    "$SCRIPT_DIR"
