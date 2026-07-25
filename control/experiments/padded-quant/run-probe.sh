#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
IMAGE_TAG=${IMAGE_TAG:-qwen-vllm:padded-quant}
CONTAINER_NAME=${CONTAINER_NAME:-qwen-vllm-padded-quant-probe}

exec docker run --rm \
    --gpus all \
    --name "$CONTAINER_NAME" \
    --entrypoint python \
    --volume "$SCRIPT_DIR/probe_padded_nvfp4.py:/opt/padded-quant/probe_padded_nvfp4.py:ro" \
    "$IMAGE_TAG" \
    /opt/padded-quant/probe_padded_nvfp4.py
