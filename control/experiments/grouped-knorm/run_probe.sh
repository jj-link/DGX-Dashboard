#!/usr/bin/env bash
set -euo pipefail

CANDIDATE_IMAGE="${CANDIDATE_IMAGE:-qwen-vllm:dflash-grouped-knorm-02a1f237}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen-vllm-dflash-grouped-knorm-probe-02a1f237}"

if [[ "${CANDIDATE_IMAGE}" == "qwen-vllm:dflash" ]]; then
    echo "refusing to probe the production image tag" >&2
    exit 2
fi

docker run --rm \
    --gpus all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e CUDA_VISIBLE_DEVICES=0 \
    --name "${CONTAINER_NAME}" \
    --entrypoint python3 \
    "${CANDIDATE_IMAGE}" \
    /opt/grouped-knorm/probe_grouped_knorm.py "$@"
