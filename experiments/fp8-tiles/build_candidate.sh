#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE_TAG="${1:-qwen-vllm:fp8-tiles-cutlass46-bootstrap}"
BASE_IMAGE="${BASE_IMAGE:-qwen-vllm:dflash}"

if [[ "$IMAGE_TAG" == "$BASE_IMAGE" || "$IMAGE_TAG" == "qwen-vllm:dflash" ]]; then
  echo "candidate tag must not overwrite the production image" >&2
  exit 2
fi

exec docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "MAX_JOBS=${MAX_JOBS:-2}" \
  --build-arg "NVCC_THREADS=${NVCC_THREADS:-1}" \
  --tag "$IMAGE_TAG" \
  "$ROOT"
