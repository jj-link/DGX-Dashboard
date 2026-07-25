#!/usr/bin/env bash
set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-qwen-vllm:dflash}"
CANDIDATE_IMAGE="${CANDIDATE_IMAGE:-qwen-vllm:dflash-grouped-knorm-02a1f237}"
MAX_JOBS="${MAX_JOBS:-8}"
NVCC_THREADS="${NVCC_THREADS:-2}"

if [[ "${CANDIDATE_IMAGE}" == "${BASE_IMAGE}" ]]; then
    echo "refusing to overwrite production image tag ${BASE_IMAGE}" >&2
    exit 2
fi

docker image inspect "${BASE_IMAGE}" >/dev/null
docker build \
    --file Dockerfile \
    --tag "${CANDIDATE_IMAGE}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "MAX_JOBS=${MAX_JOBS}" \
    --build-arg "NVCC_THREADS=${NVCC_THREADS}" \
    .

printf 'built %s from %s\n' "${CANDIDATE_IMAGE}" "${BASE_IMAGE}"
