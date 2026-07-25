#!/usr/bin/env bash
set -euo pipefail

BUILDER_REPO=${BUILDER_REPO:-https://github.com/eugr/spark-vllm-docker.git}
BUILDER_REF=${BUILDER_REF:-4095335a9a31b9f2f42cb8a7836c9e55a6262661}
VLLM_REF=${VLLM_REF:-ab666069935c1f23e8ef56038b4659ac9e8f19f8}
IMAGE=${IMAGE:-hy3-vllm-upstream:ab666069}
BUILD_DIR=${BUILD_DIR:-/home/jjlink/inference/build/spark-vllm-docker-hy3}

if [ ! -d "${BUILD_DIR}/.git" ]; then
  git clone "${BUILDER_REPO}" "${BUILD_DIR}"
fi
git -C "${BUILD_DIR}" fetch origin
git -C "${BUILD_DIR}" checkout --detach "${BUILDER_REF}"
cd "${BUILD_DIR}"
exec ./build-and-copy.sh --vllm-ref "${VLLM_REF}" -t "${IMAGE}" --tf5
