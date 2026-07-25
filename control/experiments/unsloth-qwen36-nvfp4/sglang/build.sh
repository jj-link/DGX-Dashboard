#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="${IMAGE_TAG:-unsloth-qwen36-sglang:7de33ce8-mixed-ct-dflashq-fi0614}"
SOURCE_COMMIT="7de33ce806c12664b647604d61cf1403d2d18013"
PATCH_SHA256="838341d55916a1d0f2d4dd819c0fef1ddb4b2b8f03490b2d32b8c7375343f50d"
FLASHINFER_VERSION="0.6.14"
SGL_DEEP_GEMM_VERSION="0.1.4.post1"
PATCH="${ROOT}/0001-compressed-tensors-per-group-format.patch"

actual_patch_sha256="$(sha256sum "${PATCH}" | cut -d ' ' -f 1)"
if [[ "${actual_patch_sha256}" != "${PATCH_SHA256}" ]]; then
    echo "patch hash mismatch: expected ${PATCH_SHA256}, got ${actual_patch_sha256}" >&2
    exit 1
fi

cd "${ROOT}"
DOCKER_BUILDKIT=1 docker build --pull=false --tag "${IMAGE_TAG}" .

docker image inspect "${IMAGE_TAG}" --format \
    'image_tag={{index .RepoTags 0}} image_id={{.Id}} source_commit={{index .Config.Labels "org.opencontainers.image.revision"}} patch_sha256={{index .Config.Labels "io.unsloth.sglang.patch-sha256"}} flashinfer_version={{index .Config.Labels "io.unsloth.sglang.flashinfer-version"}} deep_gemm_version={{index .Config.Labels "io.unsloth.sglang.deep-gemm-version"}}'
echo "source_commit=${SOURCE_COMMIT}"
echo "patch_sha256=${PATCH_SHA256}"
echo "flashinfer_version=${FLASHINFER_VERSION}"
echo "deep_gemm_version=${SGL_DEEP_GEMM_VERSION}"
