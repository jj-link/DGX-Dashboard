#!/usr/bin/env bash
set -euo pipefail
TAG=${TAG:-hy3-vllm-ray:v0.23.0}
WORKER_SSH_TARGET=${WORKER_SSH_TARGET:-jjlink@10.0.0.2}
SSH_KEY=${SSH_KEY:-/home/jjlink/.ssh/gb10_ed25519}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ssh_worker=(ssh -i "${SSH_KEY}" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "${WORKER_SSH_TARGET}")

echo "== build ${TAG} on Spark2 =="
docker build -t "${TAG}" "${SCRIPT_DIR}"

echo "== copy Dockerfile to Spark3 and build ${TAG} =="
"${ssh_worker[@]}" "mkdir -p /home/jjlink/inference/serve/cluster/hy3-nvfp4-mtp"
scp -i "${SSH_KEY}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new "${SCRIPT_DIR}/Dockerfile" "${WORKER_SSH_TARGET}:/home/jjlink/inference/serve/cluster/hy3-nvfp4-mtp/Dockerfile"
"${ssh_worker[@]}" "docker build -t '${TAG}' /home/jjlink/inference/serve/cluster/hy3-nvfp4-mtp"
