#!/usr/bin/env bash
set -euo pipefail
WORKER_SSH_TARGET=${WORKER_SSH_TARGET:-jjlink@10.0.0.2}
SSH_KEY=${SSH_KEY:-/home/jjlink/.ssh/gb10_ed25519}
CONTAINER_HEAD=${CONTAINER_HEAD:-hy3-head}
CONTAINER_WORKER=${CONTAINER_WORKER:-hy3-worker}
ssh_worker=(ssh -i "${SSH_KEY}" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "${WORKER_SSH_TARGET}")
echo "== stop Spark2 head =="
docker rm -f "${CONTAINER_HEAD}" 2>/dev/null || true
echo "== stop Spark3 worker =="
"${ssh_worker[@]}" "docker rm -f '${CONTAINER_WORKER}' 2>/dev/null || true"
