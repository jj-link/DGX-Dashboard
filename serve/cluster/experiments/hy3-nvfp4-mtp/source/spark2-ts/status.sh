#!/usr/bin/env bash
set -euo pipefail
WORKER_SSH_TARGET=${WORKER_SSH_TARGET:-jjlink@10.0.0.2}
SSH_KEY=${SSH_KEY:-/home/jjlink/.ssh/gb10_ed25519}
PORT=${PORT:-8888}
ssh_worker=(ssh -i "${SSH_KEY}" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "${WORKER_SSH_TARGET}")
echo "== API =="
curl -fsS "http://127.0.0.1:${PORT}/v1/models" || true
echo
echo "== Spark2 containers =="
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
echo "== Spark2 GPU =="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
echo "== Ray =="
docker exec hy3-head ray status 2>/dev/null || true
echo "== Spark3 containers =="
"${ssh_worker[@]}" "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'"
echo "== Spark3 GPU =="
"${ssh_worker[@]}" "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true"
