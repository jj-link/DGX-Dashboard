#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

for host in "$HEAD_HOST" "$WORKER_HOST"; do
  echo "=== $host container ==="
  ssh "$host" "docker ps -a --filter name='^/${CONTAINER}$' --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' || true"
  echo "=== $host gpu ==="
  ssh "$host" "nvidia-smi --query-gpu=name,index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true"
done

echo "=== API models ==="
ssh "$HEAD_HOST" "curl -fsS --max-time 5 'http://127.0.0.1:$API_PORT/v1/models' 2>/dev/null || echo 'API not ready'"
