#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
for host in "$HEAD_HOST" "$WORKER_HOST"; do
  echo "=== $host ==="
  ssh "$host" "docker ps -a --filter name='sglang_qwen36_unsloth_nvfp4' --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' || true"
done
echo "=== API ==="
ssh "$HEAD_HOST" "curl -sS --max-time 5 http://127.0.0.1:$API_PORT/v1/models || true"
echo
