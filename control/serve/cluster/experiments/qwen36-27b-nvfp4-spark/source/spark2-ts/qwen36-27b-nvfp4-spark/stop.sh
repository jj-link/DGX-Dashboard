#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

echo "=== stopping $CONTAINER on Spark nodes ==="
ssh "$WORKER_HOST" "docker rm -f '$CONTAINER' >/dev/null 2>&1 || true"
ssh "$HEAD_HOST" "docker rm -f '$CONTAINER' >/dev/null 2>&1 || true"
echo "stopped"
