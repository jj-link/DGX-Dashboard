#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

TARGET="${1:-both}"
LINES="${2:-100}"
case "$TARGET" in
  head|rank0|spark2) ssh "$HEAD_HOST" "docker logs --tail '$LINES' '$CONTAINER' 2>&1" ;;
  worker|rank1|spark3) ssh "$WORKER_HOST" "docker logs --tail '$LINES' '$CONTAINER' 2>&1" ;;
  follow-head) ssh "$HEAD_HOST" "docker logs -f --tail '$LINES' '$CONTAINER'" ;;
  follow-worker) ssh "$WORKER_HOST" "docker logs -f --tail '$LINES' '$CONTAINER'" ;;
  both)
    echo "=== rank0/head: $HEAD_HOST ==="
    ssh "$HEAD_HOST" "docker logs --tail '$LINES' '$CONTAINER' 2>&1" || true
    echo "=== rank1/worker: $WORKER_HOST ==="
    ssh "$WORKER_HOST" "docker logs --tail '$LINES' '$CONTAINER' 2>&1" || true
    ;;
  *) echo "usage: $0 [both|head|worker|follow-head|follow-worker] [lines]" >&2; exit 2 ;;
esac
