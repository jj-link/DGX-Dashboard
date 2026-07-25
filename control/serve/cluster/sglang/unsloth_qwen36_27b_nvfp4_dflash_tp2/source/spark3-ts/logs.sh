#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
TARGET="${1:-head}"
LINES="${2:-120}"
case "$TARGET" in
  head|spark2) ssh "$HEAD_HOST" "docker logs '$CONTAINER' --tail '$LINES' 2>&1" ;;
  worker|spark3) ssh "$WORKER_HOST" "docker logs '$CONTAINER' --tail '$LINES' 2>&1" ;;
  both) echo "=== head/$HEAD_HOST ==="; ssh "$HEAD_HOST" "docker logs '$CONTAINER' --tail '$LINES' 2>&1"; echo "=== worker/$WORKER_HOST ==="; ssh "$WORKER_HOST" "docker logs '$CONTAINER' --tail '$LINES' 2>&1" ;;
  follow) ssh "$HEAD_HOST" "docker logs -f '$CONTAINER' 2>&1" ;;
  *) echo "usage: $0 [head|worker|both|follow] [lines]" >&2; exit 2 ;;
esac
