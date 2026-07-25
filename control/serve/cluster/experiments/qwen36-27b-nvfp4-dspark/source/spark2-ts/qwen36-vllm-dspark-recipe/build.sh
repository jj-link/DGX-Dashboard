#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
for HOST in "$HEAD_HOST" "$WORKER_HOST"; do
  echo "=== building $IMAGE on $HOST ==="
  ssh "$HOST" "cd $RECIPE_DIR && docker build -t $IMAGE ."
done
