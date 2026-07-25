#!/usr/bin/env bash
# Tail logs from DSpark cluster nodes
# Usage: ./logs.sh [head|worker|both] [lines]
set -eu

NODE="${1:-both}"
LINES="${2:-30}"

case "$NODE" in
  head|rank0|spark2)
    echo "=== spark2-ts (rank 0) ==="
    ssh spark2-ts "docker logs dspark-recipe-vllm-dspark-1 --tail $LINES 2>&1"
    ;;
  worker|rank1|spark3)
    echo "=== spark3-ts (rank 1) ==="
    ssh spark3-ts "docker logs dspark-recipe-vllm-dspark-1 --tail $LINES 2>&1"
    ;;
  both)
    echo "=== spark2-ts (rank 0) ==="
    ssh spark2-ts "docker logs dspark-recipe-vllm-dspark-1 --tail $LINES 2>&1"
    echo ""
    echo "=== spark3-ts (rank 1) ==="
    ssh spark3-ts "docker logs dspark-recipe-vllm-dspark-1 --tail $LINES 2>&1"
    ;;
  *)
    echo "Usage: $0 [head|worker|both] [lines]"
    exit 1
    ;;
esac
