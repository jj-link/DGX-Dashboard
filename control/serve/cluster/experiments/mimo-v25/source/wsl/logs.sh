#!/usr/bin/env bash
# Tail logs from MiMo-V2.5-NVFP4 cluster nodes
# Usage: ./logs.sh [head|worker|both] [lines]
set -eu

NODE="${1:-both}"
LINES="${2:-30}"

case "$NODE" in
  head|rank0|spark2)
    echo "=== spark2-ts (head, rank 0) ==="
    ssh spark2-ts "docker logs vllm_mimo_tp2 --tail $LINES 2>&1" 2>/dev/null || echo "  cannot reach spark2-ts"
    ;;
  worker|rank1|spark3)
    echo "=== spark3-ts (worker, rank 1) ==="
    ssh spark3-ts "docker logs vllm_mimo_tp2 --tail $LINES 2>&1" 2>/dev/null || echo "  cannot reach spark3-ts"
    ;;
  both)
    echo "=== spark2-ts (head, rank 0) ==="
    ssh spark2-ts "docker logs vllm_mimo_tp2 --tail $LINES 2>&1" 2>/dev/null || echo "  cannot reach spark2-ts"
    echo ""
    echo "=== spark3-ts (worker, rank 1) ==="
    ssh spark3-ts "docker logs vllm_mimo_tp2 --tail $LINES 2>&1" 2>/dev/null || echo "  cannot reach spark3-ts"
    ;;
  *)
    echo "Usage: $0 [head|worker|both] [lines]"
    exit 1
    ;;
esac
