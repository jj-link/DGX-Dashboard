#!/usr/bin/env bash
# Stop MiMo-V2.5-NVFP4 on the 2x GB10 Spark cluster
set -eu

echo "=== Stopping MiMo-V2.5-NVFP4 cluster ==="

for HOST in spark2-ts spark3-ts; do
  echo "--- $HOST ---"
  ssh "$HOST" "docker rm -f vllm_mimo_tp2 2>/dev/null && echo '  container removed' || echo '  no container'"
done

echo "=== Stopped ==="
