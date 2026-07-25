#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
for name in sglang_qwen36_unsloth_nvfp4_dflash_tp2 sglang_qwen36_unsloth_nvfp4_tp2; do
  ssh "$HEAD_HOST" "docker rm -f '$name' >/dev/null 2>&1 || true"
  ssh "$WORKER_HOST" "docker rm -f '$name' >/dev/null 2>&1 || true"
done
echo "stopped SGLang Qwen3.6 Unsloth NVFP4 containers on $HEAD_HOST and $WORKER_HOST"
