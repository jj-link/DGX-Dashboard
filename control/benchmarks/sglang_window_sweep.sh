#!/usr/bin/env bash
# Sweep --speculative-draft-window-size: for each value, relaunch the SGLang
# DFlash server and run the long-context decode sweep. Unattended (~30-40 min).
#   ./benchmarks/sglang_window_sweep.sh
#   WINDOWS="2048 8192" CONTEXTS="8000 64000" ./benchmarks/sglang_window_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."
CN=sglang-serve_qwen36_27b_fp8_dflash
OUT=benchmarks/results/DFLASH_SGLANG_SM120.md
WINDOWS="${WINDOWS:-2048 3072 6144 8192}"
export CONTEXTS="${CONTEXTS:-8000 32000 64000 96000}"
for win in $WINDOWS; do
  echo "===== relaunch window=${win} ====="
  docker rm -f "$CN" >/dev/null 2>&1 || true
  DRAFT_WINDOW="$win" KEEP=1 DETACH=1 ./serve.sh sglang 27b_fp8_dflash >/dev/null 2>&1
  ready=""
  for i in $(seq 1 240); do
    curl -s -m2 http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen && { ready=1; break; }
    docker ps --format '{{.Names}}' | grep -q "^${CN}$" || break
    sleep 5
  done
  if [ -z "$ready" ]; then
    echo "window=${win} FAILED TO START"
    { echo ""; echo "**window=${win} FAILED TO START:**"; docker logs "$CN" 2>&1 | tail -12; } >> "$OUT"
    continue
  fi
  ./benchmarks/sglang_ctx_sweep.sh "window=${win}"
done
echo "WINDOW_SWEEP_DONE"
