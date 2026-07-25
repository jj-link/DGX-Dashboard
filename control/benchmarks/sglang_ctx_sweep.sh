#!/usr/bin/env bash
# Long-context decode sweep against the RUNNING SGLang DFlash server. Appends a
# labeled table (decode tok/s + draft accept len, scraped from the server's
# "Decode batch ... accept len" log) to the results md. One row per context.
#   ./benchmarks/sglang_ctx_sweep.sh "window=4096"
#   CONTEXTS="8000 32000 64000" ./benchmarks/sglang_ctx_sweep.sh "window=off"
set -uo pipefail
cd "$(dirname "$0")/.."
CN=sglang-serve_qwen36_27b_fp8_dflash
LABEL="${1:?usage: sglang_ctx_sweep.sh <label>}"
OUT=benchmarks/results/DFLASH_SGLANG_SM120.md
CONTEXTS="${CONTEXTS:-8000 16000 32000 48000 64000 96000}"
{
  echo ""
  echo "### sweep: ${LABEL}  ($(date +%H:%M))"
  echo "| ctx_target | prompt_tok | decode tok/s | accept_len | accept_rate |"
  echo "|---|---|---|---|---|"
} >> "$OUT"
for ctx in $CONTEXTS; do
  raw=$(python3 benchmarks/dflash_longctx.py --model Qwen3.6-27B --ctx "$ctx" --gen 512 ${CORPUS:+--corpus "$CORPUS"} 2>&1 | tail -1)
  if echo "$raw" | grep -q tok_s; then
    pt=$(echo "$raw" | grep -oE "prompt_tokens=[0-9]+" | cut -d= -f2)
    tps=$(echo "$raw" | grep -oE "tok_s=[0-9.]+" | cut -d= -f2)
    db=$(docker logs "$CN" 2>&1 | grep "Decode batch" | tail -1)
    al=$(echo "$db" | grep -oE "accept len: [0-9.]+" | grep -oE "[0-9.]+$")
    ar=$(echo "$db" | grep -oE "accept rate: [0-9.]+" | grep -oE "[0-9.]+$")
    echo "| ${ctx} | ${pt:-?} | ${tps:-?} | ${al:-?} | ${ar:-?} |" >> "$OUT"
    echo "[${LABEL}] ctx=${ctx} prompt_tok=${pt} tok_s=${tps} accept_len=${al}"
  else
    echo "| ${ctx} | FAIL | ${raw} | | |" >> "$OUT"
    echo "[${LABEL}] ctx=${ctx} FAILED: ${raw}"
    docker ps --format '{{.Names}}' | grep -q "^${CN}$" || { echo "**SERVER DOWN after ctx=${ctx}**" >> "$OUT"; echo "SERVER_DOWN at ctx=${ctx}"; break; }
  fi
done
echo "SWEEP_DONE: ${LABEL}"
