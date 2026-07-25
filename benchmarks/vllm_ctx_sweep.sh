#!/usr/bin/env bash
# vLLM DFlash long-context decode sweep (the cross-engine baseline for the SGLang
# comparison). Same client/corpus as sglang_ctx_sweep.sh, but scrapes vLLM's
# "SpecDecoding metrics: Mean acceptance length: X" log line. Requires the vLLM
# DFlash server running on :8000 (./serve.sh 27b_fp8_dflash).
#   ./benchmarks/vllm_ctx_sweep.sh "vLLM DFlash"
set -uo pipefail
cd "$(dirname "$0")/.."
CN=vllm-serve_qwen36_27b_fp8_dflash
LABEL="${1:-vLLM DFlash}"
OUT=benchmarks/results/DFLASH_SGLANG_SM120.md
CONTEXTS="${CONTEXTS:-8000 32000 64000 96000}"
{
  echo ""
  echo "### sweep: ${LABEL}  ($(date +%H:%M))"
  echo "| ctx_target | prompt_tok | decode tok/s | accept_len |"
  echo "|---|---|---|---|"
} >> "$OUT"
for ctx in $CONTEXTS; do
  raw=$(python3 benchmarks/dflash_longctx.py --model Qwen3.6-27B --ctx "$ctx" --gen 512 ${CORPUS:+--corpus "$CORPUS"} 2>&1 | tail -1)
  if echo "$raw" | grep -q tok_s; then
    pt=$(echo "$raw" | grep -oE "prompt_tokens=[0-9]+" | cut -d= -f2)
    tps=$(echo "$raw" | grep -oE "tok_s=[0-9.]+" | cut -d= -f2)
    al=$(docker logs "$CN" 2>&1 | grep -oE "Mean acceptance length: [0-9.]+" | tail -1 | grep -oE "[0-9.]+$")
    echo "| ${ctx} | ${pt:-?} | ${tps:-?} | ${al:-?} |" >> "$OUT"
    echo "[${LABEL}] ctx=${ctx} prompt_tok=${pt} tok_s=${tps} accept_len=${al}"
  else
    echo "| ${ctx} | FAIL | ${raw} | |" >> "$OUT"
    echo "[${LABEL}] ctx=${ctx} FAILED: ${raw}"
    docker ps --format '{{.Names}}' | grep -q "^${CN}$" || { echo "**vLLM SERVER DOWN @ctx=${ctx}**" >> "$OUT"; break; }
  fi
done
echo "VLLM_SWEEP_DONE"
