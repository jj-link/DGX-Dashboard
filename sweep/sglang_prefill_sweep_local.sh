#!/usr/bin/env bash
# Sweep SGLang prefill chunk scheduler settings for local A3B DFlash.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVE="${SERVE:-serve/sglang/serve_qwen36_a3b_fp8_dflash.sh}"
MODEL="${MODEL:-Qwen3.6-35B-A3B}"
CHUNK_VALUES="${CHUNK_VALUES:-2048 4096 8192 16384}"
MAX_PREFILL_VALUES="${MAX_PREFILL_VALUES:-16384 32768}"
PROMPT_TOKENS="${PROMPT_TOKENS:-50000}"
RUNS="${RUNS:-2}"
WARMUP="${WARMUP:-1}"
MAX_TOKENS="${MAX_TOKENS:-16}"
OUT="${OUT:-$ROOT/results/sglang-prefill-a3b-dflash-$(date +%Y%m%d-%H%M%S).tsv}"
CNAME="sglang-$(basename "$SERVE" .sh)"

mkdir -p "$(dirname "$OUT")"
printf "chunked_prefill_size\tmax_prefill_tokens\tprompt_tokens\tavg_ttft_s\tprefill_tok_s\n" | tee "$OUT"

for chunk in $CHUNK_VALUES; do
  for max_prefill in $MAX_PREFILL_VALUES; do
    echo "[chunk=$chunk max_prefill=$max_prefill] restarting $SERVE" >&2
    docker rm -f "$CNAME" >/dev/null 2>&1 || true
    NUM_SPEC=12 DRAFT_WINDOW=2048 \
      CHUNKED_PREFILL_SIZE="$chunk" MAX_PREFILL_TOKENS="$max_prefill" \
      DETACH=1 KEEP=1 "$ROOT/run_sglang_docker.sh" "$SERVE" >/dev/null

    echo "[chunk=$chunk max_prefill=$max_prefill] waiting for server" >&2
    ready=0
    for _ in $(seq 1 180); do
      if curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; then
        ready=1
        break
      fi
      sleep 10
    done
    if [ "$ready" != 1 ]; then
      echo "[chunk=$chunk max_prefill=$max_prefill] SERVER DID NOT COME UP" >&2
      docker logs --tail 80 "$CNAME" >&2 || true
      printf "%s\t%s\tSERVER_FAILED\n" "$chunk" "$max_prefill" | tee -a "$OUT"
      docker rm -f "$CNAME" >/dev/null 2>&1 || true
      continue
    fi

    echo "[chunk=$chunk max_prefill=$max_prefill] server args:" >&2
    docker logs "$CNAME" 2>&1 | grep -oE "chunked_prefill_size=[0-9]+|max_prefill_tokens=[0-9]+" | tail -2 >&2 || true

    tmp="$(mktemp)"
    python3 "$ROOT/bench/bench_prefill.py" \
      --base http://localhost:8000 \
      --model "$MODEL" \
      --prompt-tokens "$PROMPT_TOKENS" \
      --max-tokens "$MAX_TOKENS" \
      --runs "$RUNS" \
      --warmup "$WARMUP" | tee "$tmp"

    summary="$(grep '^==> prefill:' "$tmp" | tail -1)"
    rm -f "$tmp"
    prompt="$(sed -nE 's/.*prompt_tokens=([0-9]+).*/\1/p' <<<"$summary")"
    ttft="$(sed -nE 's/.*ttft_avg=([0-9.]+)s.*/\1/p' <<<"$summary")"
    tps="$(sed -nE 's/.*tok\/s=([0-9.]+).*/\1/p' <<<"$summary")"
    printf "%s\t%s\t%s\t%s\t%s\n" "$chunk" "$max_prefill" "$prompt" "$ttft" "$tps" | tee -a "$OUT"

    docker rm -f "$CNAME" >/dev/null 2>&1 || true
  done
done

sort -t $'\t' -k5,5nr "$OUT" | sed -n '1,10p'
