#!/usr/bin/env bash
# Sweep vLLM chunked-prefill batch size on the local RTX PRO 6000.
#
# Example:
#   ./sweep/prefill_chunk_sweep_local.sh serve/vllm/serve_qwen36_27b_fp8.sh Qwen3.6-27B
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
usage: sweep/prefill_chunk_sweep_local.sh [serve_script] [model]

Sweeps MAX_NUM_BATCHED over chunked-prefill values and writes a TSV result.

Examples:
  ./sweep/prefill_chunk_sweep_local.sh
  VALUES="4096 8192 16384" RUNS=2 ./sweep/prefill_chunk_sweep_local.sh serve/vllm/serve_qwen36_27b_fp8.sh Qwen3.6-27B
EOF
    exit 0
    ;;
esac

SERVE="${1:-serve/vllm/serve_qwen36_27b_fp8.sh}"
MODEL="${2:-Qwen3.6-27B}"
VALUES="${VALUES:-2048 4096 8192 16384 32768}"
PROMPT_TOKENS="${PROMPT_TOKENS:-50000}"
RUNS="${RUNS:-3}"
WARMUP="${WARMUP:-1}"
MAX_TOKENS="${MAX_TOKENS:-16}"
OUT="${OUT:-$ROOT/results/prefill-chunk-$(date +%Y%m%d-%H%M%S).tsv}"
CNAME="vllm-$(basename "$SERVE" .sh)"

mkdir -p "$(dirname "$OUT")"
printf "max_num_batched_tokens\tprompt_tokens\tavg_ttft_s\tprefill_tok_s\n" | tee "$OUT"

for value in $VALUES; do
  echo "[chunk=$value] restarting $SERVE" >&2
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  MAX_NUM_BATCHED="$value" DETACH=1 "$ROOT/run_vllm_docker.sh" "$SERVE" >/dev/null

  echo "[chunk=$value] waiting for server" >&2
  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 10
  done
  if [ "$ready" != 1 ]; then
    echo "[chunk=$value] SERVER DID NOT COME UP" >&2
    docker logs --tail 80 "$CNAME" >&2 || true
    exit 1
  fi

  echo "[chunk=$value] server args:" >&2
  docker logs "$CNAME" 2>&1 | grep -o "max_num_batched_tokens=[0-9]*" | tail -1 >&2 || true

  tmp="$(mktemp)"
  python "$ROOT/bench/bench_prefill.py" \
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
  printf "%s\t%s\t%s\t%s\n" "$value" "$prompt" "$ttft" "$tps" | tee -a "$OUT"
done

sort -t $'\t' -k4,4nr "$OUT" | sed -n '1,8p'
