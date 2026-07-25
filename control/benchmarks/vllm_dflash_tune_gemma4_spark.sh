#!/usr/bin/env bash
# Direct vLLM port of benchmarks/dflash_tune_local.sh.
# Do not change benchmark semantics here. Only backend-specific differences are:
#   - resolve serve/vllm scripts instead of serve/sglang
#   - launch vLLM with SPECULATIVE_CONFIG JSON instead of SGLang NUM_SPEC/DRAFT_WINDOW flags
#   - read vLLM /metrics instead of scraping SGLang throughput logs
#
# Usage:
#   ./benchmarks/vllm_spec_tune_local.sh <serve-name>
#   ./benchmarks/vllm_spec_tune_local.sh gemma4_31b_nvfp4

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

usage() {
  cat >&2 <<'EOF'
usage: ./benchmarks/vllm_spec_tune_local.sh <serve-name>

Examples:
  ./benchmarks/vllm_spec_tune_local.sh gemma4_31b_nvfp4

Optional env overrides:
  TOKENS_LIST="8 12 15 16 20"
  PROMPT_TOKENS=8192
  MAX_TOKENS=32768
  PORT=8000
  MODEL=<served-model-name>
  CONTAINER=<docker-container-name>
  DFLASH_MODEL=<dflash drafter repo/path>
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit $([[ $# -lt 1 ]] && echo 1 || echo 0)
fi

SERVE_NAME="$1"
shift || true
MODE="dflash"

resolve_serve_script() {
  local name="$1"
  local dir="serve/vllm"
  local matches exact f

  shopt -s nullglob
  matches=("$ROOT/$dir/"*"$name"*.sh)

  if (( ${#matches[@]} == 0 )); then
    echo "no vLLM serve script matches '$name'" >&2
    echo "available:" >&2
    for f in "$ROOT/$dir/"serve_*.sh; do
      [ -e "$f" ] || continue
      echo "  $(basename "$f" .sh | sed 's/^serve_//')" >&2
    done
    exit 1
  fi

  if (( ${#matches[@]} > 1 )); then
    exact=()
    for f in "${matches[@]}"; do
      [[ "$(basename "$f")" == *"${name}.sh" ]] && exact+=("$f")
    done
    if (( ${#exact[@]} == 1 )); then
      matches=("${exact[@]}")
    else
      echo "'$name' is ambiguous:" >&2
      for f in "${matches[@]}"; do
        echo "  $(basename "$f" .sh | sed 's/^serve_//')" >&2
      done
      exit 1
    fi
  fi

  printf '%s\n' "${matches[0]}"
}

serve_default_var() {
  local file="$1" var="$2"
  sed -nE "s/^[[:space:]]*${var}=\"\\\$\\{${var}:-([^}]*)\\}\".*/\\1/p" "$file" | head -n1
}

SERVE_SCRIPT_PATH="$(resolve_serve_script "$SERVE_NAME")"
SERVE_SCRIPT_BASENAME="$(basename "$SERVE_SCRIPT_PATH" .sh)"
CONTAINER="${CONTAINER:-vllm-${SERVE_SCRIPT_BASENAME}}"
PORT="${PORT:-$(serve_default_var "$SERVE_SCRIPT_PATH" PORT)}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-$(serve_default_var "$SERVE_SCRIPT_PATH" SERVED)}"
MODEL="${MODEL:-$(serve_default_var "$SERVE_SCRIPT_PATH" MODEL)}"
MODEL="${MODEL:-$SERVE_NAME}"
SAFE_NAME="$(basename "$SERVE_SCRIPT_BASENAME" | sed 's/^serve_//' | tr -c 'A-Za-z0-9_.-' '_')"

USER_TOKENS_LIST="${TOKENS_LIST:-}"
TOKENS_LIST="${TOKENS_LIST:-8 12 15 16 20}"
PROMPT_TOKENS="${PROMPT_TOKENS:-8192}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT="${OUT:-benchmarks/results/vllm-dflash-tune-${SAFE_NAME}-${RUN_ID}.tsv}"
LOG_DIR="${LOG_DIR:-benchmarks/results/vllm-dflash-tune-${SAFE_NAME}-${RUN_ID}.logs}"
TASKS_DIR="${TASKS_DIR:-/home/workbench/Projects/personal/agent-benchmark-runner/tasks}"
DFLASH_MODEL="${DFLASH_MODEL:-z-lab/gemma-4-31B-it-DFlash}"

make_prompt() {
  python3 - "$PROMPT_TOKENS" "$TASKS_DIR" <<'PY'
import os, sys

n = int(sys.argv[1])
tasks_dir = sys.argv[2]
descriptions = []

for root, _, files in os.walk(tasks_dir):
    for name in files:
        if name != "instruction.md":
            continue
        path = os.path.join(root, name)
        try:
            value = open(path, errors="replace").read().strip()
        except Exception:
            continue
        if len(value) > 100:
            descriptions.append(value)
    if len(descriptions) >= 12:
        break

if not descriptions:
    raise SystemExit(f"no task descriptions found under {tasks_dir}")

body = "\n\n--- TASK ---\n\n".join(descriptions)
while len(body.split()) < n:
    body = body + "\n\n--- TASK ---\n\n" + body

print(
    "You are an autonomous coding agent. Solve the following benchmark-style tasks. "
    "For each task, reason through the likely repository changes, implementation approach, "
    "tests to add or update, edge cases, and validation steps. Be concrete and exhaustive, "
    "as if preparing to implement the fixes.\n\n"
    + body
)
PY
}

run_one() {
  local prompt_file
  prompt_file="$(mktemp)"
  cat >"$prompt_file"
  python3 - "$PORT" "$MAX_TOKENS" "$MODEL" "$prompt_file" <<'PY'
import json, sys, time, urllib.request, urllib.error

port = int(sys.argv[1])
max_tokens = int(sys.argv[2])
model = sys.argv[3]
with open(sys.argv[4], "r", errors="replace") as fh:
    prompt = fh.read()
url = f"http://127.0.0.1:{port}/v1/chat/completions"
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "max_tokens": max_tokens,
}
data = json.dumps(payload).encode()
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=3600) as resp:
        body = resp.read()
except urllib.error.HTTPError as exc:
    sys.stderr.write(f"request failed: HTTP {exc.code}\n")
    sys.stderr.write(exc.read().decode("utf-8", errors="replace")[:4000] + "\n")
    raise SystemExit(1)
except Exception as exc:
    sys.stderr.write(f"request failed: {type(exc).__name__}: {exc}\n")
    raise SystemExit(1)

elapsed = time.time() - t0
obj = json.loads(body)
usage = obj.get("usage", {})
prompt_tokens = usage.get("prompt_tokens", "")
completion_tokens = usage.get("completion_tokens", "")
print(f"{elapsed:.3f}\t{prompt_tokens}\t{completion_tokens}")
PY
  local rc=$?
  rm -f "$prompt_file"
  return "$rc"
}

metric_value() {
  local name="$1"
  curl -fsS --max-time 5 "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
    | awk -v n="vllm:$name" '$1 ~ "^" n "(\\{|$)" {print $2; exit}'
}

scrape_logs() {
  # Backend-specific replacement for SGLang log scraping.
  # Emits the same three columns as dflash_tune_local.sh: accept_len, accept_rate, gen_tok_s.
  local gen0="$1" gen1="$2" dt0="$3" dt1="$4" acc0="$5" acc1="$6" draft0="$7" draft1="$8" req0="$9" req1="${10}"
  python3 - "$gen0" "$gen1" "$dt0" "$dt1" "$acc0" "$acc1" "$draft0" "$draft1" "$req0" "$req1" <<'PY'
import sys
G0,G1,DT0,DT1,A0,A1,D0,D1,R0,R1 = map(float, sys.argv[1:])
G = G1 - G0
DT = DT1 - DT0
A = A1 - A0
D = D1 - D0
R = R1 - R0
# vLLM exposes cumulative accepted/draft tokens. It does not expose SGLang's per-step
# accept_len directly, so use accepted tokens per completed request as the closest
# comparable scalar and accepted/draft as accept_rate.
accept_len = A / R if R > 0 else 0.0
accept_rate = A / D if D > 0 else 0.0
gen_tok_s = G / DT if DT > 0 else 0.0
print(f"{accept_len:.3f}\t{accept_rate:.3f}\t{gen_tok_s:.3f}")
PY
}

wait_ready() {
  local deadline=$((SECONDS + 1800))
  until curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
      echo "server container '$CONTAINER' is not running while waiting for readiness" >&2
      docker ps -a --filter "name=$CONTAINER" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' >&2 || true
      docker logs --tail 160 "$CONTAINER" >&2 || true
      return 1
    fi
    if (( SECONDS > deadline )); then
      echo "server did not become ready" >&2
      docker logs --tail 160 "$CONTAINER" >&2 || true
      return 1
    fi
    sleep 5
  done
}

kill_existing() {
  local existing
  existing="$(docker ps --filter "publish=${PORT}" --format '{{.Names}}' 2>/dev/null)"
  if [ -n "$existing" ]; then
    echo "Stopping existing container(s) on port $PORT: $existing" >&2
    echo "$existing" | xargs docker rm -f >/dev/null 2>&1
  fi
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}

start_server() {
  local draft_tokens="$1"
  local spec_json launch_log rc
  spec_json="{\"method\":\"dflash\",\"model\":\"${DFLASH_MODEL}\",\"num_speculative_tokens\":${draft_tokens}}"
  launch_log="$LOG_DIR/${draft_tokens}.launch.log"

  CONTAINER_NAME="$CONTAINER" \
  SPECULATIVE_CONFIG="$spec_json" \
  ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton_attn}" \
  DETACH=1 KEEP=1 ./serve.sh vllm "$SERVE_NAME" >"$launch_log" 2>&1
  rc=$?
  sed -n '1,80p' "$launch_log" >&2 || true
  return "$rc"
}

run_grid() {
  local tokens_list="$1"

  local base_container="$CONTAINER"
  for draft_tokens in $tokens_list; do
    CONTAINER="${base_container}-dflash-${draft_tokens}"
    echo "--- draft_tokens=$draft_tokens window=vllm container=$CONTAINER ---" >&2

    kill_existing

    if ! start_server "$draft_tokens"; then
      echo "FAILED to start container" >&2
      printf "%s\t%s\tSERVER_FAILED\n" "$draft_tokens" "vllm" | tee -a "$OUT"
      docker logs "$CONTAINER" >"$LOG_DIR/${draft_tokens}.docker.log" 2>&1 || true
      docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
      CONTAINER="$base_container"
      continue
    fi

    wait_ready || {
      echo "FAILED to become ready" >&2
      printf "%s\t%s\tSERVER_TIMEOUT\n" "$draft_tokens" "vllm" | tee -a "$OUT"
      docker logs "$CONTAINER" >"$LOG_DIR/${draft_tokens}.docker.log" 2>&1 || true
      docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
      CONTAINER="$base_container"
      continue
    }

    local gen0 dt0 acc0 draft0 req0 gen1 dt1 acc1 draft1 req1
    gen0="$(metric_value generation_tokens_total || true)"; gen0="${gen0:-0}"
    dt0="$(metric_value request_decode_time_seconds_sum || true)"; dt0="${dt0:-0}"
    acc0="$(metric_value spec_decode_num_accepted_tokens_total || true)"; acc0="${acc0:-0}"
    draft0="$(metric_value spec_decode_num_draft_tokens_total || true)"; draft0="${draft0:-0}"
    req0="$(metric_value num_requests_success_total || true)"; req0="${req0:-0}"

    docker logs --tail 0 "$CONTAINER" >/dev/null 2>&1 || true

    echo "running request max_tokens=$MAX_TOKENS prompt_tokens~$PROMPT_TOKENS" >&2
    if ! result="$(printf "%s" "$prompt" | run_one)"; then
      printf "%s\t%s\tREQUEST_FAILED\n" "$draft_tokens" "vllm" | tee -a "$OUT"
      docker logs "$CONTAINER" >"$LOG_DIR/${draft_tokens}.docker.log" 2>&1 || true
      docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
      CONTAINER="$base_container"
      continue
    fi

    sleep 2
    gen1="$(metric_value generation_tokens_total || true)"; gen1="${gen1:-0}"
    dt1="$(metric_value request_decode_time_seconds_sum || true)"; dt1="${dt1:-0}"
    acc1="$(metric_value spec_decode_num_accepted_tokens_total || true)"; acc1="${acc1:-0}"
    draft1="$(metric_value spec_decode_num_draft_tokens_total || true)"; draft1="${draft1:-0}"
    req1="$(metric_value num_requests_success_total || true)"; req1="${req1:-0}"
    stats="$(scrape_logs "$gen0" "$gen1" "$dt0" "$dt1" "$acc0" "$acc1" "$draft0" "$draft1" "$req0" "$req1")"
    printf "%s\t%s\t%s\t%s\n" "$draft_tokens" "vllm" "$result" "$stats" | tee -a "$OUT"

    docker logs "$CONTAINER" >"$LOG_DIR/${draft_tokens}.docker.log" 2>&1 || true
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    CONTAINER="$base_container"
    echo "Done." >&2
  done
}

best_candidate() {
  python3 - "$OUT" <<'PY'
import csv, sys

best = None
with open(sys.argv[1], newline="") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        try:
            draft_tokens = int(row["draft_tokens"])
            elapsed = float(row["elapsed_s"])
            completion_tokens = float(row["completion_tokens"])
        except (KeyError, TypeError, ValueError):
            continue

        try:
            score = float(row.get("gen_tok_s") or "")
        except ValueError:
            if elapsed <= 0:
                continue
            score = completion_tokens / elapsed

        if best is None or score > best[0]:
            best = (score, draft_tokens, row["window"])

if best is None:
    raise SystemExit(1)

print(f"{best[1]}\t{best[2]}")
PY
}

refine_tokens() {
  local center="$1"
  local start=$((center - 2))
  local end=$((center + 2))
  local values=()

  if (( start < 1 )); then
    start=1
  fi

  for (( value=start; value<=end; value++ )); do
    values+=("$value")
  done

  printf "%s\n" "${values[*]}"
}

filter_unmeasured_tokens() {
  local tokens="$1"
  python3 - "$OUT" "$tokens" <<'PY'
import csv, sys
path, tokens = sys.argv[1], sys.argv[2].split()
seen = set()
try:
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                float(row.get("elapsed_s") or "")
            except (TypeError, ValueError):
                continue
            seen.add(row.get("draft_tokens"))
except FileNotFoundError:
    pass
print(" ".join(t for t in tokens if t not in seen))
PY
}

mkdir -p "$(dirname "$OUT")"
mkdir -p "$LOG_DIR"

echo "=== vLLM DFlash tuning: $SERVE_NAME ===" >&2
echo "Serve script: ${SERVE_SCRIPT_PATH#$ROOT/}" >&2
echo "Container: $CONTAINER" >&2
echo "Port: $PORT" >&2
echo "Request model: $MODEL" >&2
echo "DFlash model: $DFLASH_MODEL" >&2
echo "Coarse draft tokens: $TOKENS_LIST" >&2
echo "Output: $OUT" >&2
echo "Logs: $LOG_DIR" >&2
echo "" >&2

prompt="$(make_prompt)"
printf "draft_tokens\twindow\telapsed_s\tprompt_tokens\tcompletion_tokens\taccept_len\taccept_rate\tgen_tok_s\n" | tee "$OUT"

run_grid "$TOKENS_LIST"

if [[ -z "$USER_TOKENS_LIST" ]]; then
  if best="$(best_candidate)"; then
    best_tokens="${best%%$'\t'*}"
    best_window="${best#*$'\t'}"
    narrow_tokens="$(filter_unmeasured_tokens "$(refine_tokens "$best_tokens")")"

    echo "" >&2
    echo "Best coarse candidate: draft_tokens=$best_tokens window=$best_window" >&2
    echo "Refine draft tokens: ${narrow_tokens:-<none; all already measured>}" >&2
    echo "Refine draft window: $best_window" >&2
    echo "" >&2

    if [[ -n "$narrow_tokens" ]]; then
      run_grid "$narrow_tokens"
    fi
  else
    echo "No successful coarse candidate found; skipping refine pass." >&2
  fi
fi

echo "" >&2
echo "=== Tuning complete. Results in $OUT ===" >&2
