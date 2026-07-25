#!/usr/bin/env bash
# EAGLE-3 tuning for the local 27B model (Qwen3.6-27B-FP8) on SGLang.
#
# Usage:
#   ./benchmarks/eagle3_tune_27b_local.sh
#   EAGLE3_TUNE_FOREGROUND=1 ./benchmarks/eagle3_tune_27b_local.sh
#
# Sweeps EAGLE-3 knobs that are the equivalent of DFlash's draft_tokens /
# draft_window: --speculative-num-draft-tokens (NUM_SPEC) and
# --speculative-eagle-topk (chain vs tree). --speculative-num-steps is
# included but defaults to 3 and rarely needs changing.
#
# Note: the canonical tune-dflash-extended.sh uses
# /home/workbench/Projects/personal/agent-benchmark-runner/tasks to build a
# long prompt. That directory does not exist on this host, so this script
# falls back to a repeated coding-task corpus that reaches PROMPT_TOKENS.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${TMUX:-}" && "${EAGLE3_TUNE_FOREGROUND:-0}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed; run with EAGLE3_TUNE_FOREGROUND=1" >&2
    exit 1
  fi

  session="${EAGLE3_TUNE_SESSION:-eagle3-tune-27b}"
  log="${EAGLE3_TUNE_LOG:-eagle3-tune-27b.log}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session '$session' already exists." >&2
    echo "Attach with: tmux attach -t $session" >&2
    echo "Watch log with: tail -f $log" >&2
    exit 1
  fi

  cmd="cd $(printf '%q' "$ROOT") && EAGLE3_TUNE_FOREGROUND=1 ./benchmarks/eagle3_tune_27b_local.sh 2>&1 | tee -a $(printf '%q' "$log")"
  tmux new-session -d -s "$session" "$cmd"
  echo "Started EAGLE-3 27B tuning in detached tmux session '$session'."
  echo "Attach with: tmux attach -t $session"
  echo "Watch log with: tail -f $log"
  exit 0
fi

CONTAINER="${CONTAINER:-sglang-serve_qwen36_27b_fp8_eagle3}"
PORT="${PORT:-8000}"

USER_NUM_SPEC_LIST="${NUM_SPEC_LIST:-}"
USER_TOPK_LIST="${TOPK_LIST:-}"
NUM_SPEC_LIST="${NUM_SPEC_LIST:-4 6 8 12}"
TOPK_LIST="${TOPK_LIST:-1 4}"
STEPS="${STEPS:-3}"
PROMPT_TOKENS="${PROMPT_TOKENS:-8192}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
OUT="${OUT:-benchmarks/results/eagle3-tune-27b-$(date +%Y%m%d-%H%M%S).tsv}"

make_prompt() {
  python3 - "$PROMPT_TOKENS" <<'PY'
import sys

n = int(sys.argv[1])

snippets = [
    "Implement a thread-safe bounded blocking queue in Go with Put/Get and a Close method. Explain the synchronization, then give the full code.",
    "Write a Python LRU cache with O(1) get/put using a doubly linked list and dict. Include docstrings and unit tests.",
    "Write a Rust parser/evaluator for arithmetic expressions with + - * / and parentheses. Provide the enum, parser, and tests.",
    "In C++, implement a templated fixed-size ring buffer with push/pop/full/empty. Show the header and a small main().",
    "Write a TypeScript debounce<T> higher-order function with leading/trailing options, fully typed, plus jest tests.",
    "Implement a persistent AVL tree in OCaml with insertion, deletion, and an inorder traversal that returns a list.",
]

prefix = (
    "You are an autonomous coding agent. For each of the following tasks, "
    "produce a complete, well-documented implementation, then reason about "
    "edge cases, tests, and trade-offs. Continue generating implementations "
    "until told to stop.\n\n"
)

body = "\n\n--- TASK ---\n\n".join(snippets)
while len(body.split()) < n:
    body = body + "\n\n--- TASK ---\n\n" + body

print(prefix + body)
PY
}

run_one() {
  python3 -c '
import json, sys, time, urllib.request, urllib.error

port = int(sys.argv[1])
max_tokens = int(sys.argv[2])
model = sys.argv[3]
prompt = sys.stdin.read()
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
' "$PORT" "$MAX_TOKENS" "$SERVED"
}

scrape_logs() {
  local tmp
  tmp="$(mktemp)"
  docker logs --since 20m "$CONTAINER" >"$tmp" 2>&1 || true
  python3 - "$tmp" <<'PY'
import re, sys

accept = []
gen = []
with open(sys.argv[1], "r", errors="replace") as fh:
    lines = list(fh)
for line in lines:
    m = re.search(r"accept len: ([0-9.]+), accept rate: ([0-9.]+).*gen throughput \(token/s\): ([0-9.]+)", line)
    if m:
        accept.append((float(m.group(1)), float(m.group(2))))
        gen.append(float(m.group(3)))
if accept:
    print(f"{sum(x for x,_ in accept)/len(accept):.3f}\t{sum(y for _,y in accept)/len(accept):.3f}\t{sum(gen)/len(gen):.3f}")
else:
    print("\t\t")
PY
  rm -f "$tmp"
}

wait_ready() {
  local deadline=$((SECONDS + 900))
  until curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    if (( SECONDS > deadline )); then
      echo "server did not become ready" >&2
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

run_grid() {
  local num_spec_list="$1"
  local topk_list="$2"

  for topk in $topk_list; do
    for draft_tokens in $num_spec_list; do
      echo "--- topk=$topk draft_tokens=$draft_tokens steps=$STEPS ---" >&2

      kill_existing

      export NUM_SPEC="$draft_tokens"
      export TOPK="$topk"
      export STEPS="$STEPS"
      export SERVED="${SERVED:-Qwen3.6-27B-FP8-EAGLE3}"

      DETACH=1 KEEP=1 ./serve.sh sglang 27b_fp8_eagle3 >/dev/null 2>&1 || {
        echo "FAILED to start container" >&2
        printf "%s\t%s\tSERVER_FAILED\n" "$topk" "$draft_tokens" | tee -a "$OUT"
        docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
        continue
      }

      wait_ready || {
        echo "FAILED to become ready" >&2
        printf "%s\t%s\tSERVER_TIMEOUT\n" "$topk" "$draft_tokens" | tee -a "$OUT"
        docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
        continue
      }

      docker logs --tail 0 "$CONTAINER" >/dev/null 2>&1 || true

      echo "running request max_tokens=$MAX_TOKENS prompt_tokens~$PROMPT_TOKENS" >&2
      if ! result="$(printf "%s" "$prompt" | run_one)"; then
        printf "%s\t%s\tREQUEST_FAILED\n" "$topk" "$draft_tokens" | tee -a "$OUT"
        docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
        continue
      fi

      sleep 2
      stats="$(scrape_logs)"
      printf "%s\t%s\t%s\t%s\n" "$topk" "$draft_tokens" "$result" "$stats" | tee -a "$OUT"

      docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
      echo "Done." >&2
    done
  done
}

best_candidate() {
  python3 - "$OUT" <<'PY'
import csv, sys

best = None
with open(sys.argv[1], newline="") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        try:
            topk = int(row["topk"])
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
            best = (score, topk, draft_tokens)

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

mkdir -p benchmarks/results
SERVED="${SERVED:-Qwen3.6-27B-FP8-EAGLE3}"

echo "=== EAGLE-3 tuning for local 27B ===" >&2
echo "Container: $CONTAINER" >&2
echo "Coarse topk values: $TOPK_LIST" >&2
echo "Coarse draft tokens: $NUM_SPEC_LIST" >&2
echo "Served name: $SERVED" >&2
echo "" >&2

prompt="$(make_prompt)"
printf "topk\tdraft_tokens\telapsed_s\tprompt_tokens\tcompletion_tokens\taccept_len\taccept_rate\tgen_tok_s\n" | tee "$OUT"

run_grid "$NUM_SPEC_LIST" "$TOPK_LIST"

if [[ -z "$USER_NUM_SPEC_LIST" && -z "$USER_TOPK_LIST" ]]; then
  if best="$(best_candidate)"; then
    best_topk="${best%%$'\t'*}"
    best_tokens="${best#*$'\t'}"
    narrow_tokens="$(refine_tokens "$best_tokens")"

    echo "" >&2
    echo "Best coarse candidate: topk=$best_topk draft_tokens=$best_tokens" >&2
    echo "Refine draft tokens: $narrow_tokens (topk=$best_topk)" >&2
    echo "" >&2

    run_grid "$narrow_tokens" "$best_topk"
  else
    echo "No successful coarse candidate found; skipping refine pass." >&2
  fi
fi

echo "" >&2
echo "=== Tuning complete. Results in $OUT ===" >&2
