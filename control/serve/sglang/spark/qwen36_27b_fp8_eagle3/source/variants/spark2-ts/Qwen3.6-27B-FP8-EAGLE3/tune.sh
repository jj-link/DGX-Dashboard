#!/usr/bin/env bash
# EAGLE-3 tuning for Qwen3.6-27B-FP8 on Spark via docker-compose.
#
# Run from the compose directory:
#   cd ~/inference/sglang/Qwen3.6-27B-FP8-EAGLE3
#   ../scripts/tune-eagle3-spark.sh
#
# Sweeps --speculative-eagle-topk (chain vs tree) and
# --speculative-num-draft-tokens. Uses compose env substitution for NUM_SPEC/TOPK.

set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="$MODEL_DIR/docker-compose.yml"

if [[ ! -f "$COMPOSE" ]]; then
  echo "No docker-compose.yml in $MODEL_DIR" >&2
  exit 1
fi
cd "$MODEL_DIR"

CONTAINER="sglang-qwen36-27b-fp8-eagle3"
PORT=8001
MODEL="Qwen3.6-27B-FP8-EAGLE3"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"

USER_NUM_SPEC_LIST="${NUM_SPEC_LIST:-}"
USER_TOPK_LIST="${TOPK_LIST:-}"
NUM_SPEC_LIST="${NUM_SPEC_LIST:-4 6 8 12}"
TOPK_LIST="${TOPK_LIST:-1 4}"
PROMPT_TOKENS="${PROMPT_TOKENS:-8192}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
OUT="${OUT:-eagle3-tune-$(date +%Y%m%d-%H%M%S).tsv}"

if [[ -z "${TMUX:-}" && "${EAGLE3_TUNE_FOREGROUND:-0}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed; run with EAGLE3_TUNE_FOREGROUND=1" >&2
    exit 1
  fi
  session="${EAGLE3_TUNE_SESSION:-eagle3-tune-spark}"
  log="${EAGLE3_TUNE_LOG:-eagle3-tune-spark.log}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session '$session' already exists" >&2
    exit 1
  fi
  cmd="cd $(printf '%q' "$MODEL_DIR") && EAGLE3_TUNE_FOREGROUND=1 $(printf '%q' "$0") 2>&1 | tee -a $(printf '%q' "$log")"
  tmux new-session -d -s "$session" "$cmd"
  echo "Started EAGLE-3 Spark tuning in detached tmux session '$session'."
  echo "Attach with: tmux attach -t $session"
  echo "Watch log with: tail -f $MODEL_DIR/$log"
  exit 0
fi

make_prompt() {
  python3 - "$PROMPT_TOKENS" <<'PY'
import os, sys

n = int(sys.argv[1])
tasks_dir = os.path.expanduser("~/projects/agent-benchmark-runner/tasks")

descriptions = []
if os.path.isdir(tasks_dir):
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
    snippets = [
        "Implement a thread-safe bounded blocking queue in Go with Put/Get and a Close method.",
        "Write a Python LRU cache with O(1) get/put using a doubly linked list and dict.",
        "Write a Rust parser/evaluator for arithmetic expressions with + - * / and parentheses.",
        "In C++, implement a templated fixed-size ring buffer with push/pop/full/empty.",
        "Write a TypeScript debounce<T> higher-order function with leading/trailing options.",
        "Implement a persistent AVL tree in OCaml with insertion and deletion.",
    ]
    descriptions = snippets

body = "\n\n--- TASK ---\n\n".join(descriptions)
while len(body.split()) < n:
    body = body + "\n\n--- TASK ---\n\n" + body

print(
    "You are an autonomous coding agent. For each task, produce a complete, "
    "well-documented implementation, then reason about edge cases, tests, and trade-offs. "
    "Continue generating implementations until told to stop.\n\n" + body
)
PY
}

run_one() {
  local prompt_file="$1"
  python3 - "$URL" "$MODEL" "$MAX_TOKENS" "$prompt_file" <<'PY'
import json, sys, time, urllib.request, urllib.error

url, model, max_tokens, prompt_file = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
with open(prompt_file, "r", errors="replace") as fh:
    prompt = fh.read()
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
print(f"{elapsed:.3f}\t{usage.get('prompt_tokens', '')}\t{usage.get('completion_tokens', '')}")
PY
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
  until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
    if (( SECONDS > deadline )); then
      echo "server did not become ready" >&2
      return 1
    fi
    sleep 5
  done
}

run_grid() {
  local num_spec_list="$1"
  local topk_list="$2"

  for topk in $topk_list; do
    for draft_tokens in $num_spec_list; do
      echo "--- topk=$topk draft_tokens=$draft_tokens ---" >&2

      docker compose down >/dev/null 2>&1 || true

      TOPK="$topk" NUM_SPEC="$draft_tokens" docker compose up -d || {
        echo "FAILED to start container" >&2
        printf "%s\t%s\tSERVER_FAILED\n" "$topk" "$draft_tokens" | tee -a "$OUT"
        continue
      }

      wait_ready || {
        echo "FAILED to become ready" >&2
        printf "%s\t%s\tSERVER_TIMEOUT\n" "$topk" "$draft_tokens" | tee -a "$OUT"
        docker compose down >/dev/null 2>&1 || true
        continue
      }

      docker logs --tail 0 "$CONTAINER" >/dev/null 2>&1 || true

      echo "running request max_tokens=$MAX_TOKENS prompt_tokens~$PROMPT_TOKENS" >&2
      prompt_file="$(mktemp)"
      printf "%s" "$prompt" >"$prompt_file"
      if ! result="$(run_one "$prompt_file")"; then
        rm -f "$prompt_file"
        printf "%s\t%s\tREQUEST_FAILED\n" "$topk" "$draft_tokens" | tee -a "$OUT"
        docker compose down >/dev/null 2>&1 || true
        continue
      fi

      rm -f "$prompt_file"
      sleep 2
      stats="$(scrape_logs)"
      printf "%s\t%s\t%s\t%s\n" "$topk" "$draft_tokens" "$result" "$stats" | tee -a "$OUT"

      docker compose down >/dev/null 2>&1 || true
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
  if (( start < 1 )); then start=1; fi
  for (( value=start; value<=end; value++ )); do
    if (( value == center )); then continue; fi
    values+=("$value")
  done
  printf "%s\n" "${values[*]}"
}

echo "=== EAGLE-3 Spark tuning ===" >&2
echo "Compose dir: $MODEL_DIR" >&2
echo "Container: $CONTAINER" >&2
echo "Port: $PORT" >&2
echo "Coarse topk values: $TOPK_LIST" >&2
echo "Coarse draft tokens: $NUM_SPEC_LIST" >&2
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
