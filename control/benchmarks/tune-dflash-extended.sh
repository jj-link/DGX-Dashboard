#!/usr/bin/env bash
# DFlash tuning for the local SGLang serve.sh path.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${TMUX:-}" && "${DFLASH_TUNE_FOREGROUND:-0}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then echo "tmux is not installed; run with DFLASH_TUNE_FOREGROUND=1" >&2; exit 1; fi
  session="${DFLASH_TUNE_SESSION:-dflash-tune-extended}"
  log="${DFLASH_TUNE_LOG:-dflash-tune-extended.log}"
  if tmux has-session -t "$session" 2>/dev/null; then echo "tmux session '$session' already exists." >&2; exit 1; fi
  cmd="cd $(printf '%q' "$ROOT") && DFLASH_TUNE_FOREGROUND=1 ./benchmarks/tune-dflash-extended.sh 2>&1 | tee -a $(printf '%q' "$log")"
  tmux new-session -d -s "$session" "$cmd"
  echo "Started DFlash tuning in detached tmux session '$session'."
  echo "Attach with: tmux attach -t $session"
  echo "Watch log with: tail -f $ROOT/$log"
  exit 0
fi

SERVE_TARGET="${SERVE_TARGET:-nvidia_qwen36_27b_nvfp4_dflash}"
CONTAINER="${CONTAINER:-sglang-serve_nvidia_qwen36_27b_nvfp4_dflash}"
MODEL="${MODEL:-Qwen3.6-27B-NVFP4-DFlash}"
PORT="${PORT:-8080}"
TOKENS_LIST="${TOKENS_LIST:-8 12 16 20}"
USER_TOKENS_LIST="${TOKENS_LIST:-}"
WINDOW_LIST="${WINDOW_LIST:-none 2048 4096 6144 8192}"
PROMPT_TOKENS="${PROMPT_TOKENS:-8192}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TASKS_DIR="${TASKS_DIR:-/home/workbench/Projects/personal/agent-benchmark-runner/tasks}"
OUT="${OUT:-benchmarks/results/dflash-tune-nvfp4-$(date +%Y%m%d-%H%M%S).tsv}"

export MAXLEN="${MAXLEN:-262144}"
export MEM_FRACTION="${MEM_FRACTION:-0.82}"
export RADIX="${RADIX:-1}"
export DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"

make_prompt() {
  cat <<'PROMPT'
Write a detailed engineering design and validation plan for making SGLang speculative decoding work with nvidia/Qwen3.6-27B-NVFP4 as the target model and z-lab/Qwen3.6-27B-DFlash as the draft model.

Cover architecture compatibility, ModelOpt NVFP4 loading, DFlash hidden-state capture, LM-head/logits handling, SWA behavior, KV cache materialization, scheduler behavior, attention backend constraints, CUDA graph risks, memory planning, benchmark methodology, acceptance-rate measurement, failure diagnosis, reproducibility, and rollout.

Write 40 sections. Each section must have a heading and one substantial paragraph of 5 to 7 sentences. Do not write a conclusion until all 40 sections are complete.
PROMPT
}

wait_ready() {
  local deadline=$((SECONDS + 900))
  until curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    if (( SECONDS > deadline )); then echo "server did not become ready" >&2; return 1; fi
    sleep 5
  done
}

kill_existing() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }

run_one() {
  local prompt_file
  prompt_file="$(mktemp)"
  cat > "$prompt_file"
  python3 - "$PORT" "$MODEL" "$MAX_TOKENS" "$prompt_file" <<'PY'
import json, sys, time, urllib.request, urllib.error, pathlib
port, model, max_tokens, prompt_path = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
prompt = pathlib.Path(prompt_path).read_text(errors="replace")
url = f"http://127.0.0.1:{port}/v1/chat/completions"
payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
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
obj = json.loads(body); usage = obj.get("usage", {})
print(f"{elapsed:.3f}\t{usage.get('prompt_tokens', '')}\t{usage.get('completion_tokens', '')}")
PY
  local rc=$?
  rm -f "$prompt_file"
  return "$rc"
}

scrape_logs() {
  local tmp; tmp="$(mktemp)"
  docker logs --since 20m "$CONTAINER" >"$tmp" 2>&1 || true
  python3 - "$tmp" <<'PY'
import re, sys
accept = []; gen = []
with open(sys.argv[1], "r", errors="replace") as fh:
    for line in fh:
        m = re.search(r"accept len: ([0-9.]+), accept rate: ([0-9.]+).*gen throughput \(token/s\): ([0-9.]+)", line)
        if m:
            accept.append((float(m.group(1)), float(m.group(2)))); gen.append(float(m.group(3)))
if accept: print(f"{sum(x for x,_ in accept)/len(accept):.3f}\t{sum(y for _,y in accept)/len(accept):.3f}\t{sum(gen)/len(gen):.3f}")
else: print("\t\t")
PY
  rm -f "$tmp"
}

run_grid() {
  local tokens_list="$1"; local window_list="$2"
  for draft_tokens in $tokens_list; do
    for window in $window_list; do
      echo "starting candidate draft_tokens=$draft_tokens window=$window" >&2
      kill_existing
      export NUM_SPEC="$draft_tokens" DRAFT_WINDOW="$window"
      if ! DETACH=1 KEEP=1 PORT="$PORT" ./serve.sh sglang "$SERVE_TARGET" >/dev/null 2>&1; then echo "FAILED to start container" >&2; printf "%s\t%s\tSERVER_FAILED\n" "$draft_tokens" "$window" | tee -a "$OUT"; continue; fi
      if ! wait_ready; then echo "FAILED to become ready" >&2; printf "%s\t%s\tSERVER_TIMEOUT\n" "$draft_tokens" "$window" | tee -a "$OUT"; kill_existing; continue; fi
      docker logs --since 1s "$CONTAINER" >/dev/null 2>&1 || true
      echo "running request max_tokens=$MAX_TOKENS prompt_tokens~$PROMPT_TOKENS" >&2
      if ! result="$(printf "%s" "$prompt" | run_one)"; then printf "%s\t%s\tREQUEST_FAILED\n" "$draft_tokens" "$window" | tee -a "$OUT"; kill_existing; continue; fi
      sleep 2; stats="$(scrape_logs)"
      printf "%s\t%s\t%s\t%s\n" "$draft_tokens" "$window" "$result" "$stats" | tee -a "$OUT"
      echo "finished candidate draft_tokens=$draft_tokens window=$window" >&2
    done
  done
}

best_candidate() {
  python3 - "$OUT" <<'PY'
import csv, sys
best = None
with open(sys.argv[1], newline="") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        try: score = float(row.get("gen_tok_s") or "")
        except Exception:
            try: score = float(row["completion_tokens"]) / float(row["elapsed_s"])
            except Exception: continue
        if best is None or score > best[0]: best = (score, int(row["draft_tokens"]), row["window"])
if best is None: raise SystemExit(1)
print(f"{best[1]}\t{best[2]}")
PY
}

refine_tokens() { local center="$1"; local start=$((center - 2)); local end=$((center + 2)); local values=(); if (( start < 1 )); then start=1; fi; for (( value=start; value<=end; value++ )); do values+=("$value"); done; printf "%s\n" "${values[*]}"; }

mkdir -p "$(dirname "$OUT")"
echo "=== DFlash tuning via local serve.sh ===" >&2
echo "Serve target: $SERVE_TARGET" >&2; echo "Container: $CONTAINER" >&2; echo "Port: $PORT" >&2; echo "Model: $MODEL" >&2; echo "Coarse draft tokens: $TOKENS_LIST" >&2; echo "Coarse draft windows: $WINDOW_LIST" >&2
prompt="$(make_prompt)"
printf "draft_tokens\twindow\telapsed_s\tprompt_tokens\tcompletion_tokens\taccept_len\taccept_rate\tgen_tok_s\n" | tee "$OUT"
run_grid "$TOKENS_LIST" "$WINDOW_LIST"
if [[ -z "$USER_TOKENS_LIST" ]]; then
  if best="$(best_candidate)"; then best_tokens="${best%%$'\t'*}"; best_window="${best#*$'\t'}"; narrow_tokens="$(refine_tokens "$best_tokens")"; echo "Best coarse candidate: draft_tokens=$best_tokens window=$best_window" >&2; echo "Refine draft tokens: $narrow_tokens" >&2; run_grid "$narrow_tokens" "$best_window"; else echo "No successful coarse candidate found; skipping refine pass." >&2; fi
fi
echo "=== Tuning complete. Results in $ROOT/$OUT ===" >&2
