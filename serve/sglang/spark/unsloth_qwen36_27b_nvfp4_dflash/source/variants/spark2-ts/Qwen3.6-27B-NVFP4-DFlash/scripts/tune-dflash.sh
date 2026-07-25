#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${TMUX:-}" && "${DFLASH_TUNE_FOREGROUND:-0}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not installed; run with DFLASH_TUNE_FOREGROUND=1 to run in this SSH session" >&2
    exit 1
  fi

  session="${DFLASH_TUNE_SESSION:-dflash-tune}"
  log="${DFLASH_TUNE_LOG:-tune-run.log}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session '$session' already exists." >&2
    echo "Attach with: tmux attach -t $session" >&2
    echo "Watch log with: tail -f $log" >&2
    exit 1
  fi

  cmd="cd $(printf '%q' "$ROOT") && DFLASH_TUNE_FOREGROUND=1 ./scripts/tune-dflash.sh 2>&1 | tee -a $(printf '%q' "$log")"
  tmux new-session -d -s "$session" "$cmd"
  echo "Started dflash tuning in detached tmux session '$session'."
  echo "Attach with: tmux attach -t $session"
  echo "Watch log with: tail -f $log"
  exit 0
fi

MODEL="${MODEL:-Qwen3.6-27B-NVFP4-DFLASH}"
URL="${URL:-http://127.0.0.1:8000/v1/chat/completions}"
TOKENS_LIST="${TOKENS_LIST:-8 12 16}"
WINDOW_LIST="${WINDOW_LIST:-none 2048 4096 6144 8192}"
PROMPT_TOKENS="${PROMPT_TOKENS:-8192}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TASKS_DIR="${TASKS_DIR:-$HOME/projects/agent-benchmark-runner/tasks}"
OUT="${OUT:-dflash-tune-$(date +%Y%m%d-%H%M%S).tsv}"
COMPOSE="docker-compose.yml"
BACKUP=".docker-compose.yml.tune-backup"

make_prompt() {
  python3 - "$PROMPT_TOKENS" "$TASKS_DIR" <<'PY'
import os
import sys

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

patch_compose() {
  local draft_tokens="$1"
  local window="$2"
  python3 - "$COMPOSE" "$draft_tokens" "$window" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
draft_tokens = sys.argv[2]
window = sys.argv[3]
lines = path.read_text().splitlines()

def set_arg(lines, arg, value):
    out = []
    i = 0
    found = False
    while i < len(lines):
        out.append(lines[i])
        if lines[i].strip() == f"- {arg}":
            if i + 1 < len(lines):
                out.append(f'      - "{value}"')
                i += 2
                found = True
                continue
        i += 1
    if not found:
        marker = "      - --speculative-num-draft-tokens"
        for idx, line in enumerate(out):
            if line == marker:
                out[idx + 2:idx + 2] = [f"      - {arg}", f'      - "{value}"']
                break
    return out

def remove_arg(lines, arg):
    out = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == f"- {arg}":
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return out

lines = set_arg(lines, "--speculative-num-draft-tokens", draft_tokens)
lines = remove_arg(lines, "--speculative-draft-window-size")
if window != "none":
    lines = set_arg(lines, "--speculative-draft-window-size", window)
path.write_text("\n".join(lines) + "\n")
PY
}

wait_ready() {
  local deadline=$((SECONDS + 900))
  until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do
    if (( SECONDS > deadline )); then
      echo "server did not become ready" >&2
      return 1
    fi
    sleep 5
  done
}

run_one() {
  local prompt="$1"
  python3 - "$URL" "$MODEL" "$MAX_TOKENS" "$prompt" <<'PY'
import json
import sys
import time
import urllib.request
import urllib.error

url, model, max_tokens, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
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
  docker logs --since 20m sglang-qwen36-dflash >"$tmp" 2>&1 || true
  python3 - "$tmp" <<'PY'
import re
import sys

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

cp "$COMPOSE" "$BACKUP"
trap 'mv "$BACKUP" "$COMPOSE"; docker compose down >/dev/null 2>&1 || true' EXIT

prompt="$(make_prompt)"
printf "draft_tokens\twindow\telapsed_s\tprompt_tokens\tcompletion_tokens\taccept_len\taccept_rate\tgen_tok_s\n" | tee "$OUT"

for draft_tokens in $TOKENS_LIST; do
  for window in $WINDOW_LIST; do
    echo "starting candidate draft_tokens=$draft_tokens window=$window" >&2
    patch_compose "$draft_tokens" "$window"
    echo "restarting sglang" >&2
    docker compose down >/dev/null 2>&1
    docker compose up -d >/dev/null 2>&1
    echo "waiting for sglang readiness" >&2
    wait_ready
    docker logs --since 1s sglang-qwen36-dflash >/dev/null 2>&1 || true
    echo "running request max_tokens=$MAX_TOKENS prompt_tokens~$PROMPT_TOKENS" >&2
    if ! result="$(run_one "$prompt")"; then
      printf "%s\t%s\tREQUEST_FAILED\n" "$draft_tokens" "$window" | tee -a "$OUT"
      docker compose down >/dev/null 2>&1 || true
      continue
    fi
    sleep 2
    stats="$(scrape_logs)"
    printf "%s\t%s\t%s\t%s\n" "$draft_tokens" "$window" "$result" "$stats" | tee -a "$OUT"
    echo "finished candidate draft_tokens=$draft_tokens window=$window" >&2
  done
done
