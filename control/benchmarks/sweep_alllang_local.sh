#!/usr/bin/env bash
# Multi-turn aider benchmark across ALL 6 languages for the currently served
# local model, run INSIDE the `aider-benchmark` docker image. That image bundles the
# go/rust/node+jest/openjdk/cmake toolchains AND the /aider/benchmark/
# {npm-test.sh,cpp-test.sh} scripts that benchmark.py's TEST_COMMANDS point
# at — which is the ONLY way the non-python/java languages actually execute
# their tests here (the host has no go/cargo and the js/cpp scripts live at
# those in-image paths). Running on the host scored only python+java; running
# in-image scores all 6.
#
# This script does not launch or swap models. Start the vLLM server first; this
# only benchmarks whatever is live at localhost:8000.
#
# Does NOT touch the spark (the 122b resume runs there, separate machine).
set -uo pipefail

BENCH="/home/workbench/Projects/personal/test/benchmarks"
AIDERDIR="$BENCH/aider"
IMG="${IMG:-aider-benchmark}"
LANGS="${LANGS:-python,javascript,go,rust,cpp,java}"
LOCAL_URL="http://localhost:8000/v1"
MAX_TOKENS="${MAX_TOKENS:-49152}"
TRIES="${TRIES:-2}"
EDIT_FORMAT="${EDIT_FORMAT:-whole}"
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$BENCH/results"
LOG="$BENCH/results/sweep-alllang-local-${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "[alllang] langs=$LANGS tries=$TRIES image=$IMG log=$LOG"

served=$(curl -fsS --max-time 5 "$LOCAL_URL/models" 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4 || true)
[ -n "$served" ] || { echo "[alllang] no model served at $LOCAL_URL" >&2; exit 2; }

# KV-based threads from the live vLLM container's log (cap 32 = max-num-seqs).
vcid=$(docker ps -q --filter "name=^vllm-" | head -1)
KV=""
[ -n "$vcid" ] && KV=$(docker logs "$vcid" 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+ tokens' | tail -1 | grep -oE '[0-9,]+' | tr -d ',' || true)
if [ -n "$KV" ]; then THREADS=$(awk -v k="$KV" 'BEGIN{t=int(k/32768); if(t<1)t=1; if(t>32)t=32; print t}'); else THREADS=16; fi

cat > "$AIDERDIR/_sweep_settings.yml" <<YAML
- name: openai/$served
  edit_format: $EDIT_FORMAT
  use_repo_map: false
  send_undo_reply: false
  examples_as_sys_msg: true
  streaming: true
  extra_params:
    max_tokens: $MAX_TOKENS
    temperature: 0.0
YAML

run="${served//[^A-Za-z0-9]/_}-aiderdkr-$(date +%Y%m%d-%H%M%S)"
echo "======== served=$served KV=${KV:-?} threads=$THREADS run=$run :: bench $(date +%H:%M:%S) ========"
echo "[alllang] running benchmark..."

cname="aider-bench-${run}"
cleanup() {
  docker rm -f "$cname" >/dev/null 2>&1 || true
}
trap cleanup INT TERM

set +e
docker run --rm --name "$cname" --network host \
  -v "$AIDERDIR:/aider" -v "$AIDERDIR/tmp.benchmarks:/benchmarks" \
  -e AIDER_DOCKER=1 -e AIDER_BENCHMARK_DIR=/benchmarks \
  -e OPENAI_API_BASE="$LOCAL_URL" -e OPENAI_API_KEY=dummy \
  -e PYTHONPATH=/aider:/aider/benchmark \
  -w /aider "$IMG" \
  python3 benchmark/benchmark.py "$run" --model "openai/$served" \
    --read-model-settings /aider/_sweep_settings.yml --edit-format "$EDIT_FORMAT" \
    --threads "$THREADS" --languages "$LANGS" --num-tests "${NUM_TESTS:--1}" \
    --tries "$TRIES" --new --exercises-dir polyglot-benchmark \
  2>&1 | python3 -c '
import os, re, sys

buf = ""
patterns = [
    re.compile(r"\d+%\|.*\|\s*\d+/\d+\s*\[.*<.*,\s*[\d.]+s/it\]"),
    re.compile(r"\d+%\|.*\|\s*\d+/\d+\s*\[.*<.*,\s*[\d.]+it/s\]"),
    re.compile(r"The LLM did not conform to the edit format\."),
]

def emit(part):
    text = part.strip()
    if not text:
        return
    if any(p.search(text) for p in patterns):
        print(text, flush=True)

while True:
    chunk = os.read(0, 4096)
    if not chunk:
        break
    buf += chunk.decode("utf-8", "replace")
    pieces = re.split(r"[\r\n]+", buf)
    buf = pieces.pop()
    for piece in pieces:
        emit(piece)
emit(buf)
'
bench_status=${PIPESTATUS[0]}

if [ "$bench_status" -ne 0 ]; then
  echo "[alllang] benchmark returned nonzero"
fi
trap - INT TERM

rundir=$(ls -td "$AIDERDIR"/tmp.benchmarks/*--"$run" 2>/dev/null | head -1)
if [ -n "$rundir" ]; then
  docker run --rm \
    -v "$AIDERDIR:/aider" -v "$AIDERDIR/tmp.benchmarks:/benchmarks" \
    -e AIDER_BENCHMARK_DIR=/benchmarks -e PYTHONPATH=/aider:/aider/benchmark \
    -w /aider "$IMG" \
    python3 benchmark/benchmark.py --stats "/benchmarks/$(basename "$rundir")" \
    > "$rundir/_stats.yml" 2>&1 || true
  echo "[alllang] stats -> $rundir/_stats.yml"
fi

echo "ALLLANG_SWEEP_COMPLETE $(date +%H:%M:%S)"
