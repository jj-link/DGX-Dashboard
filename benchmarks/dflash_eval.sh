#!/usr/bin/env bash
# Serve Qwen3.6-27B (+ DFlash drafter at SPEC spec-tokens) and measure single-stream
# decode tok/s across a SET of coding problems (1 per language) to average out
# content variance. Each problem run REPS times. Sampling = Qwen's precise-coding
# recommendation: temp 0.6 / top_p 0.95 / top_k 20 / min_p 0 / penalties 0,1.
# Per spec value: (#problems x REPS) measurements -> mean +/- std.
# Usage: [REPS=3] [PROBLEMS="python:react cpp:zebra-puzzle rust:forth"] dflash_eval.sh <spec|off> [bf16|fp8]
set -uo pipefail
SPEC="${1:?spec tokens or 'off'}"
PREC="${2:-bf16}"
REPS="${REPS:-3}"
PROBLEMS="${PROBLEMS:-python:react cpp:zebra-puzzle rust:forth}"
TEMP="${TEMP:-0.6}"
BENCH=/home/workbench/Projects/personal/test/benchmarks
AIDERDIR="$BENCH/aider"
CACHE=/home/workbench/.cache/huggingface
SERVED=qwen36-27b-dflash
case "$PREC" in
  bf16) TARGET="Qwen/Qwen3.6-27B" ;;
  fp8)  TARGET="Qwen/Qwen3.6-27B-FP8" ;;
  *) echo "bad prec '$PREC'"; exit 1 ;;
esac

echo ">>> serve prec=$PREC spec=$SPEC temp=$TEMP reps=$REPS problems=[$PROBLEMS]  $(date +%H:%M:%S)"
docker rm -f vllm-dflash >/dev/null 2>&1
SPECARG=()
[ "$SPEC" != "off" ] && SPECARG=(--speculative-config "{\"method\":\"dflash\",\"model\":\"z-lab/Qwen3.6-27B-DFlash\",\"num_speculative_tokens\":$SPEC}")
docker run -d --name vllm-dflash --device nvidia.com/gpu=all \
  --security-opt no-new-privileges --cap-drop ALL --pids-limit 4096 \
  -p 127.0.0.1:8000:8000 --tmpfs /tmp:rw,nosuid,nodev,size=4g \
  -e HOME=/cache -v vllm-compile-cache:/cache \
  -e HF_HOME=/models -e HF_HUB_OFFLINE=1 -v "$CACHE":/models:ro \
  --entrypoint vllm qwen-vllm:dflash serve "$TARGET" \
    --host 0.0.0.0 --port 8000 --served-model-name "$SERVED" "${SPECARG[@]}" \
    --max-model-len 131072 --gpu-memory-utilization 0.92 --max-num-seqs 32 \
    --max-num-batched-tokens 32768 --reasoning-parser qwen3 >/dev/null 2>&1

end=$(( $(date +%s) + 720 )); ready=0
while [ "$(date +%s)" -lt "$end" ]; do
  curl -fsS --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "$SERVED" && { ready=1; break; }
  docker ps --filter name=vllm-dflash --format '{{.Names}}' | grep -q vllm-dflash || { echo "CONTAINER DIED"; docker logs --tail 40 vllm-dflash 2>&1 | tail -40; exit 2; }
  sleep 6
done
[ "$ready" = 1 ] || { echo "TIMEOUT"; exit 3; }
echo ">>> ready $(date +%H:%M:%S)"

metric(){ curl -fsS --max-time 4 http://127.0.0.1:8000/metrics 2>/dev/null | grep "^vllm:$1{" | awk '{print $2}' | head -1; }

cat > "$AIDERDIR/_dflash_settings.yml" <<YAML
- name: openai/$SERVED
  edit_format: whole
  use_repo_map: false
  send_undo_reply: false
  examples_as_sys_msg: true
  streaming: true
  extra_params:
    max_tokens: 49152
    temperature: $TEMP
    top_p: 0.95
    top_k: 20
    min_p: 0.0
    presence_penalty: 0.0
    repetition_penalty: 1.0
YAML

TPSLIST=(); ALLIST=()
for prob in $PROBLEMS; do
  lang="${prob%%:*}"; name="${prob##*:}"
  for rep in $(seq 1 "$REPS"); do
    A0=$(metric spec_decode_num_accepted_tokens_total); D0=$(metric spec_decode_num_drafts_total)
    T0=$(metric spec_decode_num_draft_tokens_total); G0=$(metric generation_tokens_total)
    DT0=$(metric request_decode_time_seconds_sum)
    A0=${A0:-0}; D0=${D0:-0}; T0=${T0:-0}; G0=${G0:-0}; DT0=${DT0:-0}

    run="dflash-${PREC}-spec${SPEC}-${lang}_${name}-r${rep}-$(date +%H%M%S)"
    echo ">>> $lang/$name rep $rep/$REPS  ($run)  $(date +%H:%M:%S)"
    docker run --rm --network host \
      -v "$AIDERDIR:/aider" -v "$AIDERDIR/tmp.benchmarks:/benchmarks" \
      -e AIDER_DOCKER=1 -e AIDER_BENCHMARK_DIR=/benchmarks \
      -e OPENAI_API_BASE=http://localhost:8000/v1 -e OPENAI_API_KEY=dummy \
      -w /aider aider-benchmark \
      python3 benchmark/benchmark.py "$run" --model "openai/$SERVED" \
        --read-model-settings /aider/_dflash_settings.yml --edit-format whole \
        --threads 1 --languages "$lang" --keywords "$name" --num-tests 1 --tries 2 --new \
        --exercises-dir polyglot-benchmark >/dev/null 2>&1

    A1=$(metric spec_decode_num_accepted_tokens_total); D1=$(metric spec_decode_num_drafts_total)
    T1=$(metric spec_decode_num_draft_tokens_total); G1=$(metric generation_tokens_total)
    DT1=$(metric request_decode_time_seconds_sum)
    A1=${A1:-0}; D1=${D1:-0}; T1=${T1:-0}; G1=${G1:-0}; DT1=${DT1:-0}
    read tps al gen < <(python3 - "$A0" "$A1" "$D0" "$D1" "$G0" "$G1" "$DT0" "$DT1" <<'PY'
import sys
A0,A1,D0,D1,G0,G1,DT0,DT1=map(float,sys.argv[1:])
G=G1-G0; DT=DT1-DT0; D=D1-D0; A=A1-A0
print(f"{(G/DT if DT>0 else 0):.1f} {(A/D if D>0 else 0):.2f} {G:.0f}")
PY
)
    echo "    $lang/$name r$rep: decode_tok/s=$tps  accept_len=$al  gen_tokens=$gen"
    TPSLIST+=("$tps"); ALLIST+=("$al")
  done
done

python3 - "$PREC" "$SPEC" "$TEMP" "${TPSLIST[*]}" "${ALLIST[*]}" <<'PY' | tee -a "$BENCH/results/dflash_sweep_t06.txt"
import sys, statistics as st
prec,spec,temp,tps_s,al_s=sys.argv[1:]
tps=[float(x) for x in tps_s.split()]; al=[float(x) for x in al_s.split()]
def ms(a): return (st.mean(a), st.stdev(a) if len(a)>1 else 0.0)
mt,sd=ms(tps); ma,sa=ms(al)
print(f"=== RESULT prec={prec} spec={spec} temp={temp} n={len(tps)}  "
      f"decode_tok/s = {mt:.1f} +/- {sd:.1f}  (all: {', '.join(f'{x:.0f}' for x in tps)})")
print(f"    accept_len = {ma:.2f} +/- {sa:.2f}")
PY
echo ">>> done $(date +%H:%M:%S)"
