#!/usr/bin/env bash
# Runs on the DGX Spark. FP8 + MTP speculative-decoding sweep vs no-spec baseline.
# argv passed directly to the vllm entrypoint (no inner shell) so the
# --speculative-config JSON survives intact.
set -u
LOG=/tmp/fp8_mtp_sweep.log
: > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

IMG=vllm-qwen36:tx
MODEL=Qwen/Qwen3.6-35B-A3B-FP8
BASE_ARGS=(--served-model-name Qwen3.6-35B-A3B --host 0.0.0.0 --port 8000
  --max-model-len 65536 --gpu-memory-utilization 0.92 --kv-cache-dtype fp8
  --mamba-ssm-cache-dtype float16 --mamba-cache-dtype float16
  --disable-custom-all-reduce --generation-config vllm --reasoning-parser qwen3
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --enable-flashinfer-autotune)

# run_variant <label> [extra argv ...]
run_variant(){
  local label="$1"; shift
  local C="fp8mtp-$label"
  say "=== variant: $label  [extra: ${*:-none}] ==="
  docker rm -f "$C" >/dev/null 2>&1
  docker run -d --name "$C" --gpus all --ipc=host --ulimit memlock=-1 \
    --ulimit stack=67108864 -p 8000:8000 -v "$HOME/models:/root/.cache/huggingface" \
    -e HF_HUB_OFFLINE=1 --entrypoint vllm "$IMG" \
    serve "$MODEL" "${BASE_ARGS[@]}" "$@" >/dev/null
  local up=0
  for i in $(seq 1 100); do
    if curl -s -m3 http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen3.6-35B-A3B; then up=1; break; fi
    docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null | grep -q false && break
    sleep 10
  done
  if [ "$up" != 1 ]; then
    say "[$label] FAIL to come up — cause:"
    docker logs "$C" 2>&1 | grep -iE 'ValueError|RuntimeError|not support|Unsupported|out of memory|Assertion|speculat|error:' | tail -4 | tee -a "$LOG"
    docker rm -f "$C" >/dev/null 2>&1
    return
  fi
  say "[$label] up; benching"
  docker cp /tmp/bench_tps.py "$C":/tmp/bench_tps.py
  docker exec "$C" python3 /tmp/bench_tps.py --base http://localhost:8000 \
    --model Qwen3.6-35B-A3B --max-tokens 512 --runs 4 --warmup 1 2>&1 \
    | tee -a "$LOG" | grep -E '==> decode' | sed "s|^|[$label] |"
  local acc
  acc=$(curl -s -m5 http://localhost:8000/metrics 2>/dev/null | grep -E '^vllm:spec_decode_(num_accepted_tokens|num_draft_tokens|num_emitted_tokens)' | grep -v '#')
  [ -n "$acc" ] && { echo "[$label] spec metrics:"; echo "$acc"; tee -a "$LOG" >/dev/null <<<"[$label] spec metrics: $acc"; }
  docker rm -f "$C" >/dev/null 2>&1
}

run_variant mtp-k1 --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'
run_variant mtp-k2 --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}'
run_variant mtp-k3 --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'

say "DONE"
