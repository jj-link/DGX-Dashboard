#!/usr/bin/env bash
# Runs on the DGX Spark. FP8 optimization sweep, batch=1 decode bench.
# Fresh container per variant so the GPU is fully released between runs.
set -u
LOG=/tmp/fp8_sweep.log
: > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

IMG=vllm-qwen36:tx
MODEL=Qwen/Qwen3.6-35B-A3B-FP8
# Base flags WITHOUT the toggled ones (prefix-cache / chunked-prefill / moe-backend).
BASE="--served-model-name Qwen3.6-35B-A3B --host 0.0.0.0 --port 8000 \
--max-model-len 65536 --gpu-memory-utilization 0.92 --kv-cache-dtype fp8 \
--mamba-ssm-cache-dtype float16 --mamba-cache-dtype float16 \
--disable-custom-all-reduce --generation-config vllm --reasoning-parser qwen3 \
--enable-auto-tool-choice --tool-call-parser qwen3_xml --enable-flashinfer-autotune"

# label | extra flags appended to BASE
run_variant(){
  local label="$1"; shift
  local extra="$*"
  local C="fp8-$label"
  say "=== variant: $label  [extra: ${extra:-none}] ==="
  docker rm -f "$C" >/dev/null 2>&1
  docker run -d --name "$C" --gpus all --ipc=host --ulimit memlock=-1 \
    --ulimit stack=67108864 -p 8000:8000 -v "$HOME/models:/root/.cache/huggingface" \
    -e HF_HUB_OFFLINE=1 --entrypoint bash "$IMG" \
    -c "vllm serve $MODEL $BASE $extra" >/dev/null
  local up=0
  for i in $(seq 1 90); do
    if curl -s -m3 http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen3.6-35B-A3B; then up=1; break; fi
    docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null | grep -q false && break
    sleep 10
  done
  if [ "$up" != 1 ]; then
    say "[$label] FAIL to come up — cause:"
    docker logs "$C" 2>&1 | grep -iE 'ValueError|RuntimeError|not support|Unsupported|out of memory|Assertion' | tail -3 | tee -a "$LOG"
    docker rm -f "$C" >/dev/null 2>&1
    return
  fi
  say "[$label] up; benching"
  docker cp /tmp/bench_tps.py "$C":/tmp/bench_tps.py
  docker exec "$C" python3 /tmp/bench_tps.py --base http://localhost:8000 \
    --model Qwen3.6-35B-A3B --max-tokens 512 --runs 4 --warmup 1 2>&1 \
    | tee -a "$LOG" | grep -E '==> decode' | sed "s|^|[$label] |"
  docker rm -f "$C" >/dev/null 2>&1
}

run_variant baseline           --enable-prefix-caching --enable-chunked-prefill
run_variant no-pc-no-cp
run_variant no-pc-no-cp+deepgemm   --moe-backend deep_gemm
run_variant no-pc-no-cp+triton     --moe-backend triton
run_variant baseline+deepgemm      --enable-prefix-caching --enable-chunked-prefill --moe-backend deep_gemm

say "DONE"
