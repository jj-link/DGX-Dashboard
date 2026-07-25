#!/usr/bin/env bash
# Runs on the DGX Spark. NVFP4 MoE-backend sweep, batch=1 decode bench.
# Fresh container per backend so the GPU is fully released between runs.
set -u
LOG=/tmp/nvfp4_sweep.log
: > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

IMG=vllm-qwen36:tx
COMMON="--tokenizer Qwen/Qwen3.6-35B-A3B-FP8 --served-model-name Qwen3.6-35B-A3B \
--host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.92 \
--kv-cache-dtype fp8 --mamba-ssm-cache-dtype float16 --mamba-cache-dtype float16 \
--enable-chunked-prefill --enable-prefix-caching --disable-custom-all-reduce \
--generation-config vllm --reasoning-parser qwen3 --enable-auto-tool-choice \
--tool-call-parser qwen3_xml --enable-flashinfer-autotune"

for B in auto flashinfer_cutlass cutlass flashinfer_cutedsl; do
  C="nvfp4-$B"
  say "=== backend: $B ==="
  if [ "$B" = auto ]; then MOE=""; else MOE="--moe-backend $B"; fi
  docker rm -f "$C" >/dev/null 2>&1
  docker run -d --name "$C" --gpus all --ipc=host --ulimit memlock=-1 \
    --ulimit stack=67108864 -p 8000:8000 -v "$HOME/models:/root/.cache/huggingface" \
    -e HF_HUB_OFFLINE=1 --entrypoint bash "$IMG" \
    -c "vllm serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 $MOE $COMMON" >/dev/null
  up=0
  for i in $(seq 1 90); do
    if curl -s -m3 http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen3.6-35B-A3B; then up=1; break; fi
    docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null | grep -q false && break
    sleep 10
  done
  if [ "$up" != 1 ]; then
    say "[$B] FAIL to come up — cause:"
    docker logs "$C" 2>&1 | grep -iE 'ValueError|RuntimeError|not support|Unsupported|out of memory|Assertion' | tail -3 | tee -a "$LOG"
    docker rm -f "$C" >/dev/null 2>&1
    continue
  fi
  say "[$B] up; benching"
  docker cp /tmp/bench_tps.py "$C":/tmp/bench_tps.py
  docker exec "$C" python3 /tmp/bench_tps.py --base http://localhost:8000 \
    --model Qwen3.6-35B-A3B --max-tokens 512 --runs 4 --warmup 1 2>&1 \
    | tee -a "$LOG" | grep -E '==> decode' | sed "s|^|[$B] |"
  docker rm -f "$C" >/dev/null 2>&1
done

say "DONE"
