#!/usr/bin/env bash
# DFlash-only retry: flash_attn requires non-fp8 KV, so drop --kv-cache-dtype fp8
# (default bf16 KV) for this method. Same FP8 target, same image, same bench.
set -u
LOG=/tmp/dflash_only.log
: > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

IMG=vllm-dflash:pr40898
MODEL=Qwen/Qwen3.6-35B-A3B-FP8
# BASE without --kv-cache-dtype fp8 (flash_attn needs bf16 KV)
ARGS=(--served-model-name Qwen3.6-35B-A3B --host 0.0.0.0 --port 8000
  --max-model-len 65536 --gpu-memory-utilization 0.92
  --mamba-ssm-cache-dtype float16 --mamba-cache-dtype float16
  --disable-custom-all-reduce --generation-config vllm --reasoning-parser qwen3
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --enable-flashinfer-autotune
  --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":15}'
  --attention-backend flash_attn --max-num-batched-tokens 32768)

C=dfl-dflash2
say "=== dflash (bf16 KV, flash_attn) ==="
docker rm -f "$C" >/dev/null 2>&1
docker run -d --name "$C" --gpus all --ipc=host --ulimit memlock=-1 \
  --ulimit stack=67108864 -p 8000:8000 -v "$HOME/models:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 -e FLASHINFER_DISABLE_VERSION_CHECK=1 --entrypoint vllm "$IMG" \
  serve "$MODEL" "${ARGS[@]}" >/dev/null
up=0
for i in $(seq 1 110); do
  if curl -s -m3 http://localhost:8000/v1/models 2>/dev/null | grep -q Qwen3.6-35B-A3B; then up=1; break; fi
  docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null | grep -q false && break
  sleep 10
done
if [ "$up" != 1 ]; then
  say "[dflash] FAIL to come up — cause:"
  docker logs "$C" 2>&1 | grep -iE 'ValueError|RuntimeError|not support|Unsupported|out of memory|Assertion|speculat|dflash|error:' | tail -6 | tee -a "$LOG"
  docker rm -f "$C" >/dev/null 2>&1
  say DONE; exit 0
fi
say "[dflash] up; benching"
docker cp /tmp/bench_tps.py "$C":/tmp/bench_tps.py
docker exec "$C" python3 /tmp/bench_tps.py --base http://localhost:8000 \
  --model Qwen3.6-35B-A3B --max-tokens 512 --runs 4 --warmup 1 2>&1 \
  | tee -a "$LOG" | grep -E '==> decode' | sed "s|^|[dflash] |"
acc=$(curl -s -m5 http://localhost:8000/metrics 2>/dev/null | grep -E '^vllm:spec_decode_(num_accepted_tokens|num_draft_tokens)_total' | grep -v '#' | tr '\n' ' ')
[ -n "$acc" ] && say "[dflash] spec: $acc"
docker rm -f "$C" >/dev/null 2>&1
say DONE
