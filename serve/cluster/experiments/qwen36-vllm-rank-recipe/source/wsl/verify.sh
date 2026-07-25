#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

PROMPT_SHORT='Say exactly: qwen rank deployment ok.'
PROMPT_LONG='Write a detailed 1200-word technical note about why tensor parallel inference should keep both worker GPUs active during long decode. Include numbered sections and do not stop early.'

curl_json() {
  local data="$1"
  ssh "$HEAD_HOST" "curl -fsS --max-time 300 'http://127.0.0.1:$API_PORT/v1/chat/completions' -H 'Content-Type: application/json' --data-binary @-" <<<"$data"
}

sample_gpus() {
  local seconds="${1:-60}"
  local out="/tmp/qwen36-gpu-sample-$RANDOM.tsv"
  {
    echo "ts host name index util memory_used memory_total"
    local end=$((SECONDS + seconds))
    while (( SECONDS < end )); do
      for host in "$HEAD_HOST" "$WORKER_HOST"; do
        ssh "$host" "nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits" | sed "s/^/$host,/"
      done
      sleep 2
    done
  } > "$out"
  echo "$out"
}

echo "=== /v1/models ==="
ssh "$HEAD_HOST" "curl -fsS --max-time 10 'http://127.0.0.1:$API_PORT/v1/models'" | tee models.json

echo "=== short generation ==="
python3 - <<'PY' > /tmp/qwen36_short_payload.json
import json
print(json.dumps({"model":"qwen36-27b-nvfp4","messages":[{"role":"user","content":"Say exactly: qwen rank deployment ok."}],"max_tokens":32,"temperature":0}))
PY
curl_json "$(cat /tmp/qwen36_short_payload.json)" | tee short.json

echo "=== long generation with GPU sampling ==="
python3 - <<'PY' > /tmp/qwen36_long_payload.json
import json
prompt='Write a detailed 1200-word technical note about why tensor parallel inference should keep both worker GPUs active during long decode. Include numbered sections and do not stop early.'
print(json.dumps({"model":"qwen36-27b-nvfp4","messages":[{"role":"user","content":prompt}],"max_tokens":1600,"temperature":0.2}))
PY
sample_gpus 180 &
SAMPLE_PID=$!
curl_json "$(cat /tmp/qwen36_long_payload.json)" | tee long.json
wait "$SAMPLE_PID"
SAMPLE_FILE=$(ls -t /tmp/qwen36-gpu-sample-*.tsv | head -1)
cp "$SAMPLE_FILE" gpu-sample.tsv

echo "=== GPU utilization summary ==="
python3 - <<'PY'
import csv, re
from collections import defaultdict
rows=[]
with open('gpu-sample.tsv') as f:
    next(f, None)
    for line in f:
        parts=[p.strip() for p in line.rstrip('\n').split(',')]
        if len(parts) < 7: continue
        host=parts[0]
        util=int(re.sub(r'[^0-9]','',parts[4]) or 0)
        mem=int(re.sub(r'[^0-9]','',parts[5]) or 0)
        rows.append((host, util, mem))
by=defaultdict(list)
for h,u,m in rows: by[h].append((u,m))
for h, vals in sorted(by.items()):
    active=sum(1 for u,m in vals if u>5)
    maxu=max([u for u,m in vals] or [0])
    avgu=sum(u for u,m in vals)/len(vals) if vals else 0
    maxm=max([m for u,m in vals] or [0])
    print(f'{h}: samples={len(vals)} active_samples={active} max_util={maxu}% avg_util={avgu:.1f}% max_mem={maxm}MiB')
PY

echo "=== recent distributed/NCCL log lines ==="
./logs.sh both 300 | egrep -i 'NCCL|rank|distributed|error|traceback|ray|collective_rpc|worker|startup|engine' | tail -120 || true
