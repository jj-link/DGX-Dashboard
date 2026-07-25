#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="verify-$TS"
mkdir -p "$OUT_DIR"
BASE="http://127.0.0.1:$API_PORT/v1"

echo "[verify] models"
ssh "$HEAD_HOST" "curl -fsS --max-time 10 '$BASE/models'" | tee "$OUT_DIR/models.json" >/dev/null

echo "[verify] short chat"
ssh "$HEAD_HOST" "curl -fsS --max-time 120 '$BASE/chat/completions' -H 'Content-Type: application/json' -d @-" <<JSON | tee "$OUT_DIR/short.json" >/dev/null
{"model":"$SERVED_MODEL_NAME","messages":[{"role":"user","content":"Say exactly: SGLang DFlash TP2 OK"}],"max_tokens":32,"temperature":0}
JSON

echo "[verify] long generation with GPU sampling"
(
  for i in $(seq 1 90); do
    printf "%s\t" "$(date +%s)"
    ssh "$HEAD_HOST" "echo -n '$HEAD_HOST '; nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || docker exec '$CONTAINER' nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || true"
    printf "%s\t" "$(date +%s)"
    ssh "$WORKER_HOST" "echo -n '$WORKER_HOST '; nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || docker exec '$CONTAINER' nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || true"
    sleep 2
  done
) > "$OUT_DIR/gpu-sample.tsv" &
SAMPLE_PID=$!
ssh "$HEAD_HOST" "python3 - '$BASE/chat/completions' '$SERVED_MODEL_NAME'" <<'PY' | tee "$OUT_DIR/long.json" >/dev/null
import json, sys, urllib.request, time
url, model = sys.argv[1], sys.argv[2]
prompt = "Write a detailed reproducibility checklist for a two-node tensor-parallel inference deployment. " * 1200
payload = {"model": model, "messages": [{"role":"user","content": prompt}], "max_tokens": 1024, "temperature": 0.2}
req=urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
t0=time.time()
with urllib.request.urlopen(req, timeout=900) as r:
    body=r.read().decode()
print(body)
print(json.dumps({"elapsed_s": time.time()-t0}, indent=2), file=sys.stderr)
PY
kill "$SAMPLE_PID" >/dev/null 2>&1 || true
wait "$SAMPLE_PID" >/dev/null 2>&1 || true

echo "[verify] NCCL/RoCE log evidence"
./logs.sh both 2000 | tee "$OUT_DIR/logs-tail.txt" >/dev/null
grep -E "NCCL INFO.*NET/IB|rocep1s0f1|DFLASH|accept len|accept rate|speculative" "$OUT_DIR/logs-tail.txt" | tee "$OUT_DIR/evidence.txt" || true

echo "[verify] throughput summary"
python3 - "$OUT_DIR" <<'PY'
import json, pathlib, re, sys
out=pathlib.Path(sys.argv[1])
long=json.loads(out.joinpath('long.json').read_text().splitlines()[0])
usage=long.get('usage',{})
print('completion_tokens', usage.get('completion_tokens'))
logs=out.joinpath('logs-tail.txt').read_text(errors='replace')
vals=[float(x) for x in re.findall(r'gen throughput \(token/s\): ([0-9.]+)', logs)]
print('sglang_gen_tok_s_samples', vals[-10:])
if vals: print('sglang_gen_tok_s_last', vals[-1], 'avg_last10', sum(vals[-10:])/min(len(vals),10))
for host in ['spark2-ts','spark3-ts']:
    utils=[]
    for line in out.joinpath('gpu-sample.tsv').read_text(errors='replace').splitlines():
        if host in line:
            m=re.search(host+r'\s+(\d+)', line)
            if m: utils.append(int(m.group(1)))
    if utils: print(host, 'samples', len(utils), 'active', sum(u>0 for u in utils), 'max_util', max(utils), 'avg_util', round(sum(utils)/len(utils),1))
PY

echo "wrote $OUT_DIR"
