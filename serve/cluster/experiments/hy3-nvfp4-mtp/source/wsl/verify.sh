#!/usr/bin/env bash
set -euo pipefail
PORT=${PORT:-8888}
MODEL=${MODEL:-hy3}
TOKENS=${TOKENS:-512}
PROMPT=${PROMPT:-Write a concise technical explanation of why speculative decoding can improve LLM serving throughput, then give a haiku about two small GPUs serving a giant model.}
export PORT MODEL TOKENS PROMPT

echo "== models =="
curl -fsS "http://127.0.0.1:${PORT}/v1/models"
echo

echo "== generation (${TOKENS} max tokens) =="
python3 - <<'PY'
import json, os, time, urllib.request
port=os.environ.get('PORT','8888')
model=os.environ.get('MODEL','hy3')
tokens=int(os.environ.get('TOKENS','512'))
prompt=os.environ.get('PROMPT') or 'Write a concise technical explanation of speculative decoding.'
body={
  'model': model,
  'messages': [{'role':'user','content': prompt}],
  'temperature': 0.9,
  'top_p': 1.0,
  'max_tokens': tokens,
}
req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
t0=time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    data=json.loads(r.read().decode())
dt=time.time()-t0
msg=data['choices'][0]['message'].get('content','')
usage=data.get('usage',{})
out=usage.get('completion_tokens') or len(msg.split())
print(msg[:2000])
print('\n== metrics ==')
print(json.dumps({'elapsed_sec': round(dt,3), 'usage': usage, 'approx_completion_tok_per_sec': round(out/dt,3) if dt else None}, indent=2))
PY

echo "== recent MTP/speculative log lines =="
docker exec hy3-head bash -lc "grep -Ei 'speculat|mtp|accept|throughput|generation' /tmp/hy3-serve.log | tail -40" || true
