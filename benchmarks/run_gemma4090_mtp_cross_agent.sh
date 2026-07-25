#!/usr/bin/env bash
set -euo pipefail
cd /home/workbench/inference/benchmarks

export OPENAI_API_BASE=http://localhost:8001/v1
export OPENAI_API_KEY=***
export AIDER_MODEL_SETTINGS_FILE=/home/workbench/inference/benchmarks/aider-gemma-model-settings.yml
export AIDER_MODEL_METADATA_FILE=/home/workbench/inference/benchmarks/aider-gemma-model-metadata.json
export XDG_CONFIG_HOME=/home/workbench/inference/benchmarks/opencode-gemma-config
export CONCURRENCY=1

MODEL_ID=gemma-4-12b-it-UD-Q4_K_XL.gguf
OPENCODE_MODEL=local-gemma-mtp/gemma-4-12b-it-UD-Q4_K_XL.gguf

echo "[gemma4090] restarting stable MTP server: draft-mtp, n-max=2, flash-attn=off, parallel=1, ctx-checkpoints=0"
docker rm -f llama-gemma4-12b >/dev/null 2>&1 || true
docker run -d --name llama-gemma4-12b \
  --gpus device=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p 127.0.0.1:8001:8080 \
  -v /home/workbench/models/hub/unsloth/gemma-4-12b-it-GGUF:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/gemma-4-12b-it-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 --port 8080 \
  --n-gpu-layers 999 \
  --ctx-size 131072 \
  --cache-ram 0 \
  --ctx-checkpoints 0 \
  --parallel 1 \
  --no-warmup \
  --flash-attn off \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-model /models/MTP/gemma-4-12b-it-F16-MTP.gguf

for i in $(seq 1 120); do
  if python3 - <<'PY'
import json, urllib.request
try:
    data=json.loads(urllib.request.urlopen('http://localhost:8001/v1/models', timeout=2).read().decode())
    print(data['data'][0]['id'], data['data'][0].get('meta',{}).get('n_ctx'))
except Exception:
    raise SystemExit(1)
PY
  then break; fi
  sleep 2
  if [ "$i" = 120 ]; then docker logs --tail 120 llama-gemma4-12b; exit 1; fi
done

echo "[gemma4090] docker status before benchmark:"
docker inspect llama-gemma4-12b --format '{{.State.Status}} {{.State.ExitCode}} {{json .Args}}'

echo "[gemma4090] starting single-shot full benchmark"
python3 cross_agent_oneshot.py \
  --base-url http://localhost:8001/v1 \
  --api-key dummy \
  --served-model "$MODEL_ID" \
  --opencode-model "$OPENCODE_MODEL" \
  --agents aider,opencode \
  --langs python,javascript,go,rust,cpp,java \
  --num-tests all \
  --concurrency 1 \
  --agent-timeout 900 \
  --test-timeout 300

echo "[gemma4090] docker status after single-shot:"
docker inspect llama-gemma4-12b --format '{{.State.Status}} {{.State.ExitCode}}'

echo "[gemma4090] starting multi-turn full benchmark"
python3 cross_agent_multiturn.py \
  --base-url http://localhost:8001/v1 \
  --api-key dummy \
  --served-model "$MODEL_ID" \
  --opencode-model "$OPENCODE_MODEL" \
  --agents aider,opencode \
  --langs python,javascript,go,rust,cpp,java \
  --num-tests all \
  --turns 3 \
  --concurrency 1 \
  --agent-timeout 900 \
  --test-timeout 300

echo "[gemma4090] docker status after multi-turn:"
docker inspect llama-gemma4-12b --format '{{.State.Status}} {{.State.ExitCode}}'

echo "[gemma4090] latest summaries:"
python3 - <<'PY'
from pathlib import Path
import json
base=Path('results')
for pattern in ['cross-agent-oneshot-*.json','cross-agent-multiturn-*.json']:
    p=max(base.glob(pattern), key=lambda x:x.stat().st_mtime)
    d=json.loads(p.read_text())
    print(p)
    print(json.dumps(d.get('summary'), indent=2))
PY
