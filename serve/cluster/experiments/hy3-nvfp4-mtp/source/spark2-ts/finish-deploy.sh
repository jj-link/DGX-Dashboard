#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_DIR=${MODEL_DIR:-/home/jjlink/models/hy3-nvfp4-w4a16}
WORKER_SSH_TARGET=${WORKER_SSH_TARGET:-jjlink@10.0.0.2}
SSH_KEY=${SSH_KEY:-/home/jjlink/.ssh/gb10_ed25519}
PORT=${PORT:-8888}
ssh_worker=(ssh -i "${SSH_KEY}" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "${WORKER_SSH_TARGET}")
rsync_ssh="ssh -i ${SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

count_shards() { find "${MODEL_DIR}" -maxdepth 1 -name 'model-*.safetensors' | wc -l; }

echo "== wait for Spark2 HF download to complete =="
while true; do
  shards=$(count_shards)
  size=$(du -sh "${MODEL_DIR}" | awk '{print $1}')
  echo "$(date -Is) Spark2: ${shards}/99 shards, ${size}"
  if [ "${shards}" -eq 99 ] && [ -f "${MODEL_DIR}/config.json" ] && [ -f "${MODEL_DIR}/model.safetensors.index.json" ]; then
    break
  fi
  sleep 120
done

echo "== stop any worker-side duplicate HF download before authoritative fabric sync =="
"${ssh_worker[@]}" "pkill -f 'hf download kodelow/Hy3-NVFP4-W4A16' 2>/dev/null || true; rm -rf '${MODEL_DIR}/.cache'"

echo "== rsync completed model from Spark2 to Spark3 over fabric =="
rsync -a --delete --exclude '.cache/' -e "${rsync_ssh}" "${MODEL_DIR}/" "${WORKER_SSH_TARGET}:${MODEL_DIR}/"

echo "== verify model layout on both nodes =="
head_shards=$(count_shards)
worker_shards=$("${ssh_worker[@]}" "find '${MODEL_DIR}' -maxdepth 1 -name 'model-*.safetensors' | wc -l")
head_size=$(du -sh "${MODEL_DIR}" | awk '{print $1}')
worker_size=$("${ssh_worker[@]}" "du -sh '${MODEL_DIR}' | awk '{print \$1}'")
echo "Spark2: ${head_shards} shards, ${head_size}"
echo "Spark3: ${worker_shards} shards, ${worker_size}"
[ "${head_shards}" -eq 99 ] && [ "${worker_shards}" -eq 99 ]

cd "${SCRIPT_DIR}"
echo "== launch Hy3 service =="
./start.sh

echo "== verify generation and MTP logs =="
TOKENS=512 ./verify.sh | tee /tmp/hy3-verify.out
metric_line=$(python3 - <<'PY'
import json,re
s=open('/tmp/hy3-verify.out').read()
blocks=re.findall(r'\{\n  "elapsed_sec".*?\n\}', s, re.S)
if blocks:
    m=json.loads(blocks[-1])
    print(f"- Measured {m.get('usage',{}).get('completion_tokens')} completion tokens in {m.get('elapsed_sec')}s = {m.get('approx_completion_tok_per_sec')} tok/s (single stream, verify.sh).")
else:
    print('- verify.sh completed; metric JSON not parsed.')
PY
)
python3 - <<PY
from pathlib import Path
p=Path('${SCRIPT_DIR}/README.md')
s=p.read_text()
old='- Pending final run after the 181GB model download completes on Spark2 and is rsynced to Spark3.\n- Upstream reference for the same recipe: 21.8 tok/s single-stream, 59.7 tok/s aggregate over 6-way concurrency with enforce-eager + MTP spec-1.\n'
new='${metric_line}\n- Upstream reference for the same recipe: 21.8 tok/s single-stream, 59.7 tok/s aggregate over 6-way concurrency with enforce-eager + MTP spec-1.\n'
p.write_text(s.replace(old,new))
PY

echo "== final status =="
./status.sh
