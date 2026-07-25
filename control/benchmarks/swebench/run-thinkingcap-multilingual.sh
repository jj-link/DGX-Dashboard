#!/usr/bin/env bash
# Pinned mini-SWE-agent v2 run against SWE-bench Multilingual.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MINI_EXTRA="${ROOT}/.venv/bin/mini-extra"
PYTHON="${ROOT}/.venv/bin/python"
CONFIG="${ROOT}/thinkingcap-swebench.yaml"
DATASET="/home/workbench/.cache/huggingface/hub/datasets--SWE-bench--SWE-bench_Multilingual/snapshots/2b7aced941b4873e9cad3e76abbae93f481d1beb"
OUTPUT="${OUTPUT:-${ROOT}/runs/thinkingcap-qwen36-27b-nvfp4-mswea-v2.4.5}"
WORKERS="${WORKERS:-6}"
GRADE="${GRADE:-0}"
GRADE_WORKERS="${GRADE_WORKERS:-1}"
RUN_ID="${RUN_ID:-thinkingcap-dflash-$(date -u +%Y%m%d-%H%M%S)}"
DATASET_JSON="${ROOT}/.cache/swebench-multilingual-2b7aced941b4873e9cad3e76abbae93f481d1beb.json"

[ -x "${MINI_EXTRA}" ] || { echo "Missing mini-SWE-agent v2.4.5 environment" >&2; exit 1; }
[ -f "${CONFIG}" ] || { echo "Missing benchmark config: ${CONFIG}" >&2; exit 1; }
[ -f "${DATASET}/data/test-00000-of-00001.parquet" ] || {
  echo "Missing pinned SWE-bench Multilingual dataset revision" >&2
  exit 1
}

"${MINI_EXTRA}" swebench \
  --config "${CONFIG}" \
  --subset "${DATASET}" \
  --split test \
  --output "${OUTPUT}" \
  --workers "${WORKERS}" \
  "$@"

if [[ "${GRADE}" != "1" ]]; then
  exit 0
fi

[ -f "${OUTPUT}/preds.json" ] || {
  echo "Missing predictions: ${OUTPUT}/preds.json" >&2
  exit 1
}

mkdir -p "$(dirname "${DATASET_JSON}")"
if [[ ! -f "${DATASET_JSON}" ]]; then
  "${PYTHON}" - "${DATASET}" "${DATASET_JSON}" <<'PY'
import json
import sys

from datasets import load_dataset

dataset = load_dataset(sys.argv[1], split="test")
with open(sys.argv[2], "w", encoding="utf-8") as output:
    json.dump([dict(instance) for instance in dataset], output)
PY
fi

OUTPUT="$(cd "${OUTPUT}" && pwd)"
cd "${OUTPUT}"
exec "${PYTHON}" -m swebench.harness.run_evaluation \
  --dataset_name "${DATASET_JSON}" \
  --predictions_path "${OUTPUT}/preds.json" \
  --max_workers "${GRADE_WORKERS}" \
  --run_id "${RUN_ID}"
