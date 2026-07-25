#!/usr/bin/env bash
# ThinkingCap Qwen3.6-27B NVFP4 + DFlash on one DGX Spark GB10.
set -euo pipefail

MODEL_REVISION="656627c8f7ea4785413ab1e06f6ccd20bba6622f"
DRAFTER_REVISION="0919688658996800f86b895034249700e9481106"
MODEL="${MODEL:-/models/serving/thinkingcap-qwen36-27b-nvfp4-dflash-${MODEL_REVISION}}"
DRAFTER="${DRAFTER:-/models/hub/models--z-lab--Qwen3.6-27B-DFlash/snapshots/${DRAFTER_REVISION}}"
SERVED="${SERVED:-thinkingcap-qwen36-27b-nvfp4-dflash}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-262144}"
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
NUM_SPEC="${NUM_SPEC:-12}"
MM_LIMITS="${MM_LIMITS:-{\"image\":0,\"video\":0}}"

[ -f "${MODEL}/config.json" ] || {
  echo "Missing prepared model overlay: ${MODEL}" >&2
  echo "Run: python3 serve/vllm/prepare_thinkingcap_qwen36_27b_nvfp4_dflash.py" >&2
  exit 1
}
[ -f "${DRAFTER}/config.json" ] || {
  echo "Missing pinned DFlash drafter: ${DRAFTER}" >&2
  exit 1
}

python3 - "${MODEL}/config.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
quantization = config.get("quantization_config", {})
if "kv_cache_scheme" in quantization:
    raise SystemExit("DFlash requires BF16 KV; remove quantization_config.kv_cache_scheme")
PY

if [[ -z "${SPECULATIVE_CONFIG:-}" ]]; then
  SPECULATIVE_CONFIG="{\"method\":\"dflash\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":${NUM_SPEC}}"
fi

# vLLM issue #44006: speculative draft tokens can violate xgrammar's strict
# structural-tag FSM. qwen3_xml still parses tool calls with strict mode off.
export VLLM_ENFORCE_STRICT_TOOL_CALLING=0
# The patched image has FlashInfer Python 0.6.8 over NVIDIA's 0.6.7 AOT
# cache. This service uses flash_attn; allow FlashInfer's documented bypass.
export FLASHINFER_DISABLE_VERSION_CHECK=1

exec vllm serve "${MODEL}" \
  --served-model-name "${SERVED}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len "${MAXLEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --kv-cache-dtype auto \
  --attention-backend flash_attn \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt "${MM_LIMITS}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config "${SPECULATIVE_CONFIG}" \
  --trust-remote-code
