#!/usr/bin/env bash
# vLLM launch for melcheikh/gemma-4-31B-it-qat-NVFP4-Blackwell on RTX PRO 6000.
# Smoke test first; speculative assistant can be layered on after baseline works.
set -u

MODEL="${MODEL:-melcheikh/gemma-4-31B-it-qat-NVFP4-Blackwell}"
SERVED="${SERVED:-melcheikh/gemma-4-31B-it-qat-NVFP4-Blackwell}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAXLEN="${MAXLEN:-65536}"
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton_attn}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
SPEC_ARGS=()
if [ -n "$SPECULATIVE_CONFIG" ]; then
  SPEC_ARGS=(--speculative-config "$SPECULATIVE_CONFIG")
fi

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED" \
  --quantization modelopt \
  --dtype auto \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --attention-backend "$ATTENTION_BACKEND" \
  "${SPEC_ARGS[@]}" \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code
