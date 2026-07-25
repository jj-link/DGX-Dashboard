#!/usr/bin/env bash
# SGLang launch for Gemma-4-12B-IT AWQ-INT4 on RTX 4090 (GPU 1)
# Auxiliary model for vision, compression, web extraction tasks.
# Run via the hardened wrapper:
#   CUDA_VISIBLE_DEVICES=1 PORT=8001 ./run_sglang_docker.sh serve/sglang/serve_gemma4_12b_awq.sh
#
# NOTE: SGLang v0.5.12 ships with Transformers 5.6.0 which does not recognize
# gemma4_unified. We force an upgrade to 5.12.0 at startup. The pip warning
# about version conflict is harmless — SGLang works fine with 5.12.x.
#
# VRAM math (AWQ INT4 weights + FP8 KV cache):
#   - 12B AWQ INT4 weights: ~6-7 GB
#   - FP8 KV cache at 32k context: ~4-5 GB (vs ~8-10 GB with BF16)
#   - CUDA graphs (bs=8): ~1 GB
#   - Total at rest: ~12 GB / 24 GB -> plenty of headroom
set -u

MODEL="${MODEL:-cyankiwi/gemma-4-12B-it-AWQ-INT4}"
SERVED="${SERVED:-gemma-4-12b-it-awq}"
MAXLEN="${MAXLEN:-32768}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-65536}"
PORT="${PORT:-8001}"

# Upgrade Transformers to 5.12.0 (supports gemma4_unified)
echo "Upgrading Transformers to support gemma4_unified..."
pip install --upgrade --no-deps --break-system-packages transformers 2>&1 | tail -3
TRANSFORMERS_VERSION=$(python3 -c "import transformers; print(transformers.__version__)")
echo "Transformers version: $TRANSFORMERS_VERSION"

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED" \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8_e4m3 \
  --tp-size 1 \
  --attention-backend flashinfer \
  --mem-fraction-static "$MEM_FRACTION" \
  --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
  --context-length "$MAXLEN" \
  --trust-remote-code \
  --enable-metrics \
  --host 0.0.0.0 --port "$PORT"
