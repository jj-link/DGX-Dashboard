#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B (FP8) + DFlash drafter with SGLang ON THE DGX SPARK
# (GB10 / sm_121a, aarch64). The SGLang counterpart to the local
# serve/sglang/serve_qwen36_27b_fp8_dflash.sh and to the archived vLLM DFlash sweep script.
# Self-contained ("run ON the Spark" convention): scp to the spark and
#   bash serve_sglang_a3b_dflash_remote.sh
# or drive it over ssh. Leaves the server running (detached container).
#
# !! UNVERIFIED ON GB10/sm_121a !! The arm64 lmsysorg/sglang image exists, but its
# flashinfer / fp8-gemm cubins target datacenter Blackwell (sm_100, GB200) -- the
# Spark is sm_121a. The Spark's vLLM stack is a *source build* (PR #40898) for
# exactly this reason; the stock SGLang image may not ship sm_121 kernels. If it
# errors at init, see the FALLBACKS block at the bottom.
#
# WHY SGLANG (vs the vLLM sweep): --speculative-draft-window-size clamps the DFlash
# drafter to its trained ~3-4k range so acceptance doesn't collapse at long context.
# vLLM PR #40898 has --speculative-dflash-draft-window-size now too, but this mirrors
# the local SGLang setup for a like-for-like engine comparison.
set -u

IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.12-cu130}"   # multi-arch; pulls arm64 on the Spark
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"        # block-wise FP8 target (spec-decode validates against it)
DRAFTER="${DRAFTER:-z-lab/Qwen3.6-35B-A3B-DFlash}"
SERVED="${SERVED:-Qwen3.6-35B-A3B}"
PORT="${PORT:-8000}"
NUM_SPEC="${NUM_SPEC:-16}"                          # z-lab SGLang recommendation
DRAFT_WINDOW="${DRAFT_WINDOW:-6144}"               # off|0|none -> full-context drafter (the collapsing baseline)
MEM_FRACTION="${MEM_FRACTION:-0.82}"               # GB10 128GB unified; 0.82 -> ~105GB for weights + 256k KV
MAXLEN="${MAXLEN:-262144}"                          # 256k = native max
ATTN_BACKEND="${ATTN_BACKEND:-flashinfer}"         # sm_121: fa3/fa4/trtllm_mha are sm90/sm100
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"        # single-stream: small graph set
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"  # single-stream: caps mamba+spec state
HF_CACHE="${HF_CACHE:-$HOME/models}"               # spark hub lives at $HOME/models/hub
CACHE_VOL="${CACHE_VOL:-sglang-spark-cache}"       # persist torch.compile/flashinfer/triton JIT across runs
NAME="${NAME:-sglang-a3b-dflash}"

DWIN_ARG=()
case "$DRAFT_WINDOW" in off|0|none) ;; *) DWIN_ARG=(--speculative-draft-window-size "$DRAFT_WINDOW") ;; esac

docker rm -f "$NAME" >/dev/null 2>&1 || true
echo ">>> [spark] SGLang start: model=$MODEL drafter=$DRAFTER spec=$NUM_SPEC window=$DRAFT_WINDOW attn=$ATTN_BACKEND mem=$MEM_FRACTION  $(date +%H:%M:%S)"
docker run -d --name "$NAME" \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HF_CACHE:/models:ro" -e HF_HOME=/models -e HF_HUB_OFFLINE=1 \
  -e HOME=/cache -v "$CACHE_VOL:/cache" \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor -e TRITON_CACHE_DIR=/cache/triton \
  -e SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}" \
  ${SGLANG_ENABLE_DEEP_GEMM:+-e SGLANG_ENABLE_DEEP_GEMM="${SGLANG_ENABLE_DEEP_GEMM}"} \
  --entrypoint python3 "$IMAGE" \
  -m sglang.launch_server \
    --model-path "$MODEL" \
    --served-model-name "$SERVED" \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$DRAFTER" \
    --speculative-num-draft-tokens "$NUM_SPEC" \
    "${DWIN_ARG[@]}" \
    --tp-size 1 \
    --attention-backend "$ATTN_BACKEND" \
    --mem-fraction-static "$MEM_FRACTION" \
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --context-length "$MAXLEN" \
    --mamba-scheduler-strategy extra_buffer \
    --trust-remote-code \
    --enable-metrics \
    --host 0.0.0.0 --port "$PORT" \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder >/dev/null

URL="http://localhost:$PORT"
end=$(( $(date +%s) + 1200 )); ready=0
while [ "$(date +%s)" -lt "$end" ]; do
  curl -fsS --max-time 4 "$URL/v1/models" 2>/dev/null | grep -q "$SERVED" && { ready=1; break; }
  st=$(docker inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null | tr -d '[:space:]')
  { [ "$st" = "exited" ] || [ "$st" = "dead" ]; } && { echo "!! CONTAINER $st -- logs:"; docker logs --tail 80 "$NAME" 2>&1 | tail -80; exit 2; }
  sleep 6
done
[ "$ready" = 1 ] || { echo "!! TIMEOUT waiting for $SERVED -- logs:"; docker logs --tail 80 "$NAME" 2>&1 | tail -80; exit 3; }
echo ">>> [spark] SGLang ready at $URL  $(date +%H:%M:%S)"
curl -fsS "$URL/v1/models" 2>/dev/null | head -c 400; echo

# ---- FALLBACKS if it errors on sm_121 ----
#  flashinfer attention error : ATTN_BACKEND=triton bash serve_sglang_a3b_dflash_remote.sh
#  fp8 gemm / deep_gemm error : SGLANG_ENABLE_DEEP_GEMM=0 bash ...   (and/or --fp8-gemm-backend triton)
#  OOM / pool-init "not enough memory" : MEM_FRACTION=0.65 bash ...
#  no sm_121 kernels at all   : needs a source-built SGLang for sm_121a (like the vLLM pr40898 image)
