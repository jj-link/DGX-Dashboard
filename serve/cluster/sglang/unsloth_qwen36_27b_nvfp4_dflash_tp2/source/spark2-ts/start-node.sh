#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
NODE_RANK="${1:?usage: start-node.sh NODE_RANK HOST_IP}"
HOST_IP="${2:?usage: start-node.sh NODE_RANK HOST_IP}"
if [[ "$NODE_RANK" != "0" && "$NODE_RANK" != "1" ]]; then
  echo "NODE_RANK must be 0 or 1" >&2
  exit 2
fi

# Remove only this recipe's previous container; never touch unrelated servers here.
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

GRAPH_ARGS=()
if [[ "$DISABLE_CUDA_GRAPH" == "1" ]]; then
  GRAPH_ARGS+=(--disable-cuda-graph)
else
  GRAPH_ARGS+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
fi
GRAPH_ARGS_STR=""
if (( ${#GRAPH_ARGS[@]} > 0 )); then
  printf -v GRAPH_ARGS_STR ' %q' "${GRAPH_ARGS[@]}"
fi

DFLASH_ARGS=()
if [[ "$ENABLE_DFLASH" == "1" ]]; then
  echo "[mode] SGLang DFlash enabled: $DRAFT_MODEL_PATH" >&2
  DFLASH_ARGS+=(
    --speculative-algorithm "$SPECULATIVE_ALGORITHM"
    --speculative-draft-model-path "$DRAFT_MODEL_PATH"
    --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS"
    --speculative-draft-window-size "$SPECULATIVE_DRAFT_WINDOW_SIZE"
    --mamba-scheduler-strategy "$MAMBA_SCHEDULER_STRATEGY"
  )
else
  echo "[mode] NON-DFLASH fallback: DFlash long generation deadlocked in TP=2; see DFLASH-INCOMPATIBILITY.md" >&2
fi
DFLASH_ARGS_STR=""
if (( ${#DFLASH_ARGS[@]} > 0 )); then
  printf -v DFLASH_ARGS_STR ' %q' "${DFLASH_ARGS[@]}"
fi

exec docker run -d \
  --name "$CONTAINER" \
  --entrypoint /bin/bash \
  --privileged \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device=/dev/infiniband:/dev/infiniband \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NCCL_NET="$NCCL_NET" \
  -e NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  -e NCCL_IB_HCA="$NCCL_IB_HCA" \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  -e GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
  -e TP_SOCKET_IFNAME="$TP_SOCKET_IFNAME" \
  -e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX" \
  -e NCCL_CROSS_NIC="$NCCL_CROSS_NIC" \
  -e NCCL_DEBUG="$NCCL_DEBUG" \
  -e NCCL_IGNORE_CPU_AFFINITY="$NCCL_IGNORE_CPU_AFFINITY" \
  -e HF_HOME="$HF_HOME" \
  -e HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  -e TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
  -e HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
  -e SGLANG_ENABLE_SPEC_V2="$SGLANG_ENABLE_SPEC_V2" \
  -e SGLANG_ENABLE_JIT_DEEPGEMM="$SGLANG_ENABLE_JIT_DEEPGEMM" \
  -v "$HOST_HF_ROOT:/models/hub:ro" \
  -v "$RECIPE_DIR:/workspace/recipe:ro" \
  "$IMAGE" -lc "exec python3 -m sglang.launch_server \
    --model-path '$TARGET_MODEL_PATH' \
    --host 0.0.0.0 \
    --port '$SGLANG_PORT' \
    --served-model-name '$SERVED_MODEL_NAME' \
    --trust-remote-code \
    --tp-size '$TP_SIZE' \
    --nnodes '$NNODES' \
    --node-rank '$NODE_RANK' \
    --dist-init-addr '$DIST_INIT_ADDR' \
    --attention-backend flashinfer \
    --context-length '$CONTEXT_LENGTH' \
    $DFLASH_ARGS_STR \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --mem-fraction-static '$MEM_FRACTION_STATIC' \
    --max-running-requests '$MAX_RUNNING_REQUESTS' \
    --enable-metrics \
    $GRAPH_ARGS_STR \
    --chunked-prefill-size '$CHUNKED_PREFILL_SIZE'"
