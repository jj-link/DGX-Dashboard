#!/usr/bin/env bash
# Swap the local vLLM (RTX PRO 6000) to a target model — DOCKER-BASED,
# mirroring spark-serve/swap.sh. The prior bare-process version was wrong:
# `vllm serve` bound to host 0.0.0.0 is unreachable via localhost on WSL2;
# docker with `-p 127.0.0.1:8000:8000` publishes the port and works.
#
# Container per alias (vllm-serve_<alias_with_underscores>) so naming matches
# the user's existing manual `docker run ... vllm-serve_<variant>` pattern.
# Still snapshots the startup log to /tmp/local-vllm-logs/<alias>-<ts>.log so
# sweep_aider_local.sh can grep "Maximum concurrency..." from it.
#
# Usage: ./swap.sh <alias>   (aliases per local-serve/models.env)
set -euo pipefail

ALIAS="${1:?alias required}"
LOCAL_URL="${LOCAL_URL:-http://localhost:8000/v1}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
IMAGE="${IMAGE:-qwen-vllm:pinned}"
HF_CACHE_HOST="${HF_CACHE_HOST:-/home/workbench/.cache/huggingface}"
HF_CACHE_MODE="${HF_CACHE_MODE:-ro}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
WORKSPACE_HOST="${WORKSPACE_HOST:-/home/workbench/Projects/personal/test}"
RUNNER="${WORKSPACE_HOST}/run_vllm_docker.sh"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
# SSM-state cache precision. Native is float32; override MAMBA_DTYPE=float16
# only for throughput sweeps / A/B (matches spark default).
MAMBA_DTYPE="${MAMBA_DTYPE:-float32}"
LOG_DIR="${LOG_DIR:-/tmp/local-vllm-logs}"
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=models.env
source "${SCRIPT_DIR}/models.env"
select_local_model "$ALIAS"

CNAME="vllm-serve_${ALIAS//-/_}"
LOG="${LOG_DIR}/${ALIAS}-$(date +%Y%m%d-%H%M%S).log"
echo "[lswap] target=$ALIAS model=$LSERVE_MODEL served=$LSERVE_NAME script=$LSERVE_SCRIPT container=$CNAME"

# Fast path: target already serving — no-op.
LIVE=$(curl -fsS --max-time 5 "$LOCAL_URL/models" 2>/dev/null \
        | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4 || true)
if [ "$LIVE" = "$LSERVE_NAME" ]; then
  echo "[lswap] already serving — no-op"
  docker logs "$CNAME" > "$LOG" 2>&1 || true
  exit 0
fi

# Tear down any existing local vLLM container (single-name or per-alias).
EXISTING=$(docker ps -aq --filter "name=^vllm-serve_" --filter "name=^vllm$" 2>/dev/null | sort -u)
if [ -n "$EXISTING" ]; then
  echo "[lswap] removing existing container(s): $(echo $EXISTING | tr '\n' ' ')"
  docker rm -f $EXISTING >/dev/null 2>&1 || true
fi
# Also reap any stray bare-process vllm from the OLD swap.sh era.
pkill -9 -f 'vllm serve' 2>/dev/null || true
pkill -9 -f 'VLLM::EngineCore' 2>/dev/null || true
# Wait for the port to actually free.
for i in {1..20}; do
  python3 -c "import socket;s=socket.socket()
try: s.bind(('0.0.0.0',${LOCAL_PORT})); exit(0)
except OSError: exit(1)" 2>/dev/null && break
  sleep 1
done

echo "[lswap] log: $LOG"
echo "[lswap] launching docker container..."

DETACH=1 \
CONTAINER_NAME="$CNAME" \
IMAGE="$IMAGE" \
PORT="$LOCAL_PORT" \
HF_CACHE="$HF_CACHE_HOST" \
HF_CACHE_MODE="$HF_CACHE_MODE" \
HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
MODEL="$LSERVE_MODEL" \
SERVED="$LSERVE_NAME" \
MAXLEN="$LSERVE_MAX_LEN" \
TOKENIZER="$LSERVE_TOKENIZER" \
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-$LSERVE_KV_CACHE}" \
MAMBA_DTYPE="$MAMBA_DTYPE" \
MODELS_HOST="${MODELS_HOST:-/home/workbench/models}" \
  "$RUNNER" "$LSERVE_SCRIPT" >/dev/null

echo "[lswap] container started; waiting for $LOCAL_URL/models (up to ${READY_TIMEOUT}s)..."
START=$(date +%s)
while :; do
  if curl -fsS --max-time 5 "$LOCAL_URL/models" 2>/dev/null | grep -q "\"$LSERVE_NAME\""; then
    echo "[lswap] ready: $LSERVE_NAME (container=$CNAME)"
    # Snapshot the startup log to disk so sweep_aider_local.sh can grep
    # "Maximum concurrency..." from it (matches the bare-process layout).
    docker logs "$CNAME" > "$LOG" 2>&1 || true
    exit 0
  fi
  if ! docker ps --filter "name=^${CNAME}$" --format '{{.Names}}' | grep -q "$CNAME"; then
    echo "[lswap] container exited unexpectedly; last logs:" >&2
    docker logs --tail 60 "$CNAME" 2>&1 | tee "$LOG" >&2 || true
    exit 1
  fi
  NOW=$(date +%s)
  if (( NOW - START > READY_TIMEOUT )); then
    echo "[lswap] TIMEOUT after ${READY_TIMEOUT}s; last logs:" >&2
    docker logs --tail 60 "$CNAME" 2>&1 | tee "$LOG" >&2 || true
    exit 1
  fi
  sleep 5
done
