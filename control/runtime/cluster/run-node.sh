#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

[[ $# -eq 4 ]] || fail "usage: run-node.sh <action> <engine> <artifact> <rank>"
ACTION="$1"
ENGINE="$2"
ARTIFACT="$3"
RANK="$4"
[[ "$ACTION" =~ ^(preflight|start|wait-rank|status|logs|verify|stop|port-clear|api-host)$ ]] || fail "invalid node action '$ACTION'"
[[ "$ENGINE" == vllm || "$ENGINE" == sglang ]] || fail "invalid engine '$ENGINE'"
[[ "$ARTIFACT" =~ ^[a-z0-9][a-z0-9_]*$ ]] || fail "invalid artifact '$ARTIFACT'"
[[ "$RANK" == 0 || "$RANK" == 1 ]] || fail "rank must be 0 or 1"
PREFLIGHT_REPLACE_CONTAINER="${PREFLIGHT_REPLACE_CONTAINER:-}"
PREFLIGHT_REPLACE_IMAGE_ID="${PREFLIGHT_REPLACE_IMAGE_ID:-}"
PREFLIGHT_REPLACE_NETWORK_MODE="${PREFLIGHT_REPLACE_NETWORK_MODE:-}"
if [[ -n "$PREFLIGHT_REPLACE_CONTAINER$PREFLIGHT_REPLACE_IMAGE_ID$PREFLIGHT_REPLACE_NETWORK_MODE" ]]; then
  [[ "$ACTION" == preflight ]] || fail "replacement allowances apply only to preflight"
  [[ -n "$PREFLIGHT_REPLACE_CONTAINER" && -n "$PREFLIGHT_REPLACE_IMAGE_ID" && -n "$PREFLIGHT_REPLACE_NETWORK_MODE" ]] || fail "all PREFLIGHT_REPLACE_* values are required"
  [[ "$PREFLIGHT_REPLACE_CONTAINER" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "PREFLIGHT_REPLACE_CONTAINER is invalid"
  [[ "$PREFLIGHT_REPLACE_IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "PREFLIGHT_REPLACE_IMAGE_ID must be an immutable Docker image ID"
  [[ "$PREFLIGHT_REPLACE_NETWORK_MODE" =~ ^[A-Za-z0-9_.:-]+$ ]] || fail "PREFLIGHT_REPLACE_NETWORK_MODE is invalid"
fi

REPO_ROOT="${DGX_DASHBOARD_ROOT:-/home/jjlink/dgx-dashboard}"
CONTROL_ROOT="$REPO_ROOT/control"
PACKAGE="$CONTROL_ROOT/serve/cluster/$ENGINE/$ARTIFACT"
PARSER="$CONTROL_ROOT/tools/parse-runtime-env.py"
VALIDATOR="$CONTROL_ROOT/tools/validate-hf-snapshot.py"
[[ -f "$PACKAGE/runtime.env" ]] || fail "missing cluster metadata '$PACKAGE/runtime.env'"
[[ -x "$PARSER" ]] || fail "missing metadata parser '$PARSER'"
CAPABILITIES_PATH="$PACKAGE/capabilities.json"

declare -A META=()
coproc METADATA_PARSER { python3 "$PARSER" "$PACKAGE/runtime.env"; }
parser_pid=$METADATA_PARSER_PID
mapfile -d '' -t parsed <&"${METADATA_PARSER[0]}"
wait "$parser_pid" || exit $?
(( ${#parsed[@]} == 24 )) || fail "runtime metadata parser returned ${#parsed[@]} fields; expected 24"
for ((index = 0; index < ${#parsed[@]}; index += 2)); do
  META["${parsed[index]}"]="${parsed[index + 1]}"
done

IMAGE="${META[IMAGE]}"
SERVED="${META[SERVED]}"
CONTAINER_BASE="${META[CONTAINER_NAME]}"
CONTAINER="${CONTAINER_BASE}-rank${RANK}"
API_PORT=8888
DIST_IF=enp1s0f1np1
RDMA_HCA=rocep1s0f1
MASTER_ADDR=10.0.0.1
MASTER_PORT=25000
NODE_ADDR="10.0.0.$((RANK + 1))"
WORLD_SIZE=2

case "$ENGINE/$ARTIFACT" in
  sglang/unsloth_qwen36_27b_nvfp4_dflash_tp2)
    MAX_MODEL_LEN=262144
    MASTER_PORT=25001
    ;;
  vllm/deepseek_ai_deepseek_v4_flash_dspark_tp2|\
  vllm/drowzeys_keys_deepseek_v4_flash_dspark_abliterated_32_32)
    [[ "$IMAGE" == "ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8" ]] ||
      fail "official DeepSeek V4 Flash 0731 requires the audited DSpark shared-expert loader image"
    PROFILE_ROOT="$PACKAGE/profiles"
    DEFAULT_PROFILE_FILE="$PROFILE_ROOT/default"
    [[ -f "$DEFAULT_PROFILE_FILE" && ! -L "$DEFAULT_PROFILE_FILE" ]] ||
      fail "missing DeepSeek default launch profile"
    mapfile -t default_profile_lines <"$DEFAULT_PROFILE_FILE"
    (( ${#default_profile_lines[@]} == 1 )) || fail "invalid DeepSeek default launch profile"
    PROFILE="${CLUSTER_PROFILE:-${default_profile_lines[0]}}"
    [[ "$PROFILE" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || fail "invalid DeepSeek cluster profile '$PROFILE'"
    PROFILE_FILE="$PROFILE_ROOT/$PROFILE.env"
    [[ -f "$PROFILE_FILE" && ! -L "$PROFILE_FILE" ]] || fail "unknown DeepSeek cluster profile '$PROFILE'"
    # Profiles are tracked, secret-free static assignments in this exact checkout.
    set -a
    # shellcheck disable=SC1090
    source "$PROFILE_FILE"
    set +a
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
    [[ -f "$CAPABILITIES_PATH" && ! -L "$CAPABILITIES_PATH" ]] ||
      fail "missing DeepSeek model capability profile"
    CAPABILITIES_SHA256="$(sha256sum "$CAPABILITIES_PATH" | cut -d' ' -f1)"
    ;;
  *) fail "unsupported cluster artifact '$ENGINE/$ARTIFACT'" ;;
esac

resolve_nccl_gid_index() {
  command -v show_gids >/dev/null 2>&1 || fail "show_gids is unavailable"
  python3 - "$RDMA_HCA" "$NODE_ADDR" <<'PY'
import subprocess
import sys

hca, node_addr = sys.argv[1:]
output = subprocess.run(
    ["show_gids"],
    check=True,
    capture_output=True,
    text=True,
).stdout
for line in output.splitlines():
    fields = line.split()
    if (
        len(fields) >= 7
        and fields[0] == hca
        and fields[4] == node_addr
        and fields[5] == "v2"
    ):
        print(fields[2])
        break
else:
    raise SystemExit(
        f"no RoCE v2 GID for {hca} and node address {node_addr}"
    )
PY
}

repo_directory() {
  printf 'models--%s\n' "${1//\//--}"
}

resolve_repository() {
  local label="$1" repository="$2" revision="$3" variable_prefix="$4"
  local repository_directory candidate root snapshot
  local resolved_host='' resolved_container='' validation_error='' validation_output
  repository_directory="$(repo_directory "$repository")"
  local -a roots=()
  if [[ -n "${HF_CACHE:-}" ]]; then
    roots=("$HF_CACHE")
  else
    roots=("$HOME/models" /models/hub "$HOME/hf-cache" "$HOME/.cache/huggingface")
  fi
  for root in "${roots[@]}"; do
    for candidate in "$root/hub/$repository_directory" "$root/$repository_directory"; do
      snapshot="$candidate/snapshots/$revision"
      if [[ -d "$snapshot" ]]; then
        if validation_output="$(python3 "$VALIDATOR" "$snapshot" 2>&1)"; then
          resolved_host="$(cd "$candidate" && pwd -P)"
          resolved_container="/models/hub/$repository_directory"
          break 2
        elif [[ -z "$validation_error" ]]; then
          validation_error="${validation_output#error: }"
        fi
      fi
    done
  done
  if [[ -z "$resolved_host" ]]; then
    if [[ -n "$validation_error" ]]; then
      fail "$label '$repository@$revision' has an incomplete cache snapshot: $validation_error"
    fi
    fail "$label '$repository@$revision' is not installed in any configured cache root"
  fi
  printf -v "${variable_prefix}_REPO_HOST" '%s' "$resolved_host"
  printf -v "${variable_prefix}_REPO_CONTAINER" '%s' "$resolved_container"
  printf -v "${variable_prefix}_PATH" '%s' "$resolved_container/snapshots/$revision"
}

resolve_artifacts() {
  if [[ -n "${META[MODEL_HOST_PATH]}" ]]; then
    [[ -e "${META[MODEL_HOST_PATH]}" ]] || fail "model local path '${META[MODEL_HOST_PATH]}' does not exist"
    MODEL_REPO_HOST="${META[MODEL_HOST_PATH]}"
    MODEL_REPO_CONTAINER="${META[MODEL]}"
    MODEL_PATH="${META[MODEL]}"
  else
    resolve_repository model "${META[MODEL]}" "${META[MODEL_REVISION]}" MODEL
  fi
  DRAFTER_REPO_HOST=''
  DRAFTER_REPO_CONTAINER=''
  DRAFTER_PATH=''
  if [[ -n "${META[DRAFTER]}" ]]; then
    if [[ -n "${META[DRAFTER_HOST_PATH]}" ]]; then
      [[ -e "${META[DRAFTER_HOST_PATH]}" ]] || fail "drafter local path '${META[DRAFTER_HOST_PATH]}' does not exist"
      DRAFTER_REPO_HOST="${META[DRAFTER_HOST_PATH]}"
      DRAFTER_REPO_CONTAINER="${META[DRAFTER]}"
      DRAFTER_PATH="${META[DRAFTER]}"
    else
      resolve_repository drafter "${META[DRAFTER]}" "${META[DRAFTER_REVISION]}" DRAFTER
    fi
  fi
}

single_tailscale_ipv4() {
  command -v tailscale >/dev/null 2>&1 || fail "tailscale is unavailable; refusing to expose the cluster API"
  mapfile -t addresses < <(tailscale ip -4 2>/dev/null | sed '/^[[:space:]]*$/d')
  (( ${#addresses[@]} == 1 )) || fail "expected exactly one Tailscale IPv4, found ${#addresses[@]}; refusing to expose the cluster API"
  [[ "${addresses[0]}" =~ ^100\. ]] || fail "unexpected Tailscale IPv4 '${addresses[0]}'; refusing to expose the cluster API"
  printf '%s\n' "${addresses[0]}"
}

port_clear() {
  if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :$API_PORT" | grep -q .; then
    return 1
  fi
  return 0
}

verify_replacement_container() {
  python3 - "$PREFLIGHT_REPLACE_CONTAINER" "$PREFLIGHT_REPLACE_IMAGE_ID" "$PREFLIGHT_REPLACE_NETWORK_MODE" <<'PY'
import json
import subprocess
import sys

name, expected_image, expected_network_mode = sys.argv[1:]
completed = subprocess.run(
    ["docker", "container", "inspect", name],
    text=True,
    capture_output=True,
)
if completed.returncode:
    raise SystemExit(f"replacement container {name!r} is unavailable")
rows = json.loads(completed.stdout)
if len(rows) != 1:
    raise SystemExit(f"expected exactly one replacement container named {name!r}")
container = rows[0]
if not container.get("State", {}).get("Running"):
    raise SystemExit(f"replacement container {name!r} is not running")
if container.get("Image") != expected_image:
    raise SystemExit(
        f"replacement container {name!r} image is {container.get('Image')!r}; "
        f"expected {expected_image!r}"
    )
network_mode = container.get("HostConfig", {}).get("NetworkMode")
if network_mode != expected_network_mode:
    raise SystemExit(
        f"replacement container {name!r} network mode is {network_mode!r}; "
        f"expected {expected_network_mode!r}"
    )
PY
}

preflight() {
  command -v docker >/dev/null 2>&1 || fail "docker is unavailable"
  command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable"
  command -v ip >/dev/null 2>&1 || fail "ip is unavailable"
  command -v ibv_devinfo >/dev/null 2>&1 || fail "ibv_devinfo is unavailable"
  [[ -d "$REPO_ROOT/.git" ]] || fail "'$REPO_ROOT' is not a Git checkout"
  git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null || fail "'$REPO_ROOT' is detached"
  [[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]] || fail "'$REPO_ROOT' is dirty"
  local expected_commit="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
  [[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$expected_commit" ]] || fail "'$REPO_ROOT' is not at controller commit '$expected_commit'"
  mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed '/^[[:space:]]*$/d')
  (( ${#gpu_names[@]} == 1 )) || fail "expected exactly one GPU, found ${#gpu_names[@]}"
  [[ "${gpu_names[0]}" == *"GB10"* || "${gpu_names[0]}" == *"DGX Spark"* ]] || fail "expected one GB10/DGX Spark GPU, found '${gpu_names[0]}'"
  ip link show dev "$DIST_IF" | grep -q 'state UP' || fail "$DIST_IF is not UP"
  ip -4 -o addr show dev "$DIST_IF" | grep -Eq "[[:space:]]inet[[:space:]]+$NODE_ADDR/" || fail "$DIST_IF does not own $NODE_ADDR"
  ibv_devinfo -d "$RDMA_HCA" >/dev/null 2>&1 || fail "$RDMA_HCA is not an RDMA device"
  resolve_nccl_gid_index >/dev/null
  docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "image '$IMAGE' is unavailable"
  if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    [[ -n "$PREFLIGHT_REPLACE_CONTAINER" ]] || fail "container '$CONTAINER' already exists"
    [[ "$PREFLIGHT_REPLACE_CONTAINER" == "$CONTAINER" ]] ||
      fail "target container '$CONTAINER' exists but replacement '$PREFLIGHT_REPLACE_CONTAINER' was requested"
  fi
  if [[ -n "$PREFLIGHT_REPLACE_CONTAINER" ]]; then
    verify_replacement_container
    if [[ "$RANK" == 0 ]]; then
      ! port_clear || fail "recorded replacement container does not own port '$API_PORT'"
    else
      port_clear || fail "worker port '$API_PORT' is already in use"
    fi
  else
    port_clear || fail "port '$API_PORT' is already in use"
  fi
  resolve_artifacts
  local api_host=''
  if [[ "$RANK" == 0 ]]; then
    api_host="$(single_tailscale_ipv4)"
  fi
  printf 'HOST=%s\n' "$(hostname)"
  printf 'RANK=%s\n' "$RANK"
  printf 'MODEL_REPO_HOST=%s\n' "$MODEL_REPO_HOST"
  printf 'NCCL_IB_GID_INDEX=%s\n' "$(resolve_nccl_gid_index)"
  [[ -z "$DRAFTER_REPO_HOST" ]] || printf 'DRAFTER_REPO_HOST=%s\n' "$DRAFTER_REPO_HOST"
  [[ -z "$api_host" ]] || printf 'API_HOST=%s\n' "$api_host"
}

start_node() {
  local api_host
  resolve_artifacts
  if [[ "$RANK" == 0 ]]; then
    api_host="$(single_tailscale_ipv4)"
  else
    api_host=127.0.0.1
  fi
  local nccl_gid_index
  nccl_gid_index="$(resolve_nccl_gid_index)"

  local -a mounts=(
    -v "$MODEL_REPO_HOST:$MODEL_REPO_CONTAINER:ro"
    -v "$REPO_ROOT:$REPO_ROOT:ro"
    -v "${CONTAINER_BASE}-rank${RANK}-cache:/root/.cache:rw"
  )
  if [[ -n "$DRAFTER_REPO_HOST" ]]; then
    mounts+=( -v "$DRAFTER_REPO_HOST:$DRAFTER_REPO_CONTAINER:ro" )
  fi

  local -a environment=(
    -e "HOME=/root/.cache"
    -e "CUDA_VISIBLE_DEVICES=0"
    -e "MODEL_PATH=$MODEL_PATH"
    -e "DRAFTER_PATH=$DRAFTER_PATH"
    -e "SERVED=$SERVED"
    -e "NODE_RANK=$RANK"
    -e "WORLD_SIZE=$WORLD_SIZE"
    -e "MASTER_ADDR=$MASTER_ADDR"
    -e "MASTER_PORT=$MASTER_PORT"
    -e "VLLM_HOST_IP=$NODE_ADDR"
    -e "NCCL_NET=IB"
    -e "NCCL_IB_DISABLE=0"
    -e "NCCL_IB_HCA=$RDMA_HCA"
    -e "NCCL_SOCKET_IFNAME=$DIST_IF"
    -e "GLOO_SOCKET_IFNAME=$DIST_IF"
    -e "TP_SOCKET_IFNAME=$DIST_IF"
    -e "NCCL_IB_GID_INDEX=$nccl_gid_index"
    -e "NCCL_IB_ADDR_FAMILY=AF_INET"
    -e "NCCL_IB_ROCE_VERSION_NUM=2"
    -e "NCCL_CROSS_NIC=1"
    -e "NCCL_CUMEM_ENABLE=0"
    -e "NCCL_IGNORE_CPU_AFFINITY=1"
    -e "NCCL_DEBUG=INFO"
    -e "NCCL_NVLS_ENABLE=0"
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    -e "HF_HOME=/models/hub"
    -e "HF_HUB_OFFLINE=1"
    -e "TRANSFORMERS_OFFLINE=1"
    -e "HF_HUB_DISABLE_XET=1"
    -e "VLLM_CACHE_ROOT=/root/.cache/vllm-cache"
    -e "API_HOST=$api_host"
    -e "API_PORT=$API_PORT"
    -e "MAX_MODEL_LEN=$MAX_MODEL_LEN"
  )

  if [[ "$ENGINE" == vllm ]]; then
    environment+=(
      -e "CLUSTER_PROFILE=$PROFILE"
      -e "DGX_MODEL_CAPABILITIES_PATH=$CAPABILITIES_PATH"
      -e "DGX_MODEL_CAPABILITIES_SHA256=$CAPABILITIES_SHA256"
      -e "PYTHONPATH=$CONTROL_ROOT/runtime"
    )
    local -a profile_environment=(
      KV_CACHE_DTYPE MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION MTP_NUM_TOKENS LONG_PREFILL_TOKEN_THRESHOLD DECODE_AWARE_PREFILL_INTERVAL DECODE_AWARE_THROTTLED_TOKENS
      VLLM_USE_FLASHINFER_SAMPLER VLLM_USE_B12X_MOE VLLM_USE_B12X_WO_PROJECTION
      VLLM_DSPARK_CONFIDENCE_THRESHOLD VLLM_DSPARK_CONFIDENCE_SCHEDULER VLLM_DSPARK_LOCAL_ARGMAX
      VLLM_DSPARK_REPLICATE_MARKOV_W1 VLLM_DSPARK_FUSED_MARKOV_ARGMAX
      VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT DSPARK_SLOT_CLAMP
      VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM
      VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M VLLM_B12X_W4A16_FORCE_TILE_CONFIG B12X_W4A16_TC_DECODE
      VLLM_DSV4_B12X_COMPRESSED_MLA VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE
      VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT VLLM_ALLOW_LONG_MAX_MODEL_LEN
      VLLM_TRITON_MLA_SPARSE VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
      VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS TORCH_CUDA_ARCH_LIST FLASHINFER_CUDA_ARCH_LIST
      FLASHINFER_DISABLE_VERSION_CHECK TILELANG_CLEANUP_TEMP_FILES DG_JIT_USE_NVRTC DG_JIT_NVCC_COMPILER
      VLLM_SKIP_INIT_MEMORY_CHECK
    )
    local variable
    for variable in "${profile_environment[@]}"; do
      [[ -v "$variable" ]] && environment+=( -e "$variable=${!variable}" )
    done
  else
    environment+=(
      -e "SGLANG_ENABLE_SPEC_V2=1"
      -e "SGLANG_ENABLE_JIT_DEEPGEMM=0"
    )
  fi

  docker run -d \
    --name "$CONTAINER" \
    --hostname "inference-${ENGINE}-rank${RANK}" \
    --gpus all \
    --network host \
    --ipc host \
    --shm-size 32g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --device /dev/infiniband:/dev/infiniband \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    --read-only \
    --pids-limit 4096 \
    --tmpfs /tmp:rw,exec,nosuid,nodev,size=16g \
    --tmpfs /run:rw,nosuid,nodev,size=64m \
    "${mounts[@]}" \
    "${environment[@]}" \
    --entrypoint /bin/bash \
    "$IMAGE" \
    "$CONTROL_ROOT/runtime/cluster/serve-node.sh" "$ENGINE" "$ARTIFACT" >/dev/null
}

container_exists() {
  docker inspect "$CONTAINER" >/dev/null 2>&1
}

container_env_value() {
  local key="$1" line
  while IFS= read -r line; do
    if [[ "$line" == "$key="* ]]; then
      printf '%s\n' "${line#*=}"
      return 0
    fi
  done < <(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" 2>/dev/null)
  return 1
}

profile_matches_container() {
  [[ "$ENGINE" == vllm ]] || return 0
  local actual variable
  actual="$(container_env_value CLUSTER_PROFILE || true)"
  if [[ -n "$actual" ]]; then
    [[ "$actual" == "$PROFILE" ]] || return 1
  else
    for variable in KV_CACHE_DTYPE MAX_MODEL_LEN MAX_NUM_SEQS MTP_NUM_TOKENS; do
      actual="$(container_env_value "$variable" || true)"
      [[ -n "$actual" && "$actual" == "${!variable}" ]] || return 1
    done
  fi
}

status_node() {
  local state endpoint=''
  if ! container_exists || ! profile_matches_container; then
    state=absent
  elif [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)" == true ]]; then
    state=running
    if [[ "$RANK" == 0 ]]; then
      endpoint=" endpoint=http://$(single_tailscale_ipv4):$API_PORT/v1"
    fi
  else
    state=stopped
  fi
  printf 'container=%s state=%s' "$CONTAINER" "$state"
  [[ "$ENGINE" != vllm ]] || printf ' launch_profile=%s' "$PROFILE"
  printf '%s\n' "$endpoint"
}

logs_node() {
  container_exists && profile_matches_container ||
    fail "the requested lifecycle identity is not active"
  docker logs --tail "${LOG_LINES:-200}" "$CONTAINER"
}

stop_node() {
  if container_exists && profile_matches_container; then
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  fi
}

wait_rank() {
  local attempt
  for attempt in 1 2 3 4 5; do
    [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)" == true ]] || fail "container '$CONTAINER' stopped before rank 0 launch"
    sleep 1
  done
  docker logs "$CONTAINER" 2>&1 | grep -Fq "topology rank=$RANK world_size=$WORLD_SIZE master=$MASTER_ADDR:$MASTER_PORT dist_if=$DIST_IF rdma_hca=$RDMA_HCA" || fail "container '$CONTAINER' did not report the expected topology"
}

verify_node() {
  [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)" == true ]] || fail "container '$CONTAINER' is not running"
  profile_matches_container || fail "container '$CONTAINER' has the wrong launch profile"
  docker inspect "$CONTAINER" | python3 -c '
import json, sys
container = json.load(sys.stdin)[0]
host = container["HostConfig"]
assert host["Privileged"] is False
assert "ALL" in (host.get("CapDrop") or [])
assert "no-new-privileges:true" in (host.get("SecurityOpt") or [])
assert host["ReadonlyRootfs"] is True
assert host["PidsLimit"] == 4096
assert host["NetworkMode"] == "host"
for mount in container.get("Mounts", []):
    if mount["Type"] == "bind":
        assert mount["RW"] is False, mount
'
  local command_line logs expected_host
  command_line="$(docker exec "$CONTAINER" bash -lc "tr '\\0' ' ' </proc/1/cmdline")"
  [[ "$command_line" == *"--nnodes 2"* ]] || fail "container '$CONTAINER' command is missing --nnodes 2"
  [[ "$command_line" == *"--node-rank $RANK"* ]] || fail "container '$CONTAINER' command has the wrong node rank"
  if [[ "$ENGINE" == vllm ]]; then
    local loader=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/dspark.py
    docker exec "$CONTAINER" grep -Fq '("gate_up_proj", "w1", 0)' "$loader" ||
      fail "container '$CONTAINER' is missing the DSpark shared-expert w1 loader mapping"
    docker exec "$CONTAINER" grep -Fq '("gate_up_proj", "w3", 1)' "$loader" ||
      fail "container '$CONTAINER' is missing the DSpark shared-expert w3 loader mapping"
    [[ "$command_line" == *"--master-addr $MASTER_ADDR"* ]] || fail "container '$CONTAINER' command has the wrong master address"
    [[ "$command_line" == *"--master-port $MASTER_PORT"* ]] || fail "container '$CONTAINER' command has the wrong master port"
  else
    [[ "$command_line" == *"--dist-init-addr $MASTER_ADDR:$MASTER_PORT"* ]] || fail "container '$CONTAINER' command has the wrong distributed initialization address"
  fi
  [[ "$command_line" == *"--tensor-parallel-size 2"* || "$command_line" == *"--tp-size 2"* ]] || fail "container '$CONTAINER' command is missing two-way tensor parallelism"
  if [[ "$RANK" == 0 ]]; then
    expected_host="$(single_tailscale_ipv4)"
  else
    expected_host=127.0.0.1
  fi
  [[ "$command_line" == *"--host $expected_host"* ]] || fail "container '$CONTAINER' has the wrong API bind address"
  if [[ "$RANK" == 1 ]]; then
    if [[ "$ENGINE" == vllm ]]; then
      [[ "$command_line" == *"--headless"* ]] || fail "container '$CONTAINER' vLLM worker is not headless"
      port_clear || fail "vLLM worker port '$API_PORT' is listening"
    else
      [[ "$command_line" != *"--headless"* ]] || fail "container '$CONTAINER' SGLang worker uses an unsupported headless flag"
    fi
  fi
  logs="$(docker logs "$CONTAINER" 2>&1)"
  grep -Fq "topology rank=$RANK world_size=$WORLD_SIZE master=$MASTER_ADDR:$MASTER_PORT dist_if=$DIST_IF rdma_hca=$RDMA_HCA" <<<"$logs" || fail "container '$CONTAINER' logs do not prove the expected topology"
  grep -F "NCCL INFO NET/IB : Using" <<<"$logs" | grep -Fq "$RDMA_HCA:" || fail "container '$CONTAINER' logs do not prove RoCE transport"
  grep -Fq "via NET/IB/" <<<"$logs" || fail "container '$CONTAINER' logs do not prove rank traffic over RoCE"
  local variable expected
  for variable in NCCL_NET NCCL_IB_DISABLE NCCL_IB_HCA NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME MASTER_ADDR MASTER_PORT WORLD_SIZE NODE_RANK; do
    case "$variable" in
      NCCL_NET) expected=IB ;;
      NCCL_IB_DISABLE) expected=0 ;;
      NCCL_IB_HCA) expected="$RDMA_HCA" ;;
      NCCL_SOCKET_IFNAME|GLOO_SOCKET_IFNAME|TP_SOCKET_IFNAME) expected="$DIST_IF" ;;
      MASTER_ADDR) expected="$MASTER_ADDR" ;;
      MASTER_PORT) expected="$MASTER_PORT" ;;
      WORLD_SIZE) expected="$WORLD_SIZE" ;;
      NODE_RANK) expected="$RANK" ;;
    esac
    [[ "$(docker inspect -f "{{range .Config.Env}}{{println .}}{{end}}" "$CONTAINER" | grep -F -m1 "$variable=" || true)" == "$variable=$expected" ]] || fail "container '$CONTAINER' has the wrong $variable"
  done
  if [[ "$ENGINE" == vllm ]]; then
    [[ "$(container_env_value DGX_MODEL_CAPABILITIES_SHA256 || true)" == "$CAPABILITIES_SHA256" ]] ||
      fail "container '$CONTAINER' has the wrong model capability profile"
  fi
}

case "$ACTION" in
  preflight) preflight ;;
  start) start_node ;;
  wait-rank) wait_rank ;;
  status) status_node ;;
  logs) logs_node ;;
  verify) verify_node ;;
  stop) stop_node ;;
  port-clear) port_clear || fail "port '$API_PORT' is still in use" ;;
  api-host) single_tailscale_ipv4 ;;
esac
