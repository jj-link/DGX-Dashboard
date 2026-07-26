#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

(( $# >= 3 )) || fail "internal usage: run-single.sh <vllm|sglang> <rtx6000|spark> <package> [extra args...]"
ENGINE="$1"
PROFILE="$2"
PACKAGE="$3"
shift 3
EXTRA_ARGS=("$@")
[[ "$ENGINE" == vllm || "$ENGINE" == sglang ]] || fail "invalid engine '$ENGINE'"
[[ "$PROFILE" == rtx6000 || "$PROFILE" == spark ]] || fail "invalid profile '$PROFILE'"
[[ -d "$PACKAGE" && -f "$PACKAGE/runtime.env" && -f "$PACKAGE/serve.sh" ]] || fail "invalid package '$PACKAGE'"
PACKAGE="$(cd "$PACKAGE" && pwd -P)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PARSER="$ROOT/tools/parse-runtime-env.py"
VALIDATOR="$ROOT/tools/validate-hf-snapshot.py"
# shellcheck source=single-environment.sh
source "$ROOT/runtime/single-environment.sh"

coproc METADATA_PARSER { python3 "$PARSER" "$PACKAGE/runtime.env"; }
parser_pid="$METADATA_PARSER_PID"
mapfile -d '' -t parsed <&"${METADATA_PARSER[0]}"
wait "$parser_pid" || exit $?
[[ ${#parsed[@]} == 24 ]] || fail "runtime metadata parser returned an invalid record"
declare -A META=()
for ((index = 0; index < ${#parsed[@]}; index += 2)); do
  META["${parsed[index]}"]="${parsed[index + 1]}"
done

case "${EXTRA_ARGS[0]:-}" in
  status|logs|verify|stop)
    exec "$ROOT/runtime/single-lifecycle.sh" \
      "${EXTRA_ARGS[0]}" "${META[CONTAINER_NAME]}" "${META[IMAGE]}" \
      "${META[SERVED]}" "$PROFILE" "${EXTRA_ARGS[@]:1}"
    ;;
esac

OFFLINE="${OFFLINE:-0}"
[[ "$OFFLINE" == 0 || "$OFFLINE" == 1 ]] || fail "OFFLINE must be 0 or 1"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
[[ "$PREFLIGHT_ONLY" == 0 || "$PREFLIGHT_ONLY" == 1 ]] || fail "PREFLIGHT_ONLY must be 0 or 1"
PREFLIGHT_REPLACE_CONTAINER="${PREFLIGHT_REPLACE_CONTAINER:-}"
PREFLIGHT_REPLACE_IMAGE_ID="${PREFLIGHT_REPLACE_IMAGE_ID:-}"
if [[ -n "$PREFLIGHT_REPLACE_CONTAINER" || -n "$PREFLIGHT_REPLACE_IMAGE_ID" ]]; then
  [[ "$PREFLIGHT_ONLY" == 1 ]] || fail "PREFLIGHT_REPLACE_CONTAINER and PREFLIGHT_REPLACE_IMAGE_ID require PREFLIGHT_ONLY=1"
  [[ "$PREFLIGHT_REPLACE_CONTAINER" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "PREFLIGHT_REPLACE_CONTAINER is invalid"
  [[ "$PREFLIGHT_REPLACE_IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "PREFLIGHT_REPLACE_IMAGE_ID must be an immutable Docker image ID"
fi
DETACH="${DETACH:-0}"
KEEP="${KEEP:-0}"
[[ "$DETACH" == 0 || "$DETACH" == 1 ]] || fail "DETACH must be 0 or 1"
[[ "$KEEP" == 0 || "$KEEP" == 1 ]] || fail "KEEP must be 0 or 1"
RESTART_POLICY="${RESTART_POLICY:-no}"
case "$RESTART_POLICY" in
  no|on-failure|always|unless-stopped) ;;
  *) fail "RESTART_POLICY must be 'no', 'on-failure', 'always', or 'unless-stopped'" ;;
esac
if [[ "$RESTART_POLICY" != no && ( "$DETACH" != 1 || "$KEEP" != 1 ) ]]; then
  fail "RESTART_POLICY requires DETACH=1 and KEEP=1"
fi
HOST_PORT="${HOST_PORT:-8000}"
[[ "$HOST_PORT" =~ ^[1-9][0-9]{0,4}$ && "$HOST_PORT" -le 65535 ]] || fail "HOST_PORT must be between 1 and 65535"

if [[ "$PROFILE" == rtx6000 ]]; then
  BIND_ADDRESS=127.0.0.1
else
  command -v tailscale >/dev/null 2>&1 || fail "tailscale is required to resolve the Spark bind address"
  mapfile -t tailscale_addresses < <(tailscale ip -4)
  addresses=()
  for address in "${tailscale_addresses[@]}"; do
    [[ -n "$address" ]] && addresses+=("$address")
  done
  [[ ${#addresses[@]} == 1 ]] || fail "tailscale ip -4 must return exactly one address"
  BIND_ADDRESS="${addresses[0]}"
  IFS=. read -r octet1 octet2 octet3 octet4 extra <<<"$BIND_ADDRESS"
  [[ -z "${extra:-}" && "$octet1" =~ ^[0-9]+$ && "$octet2" =~ ^[0-9]+$ && "$octet3" =~ ^[0-9]+$ && "$octet4" =~ ^[0-9]+$ ]] || fail "tailscale ip -4 returned invalid IPv4 address '$BIND_ADDRESS'"
  for octet in "$octet1" "$octet2" "$octet3" "$octet4"; do
    ((10#$octet <= 255)) || fail "tailscale ip -4 returned invalid IPv4 address '$BIND_ADDRESS'"
  done
fi

repo_directory() {
  local repository="$1"
  printf 'models--%s\n' "${repository//\//--}"
}

candidate_repositories() {
  local repository_directory="$1"
  if [[ ${HF_CACHE+x} ]]; then
    [[ -n "$HF_CACHE" && "$HF_CACHE" == /* ]] || fail "HF_CACHE must be an absolute Hugging Face home"
    printf '%s\n' "$HF_CACHE/hub/$repository_directory"
    return
  fi
  if [[ "$PROFILE" == rtx6000 ]]; then
    printf '%s\n' "$HOME/.cache/huggingface/hub/$repository_directory"
    return
  fi
  printf '%s\n' \
    "$HOME/models/hub/$repository_directory" \
    "$HOME/models/$repository_directory" \
    "/models/hub/hub/$repository_directory" \
    "/models/hub/$repository_directory" \
    "$HOME/hf-cache/hub/$repository_directory" \
    "$HOME/.cache/huggingface/hub/$repository_directory"
}

download_repository() {
  local repository="$1"
  local revision="$2"
  local download_home
  [[ "$OFFLINE" == 0 ]] || return 1
  if [[ ${HF_CACHE+x} ]]; then
    download_home="$HF_CACHE"
  elif [[ "$PROFILE" == rtx6000 ]]; then
    download_home="$HOME/.cache/huggingface"
  else
    download_home="$HOME/models"
  fi
  mkdir -p "$download_home"
  if command -v hf >/dev/null 2>&1; then
    HF_HOME="$download_home" hf download "$repository" --revision "$revision" >/dev/null
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_HOME="$download_home" huggingface-cli download "$repository" --revision "$revision" >/dev/null
  else
    fail "model download requires the 'hf' or 'huggingface-cli' command"
  fi
}

mount_hosts=()
mount_targets=()
add_mount() {
  local host_path="$1"
  local container_path="$2"
  local mount_index
  for ((mount_index = 0; mount_index < ${#mount_targets[@]}; mount_index++)); do
    if [[ "${mount_targets[mount_index]}" == "$container_path" ]]; then
      [[ "${mount_hosts[mount_index]}" == "$host_path" ]] || fail "container mount target '$container_path' has conflicting sources"
      return
    fi
  done
  mount_hosts+=("$host_path")
  mount_targets+=("$container_path")
}

resolve_artifact() {
  local label="$1"
  local value="$2"
  local revision="$3"
  local host_path="$4"
  local lower_label="${label,,}"
  local repository_directory repository_path snapshot_path resolved_path
  local candidate validation_error validation_output

  if [[ -z "$value" ]]; then
    printf -v "${label}_PATH" '%s' ''
    return
  fi

  if [[ -n "$host_path" ]]; then
    [[ -d "$host_path" ]] || fail "$lower_label host path '$host_path' is not an installed directory"
    [[ "$value" == /models/* ]] || fail "$lower_label container path '$value' must be below /models"
    [[ "$(realpath -m "$value")" == "$value" ]] || fail "$lower_label container path '$value' is not canonical"
    resolved_path="$(realpath -e "$host_path")"
    add_mount "$resolved_path" "$value"
    printf -v "${label}_PATH" '%s' "$value"
    return
  fi

  repository_directory="$(repo_directory "$value")"
  repository_path=''
  validation_error=''
  while IFS= read -r candidate; do
    snapshot_path="$candidate/snapshots/$revision"
    if [[ -d "$snapshot_path" ]]; then
      if validation_output="$(python3 "$VALIDATOR" "$snapshot_path" 2>&1)"; then
        repository_path="$candidate"
        break
      elif [[ -z "$validation_error" ]]; then
        validation_error="${validation_output#error: }"
      fi
    fi
  done < <(candidate_repositories "$repository_directory")

  if [[ -z "$repository_path" ]]; then
    download_repository "$value" "$revision" || true
    validation_error=''
    while IFS= read -r candidate; do
      snapshot_path="$candidate/snapshots/$revision"
      if [[ -d "$snapshot_path" ]]; then
        if validation_output="$(python3 "$VALIDATOR" "$snapshot_path" 2>&1)"; then
          repository_path="$candidate"
          break
        elif [[ -z "$validation_error" ]]; then
          validation_error="${validation_output#error: }"
        fi
      fi
    done < <(candidate_repositories "$repository_directory")
  fi
  if [[ -z "$repository_path" ]]; then
    if [[ -n "$validation_error" ]]; then
      fail "$lower_label '$value@$revision' has an incomplete cache snapshot: $validation_error"
    fi
    fail "$lower_label '$value@$revision' is not installed in any configured cache root"
  fi

  repository_path="$(realpath -e "$repository_path")"
  snapshot_path="/models/hub/$repository_directory/snapshots/$revision"
  add_mount "$repository_path" "/models/hub/$repository_directory"
  printf -v "${label}_PATH" '%s' "$snapshot_path"
}

resolve_artifact MODEL "${META[MODEL]}" "${META[MODEL_REVISION]}" "${META[MODEL_HOST_PATH]}"
resolve_artifact DRAFTER "${META[DRAFTER]}" "${META[DRAFTER_REVISION]}" "${META[DRAFTER_HOST_PATH]}"
resolve_artifact TOKENIZER "${META[TOKENIZER]}" "${META[TOKENIZER_REVISION]}" "${META[TOKENIZER_HOST_PATH]}"

container_args=(run)
[[ "$KEEP" == 1 ]] || container_args+=(--rm)
[[ "$DETACH" == 0 ]] || container_args+=(--detach)
[[ "$RESTART_POLICY" == no ]] || container_args+=(--restart "$RESTART_POLICY")
container_args+=(
  --name "${META[CONTAINER_NAME]}"
  --gpus all
  --ipc host
  --shm-size 32g
  --cap-drop ALL
  --security-opt no-new-privileges:true
  --read-only
  --pids-limit 4096
  --ulimit memlock=-1
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=16g
  --mount "type=volume,src=${META[CONTAINER_NAME]}-cache,dst=/root/.cache"
  --env HOME=/root/.cache
  --publish "$BIND_ADDRESS:$HOST_PORT:8000/tcp"
  --mount "type=bind,src=$PACKAGE,dst=/run/inference/package,readonly"
  --env "MODEL_PATH=$MODEL_PATH"
  --env "DRAFTER_PATH=$DRAFTER_PATH"
  --env "TOKENIZER_PATH=$TOKENIZER_PATH"
  --env "SERVED=${META[SERVED]}"
)
passthrough_environment=("${single_engine_environment[@]}")
for variable in "${passthrough_environment[@]}"; do
  if [[ -v "$variable" ]]; then
    container_args+=(--env "$variable=${!variable}")
  fi
done
for ((mount_index = 0; mount_index < ${#mount_targets[@]}; mount_index++)); do
  container_args+=(--mount "type=bind,src=${mount_hosts[mount_index]},dst=${mount_targets[mount_index]},readonly")
done
if [[ "$OFFLINE" == 1 ]]; then
  container_args+=(--env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1)
fi
container_args+=(--entrypoint /bin/bash "${META[IMAGE]}" /run/inference/package/serve.sh "${EXTRA_ARGS[@]}")

print_docker_command() {
  printf 'docker'
  printf ' %q' "${container_args[@]}"
  printf '\n'
}

verify_replacement_container() {
  python3 - "$PREFLIGHT_REPLACE_CONTAINER" "$PREFLIGHT_REPLACE_IMAGE_ID" "$BIND_ADDRESS" "$HOST_PORT" <<'PY'
import json
import subprocess
import sys

name, expected_image, bind_address, host_port = sys.argv[1:]
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
actual_bindings = container.get("HostConfig", {}).get("PortBindings", {}).get("8000/tcp")
expected_bindings = [{"HostIp": bind_address, "HostPort": host_port}]
if actual_bindings != expected_bindings:
    raise SystemExit(
        f"replacement container {name!r} binding is {actual_bindings!r}; "
        f"expected {expected_bindings!r}"
    )
PY
}

verify_port_available() {
  python3 - "$BIND_ADDRESS" "$HOST_PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    try:
        listener.bind((host, port))
    except OSError as error:
        raise SystemExit(f"port {host}:{port} is unavailable: {error}")
PY
}

if [[ "${SERVE_DRY_RUN:-0}" == 1 ]]; then
  print_docker_command
  exit 0
fi
[[ "${SERVE_DRY_RUN:-0}" == 0 ]] || fail "SERVE_DRY_RUN must be 0 or 1"
command -v docker >/dev/null 2>&1 || fail "docker is required"
if ! docker image inspect "${META[IMAGE]}" >/dev/null 2>&1; then
  if [[ -x "$PACKAGE/build-image.sh" ]]; then
    fail "image '${META[IMAGE]}' is unavailable; run '$PACKAGE/build-image.sh'"
  fi
  [[ "$PREFLIGHT_ONLY" == 0 ]] || fail "image '${META[IMAGE]}' is unavailable during preflight"
  [[ "$OFFLINE" == 0 ]] || fail "image '${META[IMAGE]}' is unavailable in offline mode"
  docker pull "${META[IMAGE]}"
fi
if docker container inspect "${META[CONTAINER_NAME]}" >/dev/null 2>&1; then
  fail "container '${META[CONTAINER_NAME]}' already exists; remove it before serving"
fi
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  if [[ -n "$PREFLIGHT_REPLACE_CONTAINER" ]]; then
    verify_replacement_container
  else
    verify_port_available
  fi
  print_docker_command
  printf 'preflight complete; no container created\n'
  exit 0
fi
exec docker "${container_args[@]}"
