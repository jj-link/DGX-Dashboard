#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

(( $# >= 1 )) || fail "usage: cluster.sh <start|status|logs|verify|stop> [action args...]"
ACTION="$1"
shift
[[ "$ACTION" =~ ^(start|status|logs|verify|stop)$ ]] || fail "invalid cluster action '$ACTION'"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CONTROL_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd -P)"
REPO_ROOT="$(cd "$CONTROL_ROOT/.." && pwd -P)"
ENGINE="$(basename "$(dirname "$SCRIPT_DIR")")"
ARTIFACT="$(basename "$SCRIPT_DIR")"
PACKAGE="$CONTROL_ROOT/serve/cluster/$ENGINE/$ARTIFACT"
PARSER="$CONTROL_ROOT/tools/parse-runtime-env.py"
REMOTE_REPO_ROOT=/home/jjlink/dgx-dashboard
HEAD_HOST=spark2-ts
WORKER_HOST=spark3-ts
API_PORT=8888
case "$ACTION" in
  start)
    (( $# <= 1 )) || fail "start accepts at most one profile argument"
    if (( $# == 1 )); then
      [[ "$ENGINE" == vllm ]] || fail "the SGLang cluster start action accepts no profile argument"
      CLUSTER_PROFILE="$1"
      export CLUSTER_PROFILE
    fi
    ;;
  logs)
    (( $# <= 1 )) || fail "logs accepts at most one line-count argument"
    if (( $# == 1 )); then
      [[ "$1" =~ ^[1-9][0-9]{0,5}$ ]] || fail "log line count must be between 1 and 999999"
      LOG_LINES="$1"
      export LOG_LINES
    fi
    ;;
  *)
    (( $# == 0 )) || fail "$ACTION accepts no action arguments"
    ;;
esac


[[ -x "$PARSER" ]] || fail "missing metadata parser '$PARSER'"
declare -A META=()
coproc METADATA_PARSER { python3 "$PARSER" "$PACKAGE/runtime.env"; }
parser_pid=$METADATA_PARSER_PID
mapfile -d '' -t parsed <&"${METADATA_PARSER[0]}"
wait "$parser_pid" || exit $?
(( ${#parsed[@]} == 24 )) || fail "runtime metadata parser returned ${#parsed[@]} fields; expected 24"
for ((index = 0; index < ${#parsed[@]}; index += 2)); do
  META["${parsed[index]}"]="${parsed[index + 1]}"
done
SERVED="${META[SERVED]}"
case "$ENGINE/$ARTIFACT" in
  sglang/unsloth_qwen36_27b_nvfp4_dflash_tp2)
    MAX_MODEL_LEN=262144
    ;;
  vllm/deepseek_ai_deepseek_v4_flash_dspark_tp2)
    PROFILE="${CLUSTER_PROFILE:-quality}"
    [[ "$PROFILE" =~ ^(balanced|quality|throughput)$ ]] || fail "invalid DeepSeek cluster profile '$PROFILE'"
    PROFILE_FILE="$PACKAGE/profiles/$PROFILE.env"
    [[ -f "$PROFILE_FILE" ]] || fail "missing DeepSeek cluster profile '$PROFILE_FILE'"
    set -a
    source "$PROFILE_FILE"
    set +a
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
    ;;
  *) fail "unsupported cluster artifact '$ENGINE/$ARTIFACT'" ;;
esac

controller_commit() {
  [[ -d "$REPO_ROOT/.git" ]] || fail "'$REPO_ROOT' is not a Git checkout"
  git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null || fail "'$REPO_ROOT' is detached"
  [[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]] || fail "'$REPO_ROOT' is dirty"
  git -C "$REPO_ROOT" rev-parse HEAD
}

EXPECTED_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)"

run_remote() {
  local host="$1" node_action="$2" rank="$3"
  local -a command=(env "EXPECTED_COMMIT=$EXPECTED_COMMIT")
  [[ -z "${HF_CACHE+x}" ]] || command+=("HF_CACHE=$HF_CACHE")
  [[ -z "${CLUSTER_PROFILE+x}" ]] || command+=("CLUSTER_PROFILE=$CLUSTER_PROFILE")
  [[ -z "${LOG_LINES+x}" ]] || command+=("LOG_LINES=$LOG_LINES")
  [[ -z "${PREFLIGHT_REPLACE_CONTAINER+x}" ]] || command+=("PREFLIGHT_REPLACE_CONTAINER=$PREFLIGHT_REPLACE_CONTAINER")
  [[ -z "${PREFLIGHT_REPLACE_IMAGE_ID+x}" ]] || command+=("PREFLIGHT_REPLACE_IMAGE_ID=$PREFLIGHT_REPLACE_IMAGE_ID")
  [[ -z "${PREFLIGHT_REPLACE_NETWORK_MODE+x}" ]] || command+=("PREFLIGHT_REPLACE_NETWORK_MODE=$PREFLIGHT_REPLACE_NETWORK_MODE")
  command+=("$REMOTE_REPO_ROOT/control/runtime/cluster/run-node.sh" "$node_action" "$ENGINE" "$ARTIFACT" "$rank")
  local quoted='' argument
  for argument in "${command[@]}"; do
    printf -v quoted '%s %q' "$quoted" "$argument"
  done
  ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ForwardAgent=no -o ClearAllForwardings=yes -o RequestTTY=no -o ConnectTimeout=20 -o ConnectionAttempts=3 "$host" "exec$quoted"
}

API_HOST=''
preflight_both() {
  EXPECTED_COMMIT="$(controller_commit)"
  local head_output='' worker_output='' head_rc=0 worker_rc=0 key value
  head_output="$(run_remote "$HEAD_HOST" preflight 0)" || head_rc=$?
  worker_output="$(run_remote "$WORKER_HOST" preflight 1)" || worker_rc=$?
  printf '%s preflight:\n%s\n' "$HEAD_HOST" "$head_output"
  printf '%s preflight:\n%s\n' "$WORKER_HOST" "$worker_output"
  (( head_rc == 0 && worker_rc == 0 )) || return 1
  while IFS='=' read -r key value; do
    [[ "$key" == API_HOST ]] && API_HOST="$value"
  done <<<"$head_output"
  [[ -n "$API_HOST" ]] || fail "head preflight returned no Tailscale API address"
}

resolve_api_host() {
  API_HOST="$(run_remote "$HEAD_HOST" api-host 0)"
  [[ -n "$API_HOST" ]] || fail "head returned no Tailscale API address"
}

wait_api() {
  local attempt
  for ((attempt = 1; attempt <= 120; attempt += 1)); do
    if curl -fsS --connect-timeout 2 --max-time 5 "http://$API_HOST:$API_PORT/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

verify_api() {
  python3 - "$API_HOST" "$API_PORT" "$SERVED" "$MAX_MODEL_LEN" <<'PY'
import json
import sys
import urllib.request

host, port, served, max_model_len = sys.argv[1:]
base = f"http://{host}:{port}/v1"
with urllib.request.urlopen(f"{base}/models", timeout=30) as response:
    models = json.load(response)
rows = [row for row in models.get("data", []) if row.get("id") == served]
if len(rows) != 1:
    raise SystemExit(f"expected exactly one model named {served!r}, found {len(rows)}")
if int(rows[0].get("max_model_len", -1)) != int(max_model_len):
    raise SystemExit(f"model max_model_len mismatch: {rows[0].get('max_model_len')!r}")
payload = json.dumps({
    "model": served,
    "messages": [{"role": "user", "content": "Reply with exactly STANDARDIZATION_OK and nothing else."}],
    "max_tokens": 16,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
request = urllib.request.Request(
    f"{base}/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=180) as response:
    chat = json.load(response)
choice = chat["choices"][0]
canonical = {
    "model": chat.get("model"),
    "content": choice.get("message", {}).get("content"),
    "finish_reason": choice.get("finish_reason"),
}
if canonical["model"] != served:
    raise SystemExit(f"chat model mismatch: {canonical['model']!r}")
if canonical["content"] != "STANDARDIZATION_OK":
    raise SystemExit(f"chat content mismatch: {canonical['content']!r}")
if canonical["finish_reason"] != "stop":
    raise SystemExit(f"chat finish_reason mismatch: {canonical['finish_reason']!r}")
print(json.dumps({"model": rows[0], "chat": canonical}, sort_keys=True))
PY
}

verify_cluster() {
  local head_rc=0 worker_rc=0 api_rc=0
  run_remote "$HEAD_HOST" verify 0 || head_rc=$?
  run_remote "$WORKER_HOST" verify 1 || worker_rc=$?
  verify_api || api_rc=$?
  (( head_rc == 0 && worker_rc == 0 && api_rc == 0 ))
}

CREATED_HEAD=0
CREATED_WORKER=0
cleanup_new() {
  local rc=0
  if (( CREATED_HEAD )); then
    run_remote "$HEAD_HOST" stop 0 || rc=1
  fi
  if (( CREATED_WORKER )); then
    run_remote "$WORKER_HOST" stop 1 || rc=1
  fi
  run_remote "$HEAD_HOST" port-clear 0 || rc=1
  run_remote "$WORKER_HOST" port-clear 1 || rc=1
  return "$rc"
}

abort_start() {
  local signal_rc=$?
  trap - INT TERM HUP
  cleanup_new || true
  exit "$signal_rc"
}

start_cluster() {
  preflight_both || fail "cluster preflight failed; no containers were created"
  if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
    printf 'preflight complete; no containers created\n'
    return 0
  fi
  trap abort_start INT TERM HUP
  if ! run_remote "$WORKER_HOST" start 1; then
    cleanup_new || true
    fail "worker launch failed"
  fi
  CREATED_WORKER=1
  if ! run_remote "$WORKER_HOST" wait-rank 1; then
    cleanup_new || true
    fail "worker did not remain ready for rank 0"
  fi
  if ! run_remote "$HEAD_HOST" start 0; then
    cleanup_new || true
    fail "head launch failed"
  fi
  CREATED_HEAD=1
  if ! wait_api; then
    logs_cluster || true
    cleanup_new || true
    fail "cluster API did not become ready"
  fi
  if ! verify_cluster; then
    logs_cluster || true
    cleanup_new || true
    fail "cluster verification failed"
  fi
  trap - INT TERM HUP
  CREATED_HEAD=0
  CREATED_WORKER=0
  printf '%s/%s is ready at http://%s:%s/v1\n' "$ENGINE" "$ARTIFACT" "$API_HOST" "$API_PORT"
}

status_cluster() {
  local rc=0
  printf '%s:\n' "$HEAD_HOST"
  run_remote "$HEAD_HOST" status 0 || rc=1
  printf '%s:\n' "$WORKER_HOST"
  run_remote "$WORKER_HOST" status 1 || rc=1
  return "$rc"
}

logs_cluster() {
  local rc=0
  printf '===== %s rank 0 =====\n' "$HEAD_HOST"
  run_remote "$HEAD_HOST" logs 0 || rc=1
  printf '===== %s rank 1 =====\n' "$WORKER_HOST"
  run_remote "$WORKER_HOST" logs 1 || rc=1
  return "$rc"
}

stop_cluster() {
  local rc=0
  run_remote "$HEAD_HOST" stop 0 || rc=1
  run_remote "$WORKER_HOST" stop 1 || rc=1
  run_remote "$HEAD_HOST" port-clear 0 || rc=1
  run_remote "$WORKER_HOST" port-clear 1 || rc=1
  return "$rc"
}

case "$ACTION" in
  start) start_cluster ;;
  status) status_cluster ;;
  logs) logs_cluster ;;
  verify) resolve_api_host; verify_cluster ;;
  stop) stop_cluster ;;
esac
