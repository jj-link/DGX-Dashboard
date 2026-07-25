#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

(( $# >= 5 )) || fail "internal usage: single-lifecycle.sh <action> <container> <image> <served> <profile> [action args...]"
ACTION="$1"
CONTAINER="$2"
IMAGE="$3"
SERVED="$4"
PROFILE="$5"
shift 5
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
INSPECTOR="$ROOT/tools/inspect-single-container.py"

[[ "$ACTION" == status || "$ACTION" == logs || "$ACTION" == verify || "$ACTION" == stop ]] ||
  fail "invalid lifecycle action '$ACTION'"
command -v docker >/dev/null 2>&1 || fail "docker is required"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"
[[ -x "$INSPECTOR" ]] || fail "container inspector '$INSPECTOR' is unavailable"

inspect_json=''
inspect_container() {
  if inspect_json="$(docker container inspect "$CONTAINER" 2>/dev/null)"; then
    return 0
  fi
  docker info >/dev/null 2>&1 || fail "Docker daemon is unavailable"
  return 1
}

inspect_identity() {
  local mode="$1"
  printf '%s' "$inspect_json" |
    python3 "$INSPECTOR" "$mode" "$CONTAINER" "$IMAGE" "$SERVED" "$PROFILE"
}

case "$ACTION" in
  status)
    (($# == 0)) || fail "status accepts no arguments"
    if ! inspect_container; then
      printf 'container=%s state=absent\n' "$CONTAINER"
      exit 0
    fi
    state="$(inspect_identity identity)" || exit $?
    printf 'container=%s state=%s\n' "$CONTAINER" "$state"
    ;;
  logs)
    (($# <= 1)) || fail "logs accepts at most one line count"
    lines="${1:-100}"
    [[ "$lines" =~ ^[1-9][0-9]{0,3}$ && "$lines" -le 1000 ]] ||
      fail "log line count must be between 1 and 1000"
    inspect_container || fail "container '$CONTAINER' is absent"
    inspect_identity identity >/dev/null || exit $?
    exec docker container logs --tail "$lines" "$CONTAINER"
    ;;
  verify)
    (($# == 0)) || fail "verify accepts no arguments"
    inspect_container || fail "container '$CONTAINER' is absent"
    endpoint="$(inspect_identity verify)" || exit $?
    model="$(python3 - "$endpoint" "$SERVED" <<'PY'
import json
import sys
import urllib.request

endpoint, expected = sys.argv[1:]
try:
    with urllib.request.urlopen(f"{endpoint}/models", timeout=5) as response:
        payload = json.load(response)
except Exception as error:
    raise SystemExit(f"error: serving health check failed: {error}")
rows = payload.get("data") if isinstance(payload, dict) else None
models = [row.get("id") for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
if expected not in models:
    raise SystemExit(
        f"error: serving health check returned models {models!r}; expected {expected!r}"
    )
print(expected)
PY
)" || exit $?
    printf 'container=%s state=running endpoint=%s model=%s\n' \
      "$CONTAINER" "$endpoint" "$model"
    ;;
  stop)
    (($# == 0)) || fail "stop accepts no arguments"
    if ! inspect_container; then
      printf 'container=%s state=absent\n' "$CONTAINER"
      exit 0
    fi
    state="$(inspect_identity identity)" || exit $?
    case "$state" in
      running|restarting|paused)
        docker container stop --time 30 "$CONTAINER" >/dev/null ||
          fail "failed to stop container '$CONTAINER'"
        ;;
    esac
    if docker container rm "$CONTAINER" >/dev/null; then
      printf 'container=%s state=absent\n' "$CONTAINER"
      exit 0
    fi
    if ! inspect_container; then
      printf 'container=%s state=absent\n' "$CONTAINER"
      exit 0
    fi
    fail "failed to remove container '$CONTAINER'"
    ;;
esac
