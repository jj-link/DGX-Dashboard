#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

usage() {
  cat <<'EOF'
usage:
  ./serve.sh [vllm|sglang] <exact-artifact-name> [extra args...]
  ./serve.sh cluster <vllm|sglang> <exact-artifact-name> [start|status|logs|verify|stop] [action args...]
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

valid_artifact() {
  [[ "$1" =~ ^[a-z0-9][a-z0-9_]*$ ]]
}

detect_profile() {
  local gpu_name
  if [[ ${INFERENCE_PROFILE+x} ]]; then
    [[ "$INFERENCE_PROFILE" == rtx6000 || "$INFERENCE_PROFILE" == spark ]] || fail "INFERENCE_PROFILE must be rtx6000 or spark"
    printf '%s\n' "$INFERENCE_PROFILE"
    return
  fi
  command -v nvidia-smi >/dev/null 2>&1 || fail "unsupported GPU ''; set INFERENCE_PROFILE=rtx6000 or spark"
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  case "$gpu_name" in
    'NVIDIA RTX PRO 6000 Blackwell Workstation Edition') printf 'rtx6000\n' ;;
    'NVIDIA GB10') printf 'spark\n' ;;
    *) fail "unsupported GPU '$gpu_name'; set INFERENCE_PROFILE=rtx6000 or spark" ;;
  esac
}

list_available() {
  local engine="$1" profile="$2" destination="${3:-/dev/stderr}"
  local metadata package
  local -a packages=()
  shopt -s nullglob
  for metadata in "$ROOT/serve/$engine/$profile"/*/runtime.env; do
    package="$(dirname "$metadata")"
    [[ -f "$package/serve.sh" ]] || continue
    packages+=("$(basename "$package")")
  done
  shopt -u nullglob
  if ((${#packages[@]})); then
    {
      printf 'available %s/%s artifacts:\n' "$engine" "$profile"
      printf '  %s\n' "${packages[@]}" | LC_ALL=C sort
    } >"$destination"
  fi
}

list_cluster() {
  local engine="$1"
  local package required
  local -a packages=()
  shopt -s nullglob
  for package in "$ROOT/serve/cluster/$engine"/*; do
    [[ -d "$package" && -f "$package/runtime.env" ]] || continue
    for required in start status logs verify stop; do
      [[ -x "$package/$required.sh" ]] || continue 2
    done
    packages+=("$(basename "$package")")
  done
  shopt -u nullglob
  if ((${#packages[@]})); then
    printf 'available cluster/%s artifacts:\n' "$engine" >&2
    printf '  %s\n' "${packages[@]}" | LC_ALL=C sort >&2
  fi
}

resolve_single_script() {
  local engine="$1" profile="$2" artifact="$3"
  local base candidate package script
  base="$(realpath -e "$ROOT/serve/$engine/$profile")"
  candidate="$base/$artifact"
  if [[ ! -d "$candidate" || ! -f "$candidate/runtime.env" || ! -f "$candidate/serve.sh" ]]; then
    printf "error: no %s %s recipe named '%s'\n" "$profile" "$engine" "$artifact" >&2
    list_available "$engine" "$profile"
    return 1
  fi
  package="$(realpath -e "$candidate")"
  script="$(realpath -e "$candidate/serve.sh")"
  [[ "$package" == "$base/"* && "$script" == "$package/"* ]] || fail "recipe '$artifact' escapes '$base'"
  printf '%s\0%s\0' "$package" "$script"
}

resolve_cluster_script() {
  local engine="$1" artifact="$2" action="$3"
  local base candidate package script
  base="$(realpath -e "$ROOT/serve/cluster/$engine")"
  candidate="$base/$artifact"
  if [[ ! -d "$candidate" || ! -f "$candidate/runtime.env" || ! -f "$candidate/$action.sh" ]]; then
    printf "error: no cluster %s recipe named '%s'\n" "$engine" "$artifact" >&2
    list_cluster "$engine"
    return 1
  fi
  package="$(realpath -e "$candidate")"
  script="$(realpath -e "$candidate/$action.sh")"
  [[ "$package" == "$base/"* && "$script" == "$package/"* ]] || fail "cluster recipe '$artifact' escapes '$base'"
  printf '%s\0%s\0' "$package" "$script"
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  [[ $# == 1 ]] || { usage >&2; exit 2; }
  profile="$(detect_profile)"
  usage
  printf 'selected profile: %s\n' "$profile"
  list_available vllm "$profile" /dev/stdout
  list_available sglang "$profile" /dev/stdout
  exit 0
fi

if [[ "${1:-}" == cluster ]]; then
  shift
  (( $# >= 2 )) || { usage >&2; exit 2; }
  engine="$1"
  artifact="$2"
  action="${3:-start}"
  if (( $# >= 3 )); then
    shift 3
  else
    shift 2
  fi
  [[ "$engine" == vllm || "$engine" == sglang ]] || fail "cluster engine must be 'vllm' or 'sglang'"
  valid_artifact "$artifact" || fail "invalid artifact name '$artifact'"
  case "$action" in
    start|status|logs|verify|stop) ;;
    *) fail "invalid cluster action '$action'" ;;
  esac
  mapfile -d '' -t resolved < <(resolve_cluster_script "$engine" "$artifact" "$action")
  (( ${#resolved[@]} == 2 )) || exit 1
  package="${resolved[0]}"
  script="${resolved[1]}"
  printf 'target=cluster engine=%s artifact=%s action=%s script=%s\n' "$engine" "$artifact" "$action" "$script" >&2
  exec "$script" "$@"
fi

engine=vllm
if [[ "${1:-}" == vllm || "${1:-}" == sglang ]]; then
  engine="$1"
  shift
fi
(( $# >= 1 )) || { usage >&2; exit 2; }
artifact="$1"
shift
valid_artifact "$artifact" || fail "invalid artifact name '$artifact'"
profile="$(detect_profile)"
mapfile -d '' -t resolved < <(resolve_single_script "$engine" "$profile" "$artifact")
(( ${#resolved[@]} == 2 )) || exit 1
package="${resolved[0]}"
script="${resolved[1]}"
printf 'target=single profile=%s engine=%s artifact=%s script=%s\n' "$profile" "$engine" "$artifact" "$script" >&2
exec "$ROOT/runtime/$profile/run_${engine}_docker.sh" "$package" "$@"
