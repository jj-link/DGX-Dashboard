#!/usr/bin/env bash
set -euo pipefail

CONTROL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$CONTROL_ROOT/.." && pwd -P)"
# shellcheck source=runtime/single-environment.sh
source "$CONTROL_ROOT/runtime/single-environment.sh"

usage() {
  cat <<'EOF'
usage:
  ./serve.sh local   <vllm|sglang> <exact-artifact-name> [engine args...]
  ./serve.sh spark1  <vllm|sglang> <exact-artifact-name> [engine args...]
  ./serve.sh spark2  <vllm|sglang> <exact-artifact-name> [engine args...]
  ./serve.sh spark3  <vllm|sglang> <exact-artifact-name> [engine args...]
  ./serve.sh cluster <vllm|sglang> <exact-artifact-name> [start|status|logs|verify|stop] [profile] [action args...]

single-device lifecycle after the artifact:
  status | logs [1..1000] | verify | stop
cluster profiles:
  Profile-aware recipes accept a validated profile after the action.
  Omitting the action starts the recipe with its declared default profile.

targets:
  local    workstation (rtx6000)
  spark1   spark1-ts (spark)
  spark2   spark2-ts (spark)
  spark3   spark3-ts (spark)
  cluster  Spark 2 + Spark 3 cluster
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage_error() {
  usage >&2
  exit 2
}

valid_artifact() {
  [[ "$1" =~ ^[a-z0-9][a-z0-9_]*$ ]]
}

require_controller_gpu() {
  local gpu_name
  command -v nvidia-smi >/dev/null 2>&1 || fail "single-device commands must be initiated from the RTX workstation"
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  [[ "$gpu_name" == 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition' ]] || \
    fail "single-device commands must be initiated from the RTX workstation"
}

controller_commit() {
  [[ -d "$REPO_ROOT/.git" ]] || fail "'$REPO_ROOT' is not a Git checkout"
  git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null || fail "'$REPO_ROOT' is detached"
  [[ -z "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]] || fail "'$REPO_ROOT' is dirty"
  git -C "$REPO_ROOT" rev-parse HEAD
}

list_available() {
  local engine="$1" profile="$2" destination_fd="${3:-2}"
  local metadata package
  local -a packages=()
  shopt -s nullglob
  for metadata in "$CONTROL_ROOT/serve/$engine/$profile"/*/runtime.env; do
    package="$(dirname "$metadata")"
    [[ -f "$package/serve.sh" ]] || continue
    packages+=("$(basename "$package")")
  done
  shopt -u nullglob
  if ((${#packages[@]})); then
    {
      printf 'available %s/%s artifacts:\n' "$engine" "$profile"
      printf '  %s\n' "${packages[@]}" | LC_ALL=C sort
    } >&"$destination_fd"
  fi
}

profile_summary() (
  source "$1"
  case "$KV_CACHE_DTYPE" in
    fp8_ds_mla) kv='FP8 DS-MLA KV' ;;
    nvfp4_ds_mla) kv='NVFP4 DS-MLA KV' ;;
    *) kv="$KV_CACHE_DTYPE KV" ;;
  esac
  case "$MAX_MODEL_LEN" in
    1048576) context='1M context' ;;
    350000) context='350K context' ;;
    *) context="$MAX_MODEL_LEN context" ;;
  esac
  printf '%s, %s, %s sequences, MTP%s' "$kv" "$context" "$MAX_NUM_SEQS" "$MTP_NUM_TOKENS"
)

list_cluster() {
  local engine="$1" destination_fd="${2:-2}"
  local package required profile_file profile default_profile
  local -a packages=() profile_names=()
  shopt -s nullglob
  for package in "$CONTROL_ROOT/serve/cluster/$engine"/*; do
    [[ -d "$package" && -f "$package/runtime.env" ]] || continue
    for required in start status logs verify stop; do
      [[ -x "$package/$required.sh" ]] || continue 2
    done
    packages+=("$(basename "$package")")
  done
  shopt -u nullglob
  if ((${#packages[@]})); then
    {
      printf 'available cluster/%s artifacts:\n' "$engine"
      printf '  %s\n' "${packages[@]}" | LC_ALL=C sort
    } >&"$destination_fd"
    for package in "$CONTROL_ROOT/serve/cluster/$engine"/*; do
      [[ -d "$package/profiles" && -f "$package/profiles/default" ]] || continue
      profile_names=()
      for profile_file in "$package/profiles"/*.env; do
        profile_names+=("$(basename "${profile_file%.env}")")
      done
      mapfile -t profile_names < <(printf '%s\n' "${profile_names[@]}" | LC_ALL=C sort)
      default_profile="$(<"$package/profiles/default")"
      printf '  %s profiles: %s (default: %s)\n' \
        "$(basename "$package")" "${profile_names[*]}" "$default_profile"
      for profile in "${profile_names[@]}"; do
        printf '    %s: %s\n' "$profile" "$(profile_summary "$package/profiles/$profile.env")"
      done
    done >&"$destination_fd"
  fi
}

resolve_single_script() {
  local engine="$1" profile="$2" artifact="$3"
  local base candidate package script
  base="$(realpath -e "$CONTROL_ROOT/serve/$engine/$profile")"
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
  base="$(realpath -e "$CONTROL_ROOT/serve/cluster/$engine")"
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
  [[ $# == 1 ]] || usage_error
  usage
  list_available vllm rtx6000 1
  list_available sglang rtx6000 1
  list_available vllm spark 1
  list_available sglang spark 1
  list_cluster vllm 1
  list_cluster sglang 1
  exit 0
fi

(( $# >= 3 )) || usage_error
target="$1"
engine="$2"
artifact="$3"
shift 3
case "$target" in
  local|spark1|spark2|spark3|cluster) ;;
  *) usage_error ;;
esac
[[ "$engine" == vllm || "$engine" == sglang ]] || usage_error
valid_artifact "$artifact" || fail "invalid artifact name '$artifact'"

if [[ "$target" == cluster ]]; then
  action="${1:-start}"
  if (($#)); then shift; fi
  case "$action" in
    start|status|logs|verify|stop) ;;
    *) fail "invalid cluster action '$action'" ;;
  esac
  mapfile -d '' -t resolved < <(resolve_cluster_script "$engine" "$artifact" "$action")
  (( ${#resolved[@]} == 2 )) || exit 1
  script="${resolved[1]}"
  printf 'target=cluster engine=%s artifact=%s action=%s\n' "$engine" "$artifact" "$action" >&2
  exec "$script" "$@"
fi

single_action=start
case "${1:-}" in
  status|logs|verify|stop) single_action="$1" ;;
esac
require_controller_gpu
if [[ "$target" == local ]]; then
  mapfile -d '' -t resolved < <(resolve_single_script "$engine" rtx6000 "$artifact")
  (( ${#resolved[@]} == 2 )) || exit 1
  package="${resolved[0]}"
  script="${resolved[1]}"
  printf 'target=local profile=rtx6000 engine=%s artifact=%s action=%s\n' "$engine" "$artifact" "$single_action" >&2
  exec "$CONTROL_ROOT/runtime/rtx6000/run_${engine}_docker.sh" "$package" "$@"
fi

case "$target" in
  spark1) host=spark1-ts ;;
  spark2) host=spark2-ts ;;
  spark3) host=spark3-ts ;;
esac
expected_commit="$(controller_commit)"
mapfile -d '' -t resolved < <(resolve_single_script "$engine" spark "$artifact")
(( ${#resolved[@]} == 2 )) || exit 1
script="${resolved[1]}"
printf 'target=%s host=%s profile=spark engine=%s artifact=%s action=%s\n' "$target" "$host" "$engine" "$artifact" "$single_action" >&2
command=(env)
for variable in "${single_control_environment[@]}" "${single_engine_environment[@]}"; do
  [[ -v "$variable" ]] && command+=("$variable=${!variable}")
done
command+=(/home/jjlink/dgx-dashboard/control/runtime/spark/run-remote-single.sh "$expected_commit" "$engine" "$artifact" "$@")
quoted=''
for argument in "${command[@]}"; do
  printf -v quoted '%s %q' "$quoted" "$argument"
done
exec ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ForwardAgent=no -o ClearAllForwardings=yes -o RequestTTY=no -o ConnectTimeout=20 -o ConnectionAttempts=3 -o ServerAliveInterval=60 -o ServerAliveCountMax=30 "$host" "exec$quoted"
