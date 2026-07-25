#!/usr/bin/env bash
set -uo pipefail

CONTROL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$CONTROL_ROOT/.." && pwd -P)"
REMOTE_REPO="/home/jjlink/dgx-dashboard"
EXPECTED_ORIGIN="git@github.com:jj-link/DGX-Dashboard.git"
CANONICAL_HOSTS=(spark1-ts spark2-ts spark3-ts)

usage() {
  printf 'usage: ./sync.sh <spark1-ts|spark2-ts|spark3-ts|all> [...]\n' >&2
  exit 2
}

(($#)) || usage

declare -A requested=()
if [[ "$1" == all ]]; then
  (($# == 1)) || usage
  targets=("${CANONICAL_HOSTS[@]}")
else
  for host in "$@"; do
    case "$host" in
      spark1-ts|spark2-ts|spark3-ts) requested["$host"]=1 ;;
      *) usage ;;
    esac
  done
  targets=()
  for host in "${CANONICAL_HOSTS[@]}"; do
    [[ -n "${requested[$host]+x}" ]] && targets+=("$host")
  done
fi

local_status="$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=normal)" || {
  printf 'local: failed to read repository status\n' >&2
  exit 1
}
[[ -z "$local_status" ]] || {
  printf 'local: checkout is dirty; refusing to synchronize\n' >&2
  exit 1
}

local_origin="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null)" || {
  printf 'local: origin is unavailable\n' >&2
  exit 1
}
[[ "$local_origin" == "$EXPECTED_ORIGIN" ]] || {
  printf "local: origin is '%s'; expected '%s'\n" "$local_origin" "$EXPECTED_ORIGIN" >&2
  exit 1
}

local_branch="$(git -C "$REPO_ROOT" symbolic-ref --quiet --short HEAD)" || {
  printf 'local: checkout is detached\n' >&2
  exit 1
}
git check-ref-format --branch "$local_branch" >/dev/null 2>&1 || {
  printf "local: branch '%s' is invalid\n" "$local_branch" >&2
  exit 1
}

if ! GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.https://github.com/.insteadOf GIT_CONFIG_VALUE_0=git@github.com: \
  git -C "$REPO_ROOT" fetch --quiet origin "$local_branch"; then
  printf 'local: failed to fetch origin/%s\n' "$local_branch" >&2
  exit 1
fi

local_head="$(git -C "$REPO_ROOT" rev-parse HEAD)" || {
  printf 'local: HEAD is unavailable\n' >&2
  exit 1
}
origin_head="$(git -C "$REPO_ROOT" rev-parse "origin/$local_branch")" || {
  printf 'local: origin/%s is unavailable\n' "$local_branch" >&2
  exit 1
}
[[ "$local_head" == "$origin_head" ]] || {
  printf "local: HEAD '%s' does not equal origin/%s '%s'\n" "$local_head" "$local_branch" "$origin_head" >&2
  exit 1
}

overall=0
for host in "${targets[@]}"; do
  output="$({ ssh -o BatchMode=yes -o ConnectTimeout=20 -o NumberOfPasswordPrompts=0 \
    "$host" bash -s -- "$REMOTE_REPO" "$EXPECTED_ORIGIN" "$local_branch" "$local_head" "$host" <<'REMOTE'
set -euo pipefail
repo="$1"
expected_origin="$2"
expected_branch="$3"
expected_commit="$4"
host="$5"
legacy_backup="${repo}.legacy-pre-unified-20260725"
clone_path="${repo}.clone-in-progress"
moved_legacy=0
cloning=0

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

cleanup() {
  rc=$?
  if (( rc != 0 )); then
    if (( cloning )) && [[ -e "$clone_path" || -L "$clone_path" ]]; then
      rm -rf -- "$clone_path"
    fi
    if (( moved_legacy )) && [[ ! -e "$repo" && ! -L "$repo" ]]; then
      mv -- "$legacy_backup" "$repo"
    fi
  fi
  exit "$rc"
}
trap cleanup EXIT

is_checkout() {
  git -C "$1" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

if [[ -e "$repo" || -L "$repo" ]]; then
  if ! is_checkout "$repo"; then
    [[ "$host" == spark1-ts ]] || fail "path '$repo' exists and is not a Git checkout"
    [[ ! -e "$legacy_backup" && ! -L "$legacy_backup" ]] ||
      fail "legacy backup '$legacy_backup' already exists; refusing to overwrite it"
    mv -- "$repo" "$legacy_backup"
    moved_legacy=1
  fi
fi

if [[ ! -e "$repo" && ! -L "$repo" ]]; then
  [[ -d "${repo%/*}" && -w "${repo%/*}" ]] ||
    fail "checkout parent '${repo%/*}' is not a writable directory"
  [[ ! -e "$clone_path" && ! -L "$clone_path" ]] ||
    fail "stale clone path '$clone_path' exists"
  cloning=1
  GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.https://github.com/.insteadOf \
    GIT_CONFIG_VALUE_0=git@github.com: \
    git clone --quiet --origin origin --branch "$expected_branch" --single-branch \
      "$expected_origin" "$clone_path" || fail "clone failed"
  clone_origin="$(git -C "$clone_path" remote get-url origin 2>/dev/null)" ||
    fail "new checkout has no origin"
  [[ "$clone_origin" == "$expected_origin" ]] ||
    fail "new checkout origin is '$clone_origin'; expected '$expected_origin'"
  clone_branch="$(git -C "$clone_path" symbolic-ref --quiet --short HEAD)" ||
    fail "new checkout is detached"
  [[ "$clone_branch" == "$expected_branch" ]] ||
    fail "new checkout branch is '$clone_branch'; expected '$expected_branch'"
  clone_head="$(git -C "$clone_path" rev-parse HEAD)" ||
    fail "new checkout HEAD is unavailable"
  [[ "$clone_head" == "$expected_commit" ]] ||
    fail "new checkout HEAD is '$clone_head'; expected '$expected_commit'"
  mv -- "$clone_path" "$repo"
  cloning=0
  trap - EXIT
  git -C "$repo" rev-parse HEAD
  exit 0
fi

is_checkout "$repo" || fail "checkout '$repo' is not a Git repository"
current_branch="$(git -C "$repo" symbolic-ref --quiet --short HEAD)" ||
  fail "checkout '$repo' is detached"
[[ "$current_branch" == "$expected_branch" ]] ||
  fail "checkout '$repo' is on branch '$current_branch'; expected '$expected_branch'"
status="$(git -C "$repo" status --porcelain=v1 --untracked-files=normal)" ||
  fail "cannot read checkout status"
[[ -z "$status" ]] || fail "checkout '$repo' is dirty"
origin="$(git -C "$repo" remote get-url origin 2>/dev/null)" ||
  fail "checkout '$repo' has no origin"
[[ "$origin" == "$expected_origin" ]] ||
  fail "origin is '$origin'; expected '$expected_origin'"
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.https://github.com/.insteadOf \
  GIT_CONFIG_VALUE_0=git@github.com: \
  git -C "$repo" fetch --quiet origin "$expected_branch" ||
  fail "fetch origin/$expected_branch failed"
git -C "$repo" merge --quiet --ff-only "origin/$expected_branch" ||
  fail "fast-forward merge failed"
status="$(git -C "$repo" status --porcelain=v1 --untracked-files=normal)" ||
  fail "cannot read post-merge status"
[[ -z "$status" ]] || fail "checkout '$repo' is dirty after merge"
head="$(git -C "$repo" rev-parse HEAD)" || fail "checkout HEAD is unavailable"
[[ "$head" == "$expected_commit" ]] ||
  fail "checkout HEAD is '$head'; expected '$expected_commit'"
printf '%s\n' "$head"
REMOTE
  } 2>&1)"
  if (($? == 0)); then
    printf '%s: ok %s\n' "$host" "$output"
  else
    printf '%s: failed\n' "$host" >&2
    [[ -z "$output" ]] || printf '%s\n' "$output" >&2
    overall=1
  fi
done

exit "$overall"
