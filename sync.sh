#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REMOTE_REPO="/home/jjlink/inference"
EXPECTED_ORIGIN="git@github.com:jj-link/inference-workspace.git"
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

if ! GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.https://github.com/.insteadOf GIT_CONFIG_VALUE_0=git@github.com: git -C "$ROOT" fetch --quiet origin main; then
  printf 'local: failed to fetch origin/main\n' >&2
  exit 1
fi

local_status="$(git -C "$ROOT" status --porcelain=v1 --untracked-files=normal)" || {
  printf 'local: failed to read repository status\n' >&2
  exit 1
}
[[ -z "$local_status" ]] || {
  printf 'local: checkout is dirty; refusing to synchronize\n' >&2
  exit 1
}

local_origin="$(git -C "$ROOT" remote get-url origin 2>/dev/null)" || {
  printf 'local: origin is unavailable\n' >&2
  exit 1
}
[[ "$local_origin" == "$EXPECTED_ORIGIN" ]] || {
  printf "local: origin is '%s'; expected '%s'\n" "$local_origin" "$EXPECTED_ORIGIN" >&2
  exit 1
}

local_head="$(git -C "$ROOT" rev-parse HEAD)" || {
  printf 'local: HEAD is unavailable\n' >&2
  exit 1
}
origin_head="$(git -C "$ROOT" rev-parse origin/main)" || {
  printf 'local: origin/main is unavailable\n' >&2
  exit 1
}
[[ "$local_head" == "$origin_head" ]] || {
  printf "local: HEAD '%s' does not equal origin/main '%s'\n" "$local_head" "$origin_head" >&2
  exit 1
}

overall=0
for host in "${targets[@]}"; do
  output="$({ ssh -o BatchMode=yes "$host" bash -s -- "$REMOTE_REPO" "$EXPECTED_ORIGIN" <<'REMOTE'
set -euo pipefail
repo="$1"
expected_origin="$2"

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 || fail "checkout '$repo' is not a Git repository"
git -C "$repo" symbolic-ref --quiet HEAD >/dev/null || fail "checkout '$repo' is detached"
status="$(git -C "$repo" status --porcelain=v1 --untracked-files=normal)" || fail "cannot read checkout status"
[[ -z "$status" ]] || fail "checkout '$repo' is dirty"
origin="$(git -C "$repo" remote get-url origin 2>/dev/null)" || fail "checkout '$repo' has no origin"
[[ "$origin" == "$expected_origin" ]] || fail "origin is '$origin'; expected '$expected_origin'"
git -C "$repo" fetch --quiet origin main || fail "fetch origin/main failed"
git -C "$repo" merge --quiet --ff-only origin/main || fail "fast-forward merge failed"
status="$(git -C "$repo" status --porcelain=v1 --untracked-files=normal)" || fail "cannot read post-merge status"
[[ -z "$status" ]] || fail "checkout '$repo' is dirty after merge"
git -C "$repo" rev-parse HEAD
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
