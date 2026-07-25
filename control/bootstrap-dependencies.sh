#!/usr/bin/env bash
set -euo pipefail

CONTROL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$CONTROL_ROOT/.." && pwd -P)"
MANIFEST="${DEPENDENCY_MANIFEST:-$CONTROL_ROOT/dependencies/manifest.tsv}"
DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-$CONTROL_ROOT}"

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

[[ -f "$MANIFEST" ]] || fail "dependency manifest '$MANIFEST' does not exist"
mkdir -p "$DEPENDENCY_ROOT"
DEPENDENCY_ROOT="$(cd "$DEPENDENCY_ROOT" && pwd -P)"

if (($#)); then
  declare -A requested=()
  for dependency in "$@"; do
    [[ "$dependency" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || fail "invalid dependency name '$dependency'"
    requested["$dependency"]=0
  done
fi

declare -A seen=()
exec 3<"$MANIFEST"
IFS=$'\t' read -r h_name h_upstream h_revision h_target h_managed h_extra <&3 || fail "dependency manifest is empty"
[[ "$h_name" == name && "$h_upstream" == upstream && "$h_revision" == revision && "$h_target" == target && "$h_managed" == managed_commit && -z "${h_extra:-}" ]] || fail "dependency manifest has an invalid header"

while IFS=$'\t' read -r name upstream revision target managed_commit extra <&3; do
  [[ -n "$name$upstream$revision$target$managed_commit${extra:-}" ]] || continue
  [[ -z "${extra:-}" ]] || fail "dependency '$name' has too many manifest fields"
  [[ "$name" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || fail "invalid dependency name '$name'"
  [[ -z "${seen[$name]+x}" ]] || fail "duplicate dependency '$name' in manifest"
  seen["$name"]=1
  [[ -n "$upstream" && "$upstream" != *$'\n'* && "$upstream" != *$'\r'* ]] || fail "dependency '$name' has an invalid upstream"
  [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || fail "dependency '$name' revision must be 40 lowercase hexadecimal characters"
  [[ "$managed_commit" =~ ^[0-9a-f]{40}$ ]] || fail "dependency '$name' managed commit must be 40 lowercase hexadecimal characters"
  [[ -n "$target" && "$target" != /* && "$target" != . && "$target" != *$'\n'* && "$target" != *$'\r'* ]] || fail "dependency '$name' has an invalid target '$target'"
  case "/$target/" in
    */../*|*/./*|*//*) fail "dependency '$name' has an invalid target '$target'" ;;
  esac

  if (($#)) && [[ -z "${requested[$name]+x}" ]]; then
    continue
  fi
  if (($#)); then
    requested["$name"]=1
  fi

  package="$CONTROL_ROOT/dependencies/$name"
  target_abs="$DEPENDENCY_ROOT/$target"

  if [[ -e "$target_abs" || -L "$target_abs" ]]; then
    [[ -d "$target_abs" ]] || fail "dependency '$name' target '$target_abs' is not a directory"
    git -C "$target_abs" rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "dependency '$name' target '$target_abs' is not a Git worktree"
    status="$(git -C "$target_abs" status --porcelain=v1 --untracked-files=all)"
    [[ -z "$status" ]] || fail "dependency '$name' target '$target_abs' is dirty; refusing to modify it"
    actual_upstream="$(git -C "$target_abs" remote get-url origin 2>/dev/null || true)"
    [[ "$actual_upstream" == "$upstream" ]] || fail "dependency '$name' origin is '$actual_upstream'; expected '$upstream'"
    head="$(git -C "$target_abs" rev-parse HEAD)"
    if [[ "$head" == "$managed_commit" ]]; then
      printf '%s: ready at %s\n' "$name" "$managed_commit"
      continue
    fi
    [[ "$head" == "$revision" ]] || fail "dependency '$name' is at '$head'; expected '$revision' or '$managed_commit'"
  else
    mkdir -p "$(dirname "$target_abs")"
    git -c core.autocrlf=false clone --no-checkout "$upstream" "$target_abs"
  fi

  git -C "$target_abs" config core.autocrlf false
  git -C "$target_abs" checkout --quiet --detach "$revision"

  shopt -s nullglob
  patches=("$package"/patches/*.patch)
  shopt -u nullglob
  for patch in "${patches[@]}"; do
    git -C "$target_abs" apply --index --whitespace=nowarn "$patch"
  done

  if [[ -d "$package/overlay" ]]; then
    cp -a "$package/overlay/." "$target_abs/"
  fi
  git -C "$target_abs" add -A

  GIT_AUTHOR_NAME='Inference Workspace' \
  GIT_AUTHOR_EMAIL='inference-workspace@invalid' \
  GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' \
  GIT_COMMITTER_NAME='Inference Workspace' \
  GIT_COMMITTER_EMAIL='inference-workspace@invalid' \
  GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' \
    git -C "$target_abs" \
      -c commit.gpgSign=false \
      -c core.hooksPath=/dev/null \
      commit --quiet --allow-empty --no-verify -m "inference-workspace managed $name"

  actual="$(git -C "$target_abs" rev-parse HEAD)"
  [[ "$actual" == "$managed_commit" ]] || fail "dependency '$name' produced managed commit '$actual'; expected '$managed_commit'"
  [[ -z "$(git -C "$target_abs" status --porcelain=v1 --untracked-files=all)" ]] || fail "dependency '$name' is dirty after bootstrap"
  printf '%s: created managed commit %s\n' "$name" "$managed_commit"
done

if (($#)); then
  for dependency in "$@"; do
    [[ "${requested[$dependency]}" == 1 ]] || fail "unknown dependency '$dependency'"
  done
fi
