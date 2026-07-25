#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export GIT_CONFIG_NOSYSTEM=1
export HOME="$TMP/home"
mkdir -p "$HOME" "$TMP/upstream" "$TMP/package-root"
git -C "$TMP/upstream" init --quiet --initial-branch=main
printf 'base\n' >"$TMP/upstream/base.txt"
GIT_AUTHOR_NAME='Fixture' \
GIT_AUTHOR_EMAIL='fixture@invalid' \
GIT_AUTHOR_DATE='1999-01-01T00:00:00Z' \
GIT_COMMITTER_NAME='Fixture' \
GIT_COMMITTER_EMAIL='fixture@invalid' \
GIT_COMMITTER_DATE='1999-01-01T00:00:00Z' \
  git -C "$TMP/upstream" -c commit.gpgSign=false -c core.hooksPath=/dev/null \
    add base.txt
git -C "$TMP/upstream" config user.name Fixture
git -C "$TMP/upstream" config user.email fixture@invalid
GIT_AUTHOR_DATE='1999-01-01T00:00:00Z' GIT_COMMITTER_DATE='1999-01-01T00:00:00Z' \
  git -C "$TMP/upstream" -c commit.gpgSign=false -c core.hooksPath=/dev/null \
    commit --quiet --no-verify -m base
revision="$(git -C "$TMP/upstream" rev-parse HEAD)"

git clone --quiet --no-checkout "$TMP/upstream" "$TMP/reference"
git -C "$TMP/reference" checkout --quiet --detach "$revision"
GIT_AUTHOR_NAME='Inference Workspace' \
GIT_AUTHOR_EMAIL='inference-workspace@invalid' \
GIT_AUTHOR_DATE='2000-01-01T00:00:00Z' \
GIT_COMMITTER_NAME='Inference Workspace' \
GIT_COMMITTER_EMAIL='inference-workspace@invalid' \
GIT_COMMITTER_DATE='2000-01-01T00:00:00Z' \
  git -C "$TMP/reference" -c commit.gpgSign=false -c core.hooksPath=/dev/null \
    commit --quiet --allow-empty --no-verify -m 'inference-workspace managed codex-shim-sero'
managed="$(git -C "$TMP/reference" rev-parse HEAD)"

printf 'name\tupstream\trevision\ttarget\tmanaged_commit\n' >"$TMP/manifest.tsv"
printf 'codex-shim-sero\t%s\t%s\tdeps/codex-shim\t%s\n' \
  "$TMP/upstream" "$revision" "$managed" >>"$TMP/manifest.tsv"

first="$(DEPENDENCY_MANIFEST="$TMP/manifest.tsv" DEPENDENCY_ROOT="$TMP/package-root" \
  "$ROOT/bootstrap-dependencies.sh" codex-shim-sero)"
[[ "$first" == *"created managed commit $managed"* ]]
[[ "$(git -C "$TMP/package-root/deps/codex-shim" rev-parse HEAD)" == "$managed" ]]
[[ -z "$(git -C "$TMP/package-root/deps/codex-shim" status --porcelain=v1 --untracked-files=all)" ]]

second="$(DEPENDENCY_MANIFEST="$TMP/manifest.tsv" DEPENDENCY_ROOT="$TMP/package-root" \
  "$ROOT/bootstrap-dependencies.sh" codex-shim-sero)"
[[ "$second" == *"ready at $managed"* ]]

printf 'dirty\n' >"$TMP/package-root/deps/codex-shim/untracked.txt"
if DEPENDENCY_MANIFEST="$TMP/manifest.tsv" DEPENDENCY_ROOT="$TMP/package-root" \
  "$ROOT/bootstrap-dependencies.sh" codex-shim-sero >"$TMP/dirty.out" 2>"$TMP/dirty.err"; then
  printf 'dirty dependency was accepted\n' >&2
  exit 1
fi
grep -Fq 'is dirty; refusing to modify it' "$TMP/dirty.err"
[[ -f "$TMP/package-root/deps/codex-shim/untracked.txt" ]]

if DEPENDENCY_MANIFEST="$TMP/manifest.tsv" DEPENDENCY_ROOT="$TMP/other-root" \
  "$ROOT/bootstrap-dependencies.sh" unknown >"$TMP/unknown.out" 2>"$TMP/unknown.err"; then
  printf 'unknown dependency was accepted\n' >&2
  exit 1
fi
grep -Fq "unknown dependency 'unknown'" "$TMP/unknown.err"

printf 'bootstrap contracts: passed\n'
