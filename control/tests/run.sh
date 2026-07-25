#!/usr/bin/env bash
set -euo pipefail
CONTROL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
REPO_ROOT="$(cd "$CONTROL_ROOT/.." && pwd -P)"
BEFORE="$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)"

while IFS= read -r -d '' script; do
  bash -n "$REPO_ROOT/$script"
done < <(git -C "$REPO_ROOT" ls-files -z -- '*.sh')

(cd "$REPO_ROOT" && python3 -m pytest -q)
"$CONTROL_ROOT/tests/bootstrap.sh"
"$CONTROL_ROOT/tests/dispatcher.sh"
"$CONTROL_ROOT/tests/runtime.sh"
"$CONTROL_ROOT/tests/cluster.sh"
"$CONTROL_ROOT/tests/sync.sh"
"$CONTROL_ROOT/tests/migration.sh"
"$CONTROL_ROOT/tests/repository.sh"

AFTER="$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)"
if [[ "$AFTER" != "$BEFORE" ]]; then
  printf 'test suite changed the checkout\n--- before ---\n%s\n--- after ---\n%s\n' "$BEFORE" "$AFTER" >&2
  exit 1
fi
printf 'aggregate contracts: passed; checkout unchanged\n'
