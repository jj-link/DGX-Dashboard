#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
BEFORE="$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)"

"$ROOT/tests/bootstrap.sh"
"$ROOT/tests/dispatcher.sh"
"$ROOT/tests/runtime.sh"
"$ROOT/tests/cluster.sh"
"$ROOT/tests/sync.sh"
"$ROOT/tests/migration.sh"
"$ROOT/tests/repository.sh"

AFTER="$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)"
if [[ "$AFTER" != "$BEFORE" ]]; then
  printf 'test suite changed the checkout\n--- before ---\n%s\n--- after ---\n%s\n' "$BEFORE" "$AFTER" >&2
  exit 1
fi
printf 'aggregate contracts: passed; checkout unchanged\n'
