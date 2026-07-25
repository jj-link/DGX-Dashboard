#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
python3 "$ROOT/tests/test_recipe_contracts.py"
exec "$ROOT/scripts/audit-repository.sh"
