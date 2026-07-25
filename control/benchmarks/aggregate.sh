#!/usr/bin/env bash
# Collect aider stats from every per-model run dir into a flat comparison table.
# Optionally include BFCL accuracy if present.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=spark-serve/models.env
source "${SCRIPT_DIR}/spark-serve/models.env"

cd "${SCRIPT_DIR}/aider"
# shellcheck disable=SC1091
source .venv/bin/activate

printf "%-26s | %-8s | %-8s | %-9s | %-9s | %s\n" "model" "pass@1" "pass@2" "wellform%" "tests" "edit-fmt"
printf "%-26s-+-%-8s-+-%-8s-+-%-9s-+-%-9s-+-%s\n" "$(printf '%0.s-' {1..26})" "--------" "--------" "---------" "---------" "--------"

for alias in "${ALL_ALIASES[@]}"; do
  RUN_DIR=$(ls -td tmp.benchmarks/*--"${alias}"-* 2>/dev/null | head -1 || true)
  if [ -z "$RUN_DIR" ]; then
    printf "%-26s | %-8s | %-8s | %-9s | %-9s | %s\n" "$alias" "—" "—" "—" "—" "(no run)"
    continue
  fi

  STATS="$RUN_DIR/_stats.yml"
  if [ ! -f "$STATS" ]; then
    ./benchmark/benchmark.py --stats "$RUN_DIR" > "$STATS" 2>&1 || true
  fi

  p1=$(grep -E '^\s*pass_rate_1:' "$STATS" 2>/dev/null | head -1 | awk '{print $2}')
  p2=$(grep -E '^\s*pass_rate_2:' "$STATS" 2>/dev/null | head -1 | awk '{print $2}')
  wf=$(grep -E '^\s*percent_cases_well_formed:' "$STATS" 2>/dev/null | head -1 | awk '{print $2}')
  tc=$(grep -E '^\s*test_cases:' "$STATS" 2>/dev/null | head -1 | awk '{print $2}')
  ef=$(grep -E '^\s*edit_format:' "$STATS" 2>/dev/null | head -1 | awk '{print $2}')
  printf "%-26s | %-8s | %-8s | %-9s | %-9s | %s\n" "$alias" "${p1:-—}" "${p2:-—}" "${wf:-—}" "${tc:-—}" "${ef:-—}"
done
