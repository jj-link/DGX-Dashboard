#!/usr/bin/env bash
# Build all polybench-<lang> images with toolchain + deps baked in.
# Run once (and whenever a Dockerfile changes). Each image lets
# oneshot_bench.py run that language's tests offline with no per-run install.
set -uo pipefail   # NOT -e: one language failing must not skip the rest
cd "$(dirname "${BASH_SOURCE[0]}")"

LANGS=("${@:-python javascript go rust java cpp}")
read -ra LANGS <<< "${LANGS[*]}"

declare -a OK=() FAIL=()
for lang in "${LANGS[@]}"; do
  echo "=== building polybench-${lang} ==="
  if docker build -f "Dockerfile.${lang}" -t "polybench-${lang}:latest" .; then
    OK+=("$lang")
  else
    echo "!!! polybench-${lang} FAILED (continuing) !!!"
    FAIL+=("$lang")
  fi
done

echo
echo "built OK:   ${OK[*]:-(none)}"
echo "FAILED:     ${FAIL[*]:-(none)}"
docker image ls --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' | grep '^polybench-' | sort
[ ${#FAIL[@]} -eq 0 ] || exit 1
