#!/usr/bin/env bash
# Dispatcher: pick an engine + a short script name, launch in the hardened
# container. The easy front door to run_vllm_docker.sh / run_sglang_docker.sh.
#
#   ./serve.sh 27b_fp8_dflash            # engine defaults to vllm
#   ./serve.sh vllm 27b_fp8_dflash
#   ./serve.sh sglang 27b_fp8_dflash
#   DETACH=1 ./serve.sh sglang 27b_fp8_dflash    # env passes straight through
#
# "short name" = any unique fragment of a serve/<engine>/serve_*.sh filename;
#   27b_fp8_dflash -> serve/vllm/serve_qwen36_27b_fp8_dflash.sh
# A full path (serve/<engine>/...sh) also works and picks the engine from it.
# (Named serve.sh, not 'serve', because the serve/ directory owns that name.)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat >&2 <<'EOF'
usage: serve.sh [vllm|sglang] <name-or-path> [extra args...]

Examples:
  ./serve.sh qwen36_27b_fp8_dflash
  ./serve.sh vllm qwen36_27b_fp8_dflash
  ./serve.sh sglang qwen36_27b_fp8_dflash
EOF
}

default_var() {
  local file="$1" var="$2"
  sed -nE "s/^[[:space:]]*${var}=\"\\\$\\{${var}:-([^}]*)\\}\".*/\\1/p" "$file" | head -n1
}

list_available() {
  local engine="$1"
  local dir="serve/${engine}" f alias model drafter details
  echo "available ${engine} scripts:" >&2
  for f in "${ROOT}/${dir}/"serve_*.sh; do
    [ -e "$f" ] || continue
    alias="$(basename "$f" .sh | sed 's/^serve_//')"
    model="$(default_var "$f" MODEL)"
    drafter="$(default_var "$f" DRAFTER)"
    details="$model"
    [ -n "$drafter" ] && details="${details} + ${drafter}"
    if [ -n "$details" ]; then
      printf '  %-28s %s\n' "$alias" "$details" >&2
    else
      printf '  %s\n' "$alias" >&2
    fi
  done
}

case "${1:-}" in
  ""|-h|--help)
    usage
    echo >&2
    list_available vllm
    echo >&2
    list_available sglang
    exit 0
    ;;
esac

ENGINE="vllm"
case "${1:-}" in
  vllm|sglang) ENGINE="$1"; shift ;;
esac
NAME="${1:?usage: serve.sh [vllm|sglang] <name-or-path> [extra args...]}"; shift || true

case "${NAME}" in
  -h|--help)
    usage
    echo >&2
    list_available "$ENGINE"
    exit 0
    ;;
esac

if [ -f "${ROOT}/${NAME}" ]; then               # explicit path: use as-is, infer engine from it
  SCRIPT="${NAME}"
  case "${SCRIPT}" in
    serve/sglang/*) ENGINE="sglang" ;;
    serve/vllm/*)   ENGINE="vllm" ;;
  esac
else                                             # resolve short name within serve/<engine>/
  dir="serve/${ENGINE}"
  shopt -s nullglob
  matches=( "${ROOT}/${dir}/"*"${NAME}"*.sh )
  if (( ${#matches[@]} == 0 )); then
    echo "no ${ENGINE} script matches '${NAME}'." >&2
    list_available "$ENGINE"
    exit 1
  fi
  if (( ${#matches[@]} > 1 )); then              # disambiguate: prefer an exact suffix match
    exact=(); for f in "${matches[@]}"; do [[ "$(basename "$f")" == *"${NAME}.sh" ]] && exact+=("$f"); done
    if (( ${#exact[@]} == 1 )); then matches=("${exact[@]}")
    else
      echo "'${NAME}' is ambiguous in ${dir}:" >&2
      for f in "${matches[@]}"; do echo "  $(basename "$f" .sh | sed 's/^serve_//')" >&2; done
      exit 1
    fi
  fi
  SCRIPT="${dir}/$(basename "${matches[0]}")"
fi

echo ">> ${ENGINE}: ${SCRIPT}" >&2
exec "${ROOT}/run_${ENGINE}_docker.sh" "${SCRIPT}" "$@"
