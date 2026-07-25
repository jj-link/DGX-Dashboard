#!/usr/bin/env bash
set -euo pipefail

CONTROL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$CONTROL_ROOT/.." && pwd -P)"

usage() {
  cat <<'EOF'
usage:
  ./benchmark.sh local   oneshot [oneshot args...]
  ./benchmark.sh spark1  oneshot [oneshot args...]
  ./benchmark.sh spark2  oneshot [oneshot args...]
  ./benchmark.sh spark3  oneshot [oneshot args...]
  ./benchmark.sh cluster oneshot [oneshot args...]

Runs the oneshot client and Docker grading on the workstation against the
model already serving on the selected target. Omitting --lang runs all six
languages: cpp, go, java, javascript, python, and rust.

oneshot args:
  --lang {cpp,go,java,javascript,python,rust}
  --served-model NAME
  --num-tests N                 default: -1 (all)
  --keywords NAME,...
  --max-tokens N                default: 32768
  --temperature FLOAT           default: 1.0
  --timeout SEC                 default: 600
  --test-timeout SEC            default: 300
  --concurrency N               default: 1
  --out-dir PATH                default: /var/lib/dgx-dashboard/benchmark-results
  --backend VALUE               --quant VALUE
  --kv-cache-type VALUE         --spec-decode VALUE
  --hardware VALUE              --context-length N
  --tp-size N                   --reasoning VALUE
  --reasoning-effort {low,medium,high}
  --engine-version VALUE        --runtime-image VALUE
  --runtime-image-digest VALUE  --model-source VALUE
  --model-revision VALUE

Single-device services default to HOST_PORT=8000. The cluster uses port 8888.
EOF
}

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  [[ $# == 1 ]] || { usage >&2; exit 2; }
  usage
  exit 0
fi

(( $# >= 2 )) || { usage >&2; exit 2; }
target="$1"
benchmark="$2"
shift 2
[[ "$benchmark" == oneshot ]] || { usage >&2; exit 2; }

port="${HOST_PORT:-8000}"
[[ "$port" =~ ^[1-9][0-9]{0,4}$ && "$port" -le 65535 ]] || fail "HOST_PORT must be between 1 and 65535"

host=''
case "$target" in
  local)
    endpoint="http://127.0.0.1:$port/v1"
    ;;
  spark1) host=spark1-ts ;;
  spark2) host=spark2-ts ;;
  spark3) host=spark3-ts ;;
  cluster)
    host=spark2-ts
    [[ -z "${HOST_PORT+x}" ]] || fail "HOST_PORT is only supported for local and Spark single-device targets"
    port=8888
    ;;
  *) usage >&2; exit 2 ;;
esac

run_id="${DGX_DASHBOARD_RUN_ID:-}"
if [[ -z "$run_id" ]]; then
  run_id="$(python3 -c 'import uuid; print(uuid.uuid4())')"
fi
[[ "$run_id" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
  fail "DGX_DASHBOARD_RUN_ID must be a lowercase UUID"

if [[ -n "$host" ]]; then
  address_output=''
  ssh_rc=0
  address_output="$(ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ForwardAgent=no -o ClearAllForwardings=yes -o RequestTTY=no -o ConnectTimeout=20 -o ConnectionAttempts=3 -o ServerAliveInterval=60 -o ServerAliveCountMax=30 "$host" 'exec tailscale ip -4')" || ssh_rc=$?
  (( ssh_rc == 0 )) || exit "$ssh_rc"
  mapfile -t addresses < <(printf '%s\n' "$address_output" | sed '/^[[:space:]]*$/d')
  (( ${#addresses[@]} == 1 )) || fail "$host must return exactly one Tailscale IPv4"
  [[ "${addresses[0]}" =~ ^100\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "$host returned unexpected Tailscale IPv4 '${addresses[0]}'"
  endpoint="http://${addresses[0]}:$port/v1"
fi

if [[ -n "$host" ]]; then
  printf 'target=%s host=%s benchmark=oneshot endpoint=%s run_id=%s\n' "$target" "$host" "$endpoint" "$run_id" >&2
else
  printf 'target=%s benchmark=oneshot endpoint=%s run_id=%s\n' "$target" "$endpoint" "$run_id" >&2
fi

export OPENAI_API_BASE="$endpoint"
export OPENAI_API_KEY=dummy
export DGX_DASHBOARD_BENCHMARK_TARGET="$target"
export DGX_DASHBOARD_RUN_ID="$run_id"
exec python3 "$CONTROL_ROOT/benchmarks/oneshot_bench.py" "$@"
