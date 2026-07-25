#!/usr/bin/env bash
set -euo pipefail
which_part=${1:-head}
lines=${2:-100}
WORKER_SSH_TARGET=${WORKER_SSH_TARGET:-jjlink@10.0.0.2}
SSH_KEY=${SSH_KEY:-/home/jjlink/.ssh/gb10_ed25519}
ssh_worker=(ssh -i "${SSH_KEY}" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "${WORKER_SSH_TARGET}")
case "${which_part}" in
  head)
    docker exec hy3-head tail -n "${lines}" /tmp/hy3-serve.log 2>/dev/null || docker logs --tail "${lines}" hy3-head
    ;;
  worker)
    "${ssh_worker[@]}" "docker logs --tail '${lines}' hy3-worker"
    ;;
  both)
    echo "== head serve log =="; docker exec hy3-head tail -n "${lines}" /tmp/hy3-serve.log 2>/dev/null || docker logs --tail "${lines}" hy3-head
    echo "== worker container log =="; "${ssh_worker[@]}" "docker logs --tail '${lines}' hy3-worker"
    ;;
  *) echo "usage: $0 [head|worker|both] [lines]" >&2; exit 2 ;;
esac
