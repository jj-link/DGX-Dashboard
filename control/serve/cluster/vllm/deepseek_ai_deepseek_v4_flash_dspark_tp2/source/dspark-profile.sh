#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ACTION="${1:-}"
PROFILE="${2:-}"

if [[ ! "$ACTION" =~ ^(start|stop|status|logs|verify)$ ]] || [[ ! "$PROFILE" =~ ^(balanced|quality|throughput)$ ]]; then
  printf 'usage: %s <start|stop|status|logs|verify> <balanced|quality|throughput>\n' "$0" >&2
  exit 2
fi

exec env CLUSTER_PROFILE="$PROFILE" "$SCRIPT_DIR/../$ACTION.sh"
