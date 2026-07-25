#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ENABLE_DFLASH=1 exec ./start.sh
