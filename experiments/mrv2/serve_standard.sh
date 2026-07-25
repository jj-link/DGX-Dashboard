#!/usr/bin/env bash
set -euo pipefail
export REJECTION_SAMPLE_METHOD=standard
exec "$(dirname "$0")/serve_mrv2.sh"
