#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
docker compose rm -f qwen36-a3b-dflash
docker compose up
