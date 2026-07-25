#!/usr/bin/env bash
set -euo pipefail

echo "=== diffusiongemma-26B-A4B-it-NVFP4 Launch ==="

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# Load .env for HF_TOKEN
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  export $(grep -v "^#" "$SCRIPT_DIR/.env" | xargs)
fi

echo "Stopping any existing containers..."
docker compose down >/dev/null 2>&1 || true

echo "Starting server (model will be downloaded inside container if not present)..."
docker compose up -d

echo ""
echo "Done. Check logs:"
echo "  docker compose logs -f"
echo "Health check:"
echo "  curl http://localhost:8000/v1/models"
