#!/usr/bin/env bash
# Launch script for RedHatAI/diffusiongemma-27B-A4B-it-NVFP4 on DGX Spark
# Usage: ./launch-diffusiongemma.sh [--download-only]
#   --download-only: just download the model and print the snapshot path, don't start the server

set -euo pipefail

MODEL_NAME="RedHatAI/diffusiongemma-27B-A4B-it-NVFP4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="$SCRIPT_DIR/docker-compose.yml"

echo "=== diffusiongemma-27B-A4B-it-NVFP4 Launch ==="
echo "Model: $MODEL_NAME"
echo "Compose: $COMPOSE"
echo ""

cd "$SCRIPT_DIR"

# Download model if needed
download_model() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set. Run: export HF_TOKEN=<your-token>" >&2
    exit 1
  fi

  echo "Downloading model from HuggingFace..."
  mkdir -p "$HOME/models/hub/models--${MODEL_NAME//\//-}"
  
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$MODEL_NAME" \
      --local-dir "$HOME/models/hub/models--${MODEL_NAME//\//-}" \
      --resume-download 2>&1 | tail -5
  else
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
  '$MODEL_NAME',
  local_dir='$HOME/models/hub/models--${MODEL_NAME//\//-}',
  resume_download=True
)
"
  fi

  # Get snapshot commit
  SNAPSHOT_DIR="$HOME/models/hub/models--${MODEL_NAME//\//-}"
  MODEL_ID=$(cat "$SNAPSHOT_DIR/.git/refs/snapshot" 2>/dev/null || \
             git -C "$SNAPSHOT_DIR" rev-parse HEAD 2>/dev/null || \
             echo "UNKNOWN")
  
  echo "Model downloaded. Snapshot: $MODEL_ID"
  echo "MODEL_ID=$MODEL_ID"
}

start_server() {
  local model_id="${1:-}"
  
  if [[ -z "$model_id" ]]; then
    SNAPSHOT_DIR="$HOME/models/hub/models--${MODEL_NAME//\//-}"
    model_id=$(cat "$SNAPSHOT_DIR/.git/refs/snapshot" 2>/dev/null || \
               git -C "$SNAPSHOT_DIR" rev-parse HEAD 2>/dev/null || \
               echo "UNKNOWN")
  fi
  
  echo "Starting sglang server with snapshot: $model_id"
  export MODEL_ID="$model_id"
  
  docker compose up -d
  echo "Server starting. Check logs with:"
  echo "  docker compose logs -f"
  echo "Health check: curl http://localhost:8001/v1/models"
}

# Main
case "${1:-}" in
  --download-only)
    download_model
    ;;
  *)
    # Check if model exists, download if not
    SNAPSHOT_DIR="$HOME/models/hub/models--${MODEL_NAME//\//-}"
    if [[ ! -d "$SNAPSHOT_DIR" ]] || [[ ! -f "$SNAPSHOT_DIR/config.json" ]]; then
      download_model
    fi
    start_server
    ;;
esac

echo "=== Launch complete ==="
