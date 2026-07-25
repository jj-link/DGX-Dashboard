#!/usr/bin/env bash
# Quick-start script for codex-shim with local models
# Run from: /home/workbench/codex-local-setup/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CONTROL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
CODEX_SHIM_REPO="$CONTROL_ROOT/tools/codex-shim/setup/codex-shim"
SETTINGS_FILE="$HOME/.codex-shim/models.json"

echo "=== Codex Local Setup ==="
echo ""

# 1. Check Python
if ! command -v python3 &>/dev/null; then
    echo "FAIL: python3 not found"
    exit 1
fi
echo "[OK] python3 found: $(python3 --version)"

# 2. Check codex-shim repo
if [ -d "$CODEX_SHIM_REPO" ]; then
    echo "[OK] codex-shim repo at: $CODEX_SHIM_REPO"
else
    echo "INFO: Cloning codex-shim..."
    git clone https://github.com/0xSero/codex-shim "$CODEX_SHIM_REPO"
fi

# 3. Install codex-shim
if ! command -v codex-shim &>/dev/null; then
    echo "INFO: Installing codex-shim..."
    cd "$CODEX_SHIM_REPO" && python3 -m pip install --user -e .
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[OK] codex-shim installed: $(which codex-shim)"

# 4. Create settings if missing
if [ ! -f "$SETTINGS_FILE" ]; then
    mkdir -p "$HOME/.codex-shim"
    cp "$SCRIPT_DIR/example-models.json" "$SETTINGS_FILE"
    echo "[OK] Created $SETTINGS_FILE from example"
    echo "     -> Edit it with your actual API keys before proceeding"
else
    echo "[OK] Settings file exists: $SETTINGS_FILE"
fi

# 5. Check codex CLI
if command -v codex &>/dev/null; then
    echo "[OK] codex CLI: $(codex --version 2>/dev/null || echo 'installed')"
else
    echo "WARN: codex CLI not found. Install with: npm install -g @openai/codex"
fi

echo ""
echo "=== Next steps ==="
echo "1. Edit $SETTINGS_FILE with your API keys"
echo "2. Make sure SGLang (port 8000) and llama.cpp (port 8001) are running"
echo "3. Run: cd $CODEX_SHIM_REPO"
echo "4. Run: codex-shim generate && codex-shim start && codex-shim list"
echo "5. Run: codex-shim enable && codex-shim model use qwen3-6-27b-fp8-dflash"
echo ""
