# Codex-Shim Setup — Summary

> Created: June 18, 2026  
> Environment: WSL (Ubuntu), Python 3.11.15, Codex CLI 0.137.0  
> Shipped: MIT License, by [0xSero](https://github.com/0xSero/codex-shim)

---

## What was done

1. **Cloned** `codex-shim` to `/home/workbench/codex-shim-setup/codex-shim/`
2. **Installed** into a virtualenv at `/home/workbench/codex-shim-setup/venv/` (Python 3.11, aiohttp 3.14.1)
3. **Created** `/home/workbench/codex-shim-setup/models.json` → copied to `~/.codex-shim/models.json`
4. **Generated** the catalog with `codex-shim generate` — 2 BYOK models + 5 ChatGPT passthrough entries
5. **Started/stopped** the daemon to verify it launches cleanly (7 models healthy)
6. **Ran** `codex-shim doctor` — 16 OK, 3 WARN (minor), 0 FAIL

---

## Models configured

| Slug | Display Name | Provider | Endpoint |
|------|-------------|----------|----------|
| `qwen3-6-27b-fp8-dflash` | SGLang Qwen3.6-27B-FP8-DFLASH (GPU0) | generic-chat-completion-api | `http://127.0.0.1:8000/v1` |
| `gemma-4-12b-it-ud-q4-k-xl` | llama.cpp Gemma4-12B MTP (GPU1) | generic-chat-completion-api | `http://127.0.0.1:8001/v1` |

**GPU0 (RTX PRO 6000, 96GB):** SGLang serving Qwen3.6-27B-FP8-DFLASH on port **8000**  
**GPU1 (RTX 4090, 24GB):** llama.cpp serving Gemma4-12B-it-UD-Q4_K_XL with MTP (multi-token prediction) on port **8001**

### llama.cpp note
The llama.cpp server runs in Docker via `/home/workbench/inference/serve/llama.cpp/serve_gemma4_12b_mtp.sh`:
```bash
docker run -d --name llama-gemma4-12b --gpus device=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p 127.0.0.1:8001:8080 \
  -v /home/workbench/models/hub/unsloth/gemma-4-12b-it-GGUF:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/gemma-4-12b-it-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 --port 8080 --n-gpu-layers 999 --cache-ram 0 --no-warmup \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --spec-draft-model /models/MTP/gemma-4-12b-it-F16-MTP.gguf
```

**⚠️ llama.cpp Docker container was NOT running during setup.** Start it before using the Gemma4 model.

---

## Startup commands

### Option A: Full setup (writes config, starts daemon)
```bash
source /home/workbench/codex-shim-setup/venv/bin/activate
cd /home/workbench/codex-shim-setup/codex-shim

# Regenerate catalog + start daemon + wire Codex config
codex-shim enable
```

### Option B: Manual start (doesn't modify ~/.codex/config.toml)
```bash
source /home/workbench/codex-shim-setup/venv/bin/activate
cd /home/workbench/codex-shim-setup/codex-shim

codex-shim generate   # regenerate catalog
codex-shim start      # start daemon on 127.0.0.1:8765
```

### Option C: One-off CLI session
```bash
source /home/workbench/codex-shim-setup/venv/bin/activate
cd /home/workbench/codex-shim-setup/codex-shim

# This starts the daemon, runs Codex, then stops
codex-shim codex -- "summarize this codebase"
```

### Stop the shim
```bash
source /home/workbench/codex-shim-setup/venv/bin/activate
cd /home/workbench/codex-shim-setup/codex-shim
codex-shim stop
```

---

## How to point Codex at the shim

### Method 1: `codex-shim enable` (recommended)
Writes a managed block into `~/.codex/config.toml`:
```toml
model = "gpt-5.5"
model_provider = "codex_shim"
model_catalog_json = "/home/workbench/codex-shim-setup/codex-shim/.codex-shim/custom_model_catalog.json"

[model_providers.codex_shim]
name = "Codex Shim"
base_url = "http://127.0.0.1:8765/v1"
wire_api = "responses"
experimental_bearer_token = "dummy"
```

To revert: `codex-shim disable`

### Method 2: Inline CLI override
```bash
codex -c 'model="qwen3-6-27b-fp8-dflash"' \
      -c 'model_provider="codex_shim"' \
      -c 'model_providers.codex_shim.base_url="http://127.0.0.1:8765/v1"' \
      -c 'model_providers.codex_shim.wire_api="responses"' \
      -c 'model_providers.codex_shim.name="Codex Shim"' \
      -c 'model_providers.codex_shim.experimental_bearer_token="dummy"' \
      .
```

### Method 3: `codex-shim app` (Codex Desktop)
```bash
source /home/workbench/codex-shim-setup/venv/bin/activate
cd /home/workbench/codex-shim-setup/codex-shim
codex-shim app .    # launches Codex Desktop with shim wired in
```

---

## Available model slugs

After `codex-shim generate`:

```
gpt-5.5                    GPT-5.5  ->  chatgpt passthrough
gpt-5.4                    GPT-5.4  ->  chatgpt passthrough
gpt-5.4-mini               GPT-5.4-Mini  ->  chatgpt passthrough
gpt-5.3-codex-spark        GPT-5.3-Codex-Spark  ->  chatgpt passthrough
codex-auto-review          Codex Auto Review  ->  chatgpt passthrough
qwen3-6-27b-fp8-dflash     SGLang Qwen3.6-27B  ->  local:8000 (BYOK)
gemma-4-12b-it-ud-q4-k-xl  llama.cpp Gemma4-12B  ->  local:8001 (BYOK)
```

Set the active model: `codex-shim model use <slug>`

---

## Key files

| Path | Purpose |
|------|---------|
| `~/.codex-shim/models.json` | Model config (source of truth) |
| `/home/workbench/codex-shim-setup/models.json` | Backup/copy of model config |
| `codex-shim/.codex-shim/custom_model_catalog.json` | Generated catalog (don't edit) |
| `codex-shim/.codex-shim/config.toml` | Generated provider config (don't edit) |
| `codex-shim/.codex-shim/shim.pid` | Daemon PID file |
| `codex-shim/.codex-shim/shim.log` | Request log |
| `~/.codex/config.toml` | Codex config (modified by `enable`) |

---

## Warnings from doctor

1. **NO_PROXY**: Set `NO_PROXY="127.0.0.1,localhost,::1"` and `no_proxy="127.0.0.1,localhost,::1"` to prevent system proxies from intercepting loopback traffic to the shim.
2. **Cursor passthrough**: `cursor-agent` not found (only needed if you want Cursor/Composer routing).
3. **Python version**: System Python is 3.10 (insufficient). The venv uses Python 3.11.15 at `/home/workbench/.local/bin/python3.11`.

---

## Common commands reference

```bash
# Activate environment
source /home/workbench/codex-shim-setup/venv/bin/activate
cd /home/workbench/codex-shim-setup/codex-shim

codex-shim doctor            # diagnostics report
codex-shim generate          # regenerate catalog from models.json
codex-shim start             # start background daemon on :8765
codex-shim status            # health check + model count
codex-shim list              # show all slugs and routes
codex-shim stop              # stop daemon
codex-shim restart           # stop + regenerate + start
codex-shim enable            # start + wire ~/.codex/config.toml
codex-shim disable           # remove config block + stop
codex-shim model list        # available model slugs
codex-shim model use <slug>  # set active model
codex-shim app [path]        # launch Codex Desktop with shim
codex-shim codex -- <cmd>    # one-off CLI session through shim
```
