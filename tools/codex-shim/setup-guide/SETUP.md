# Codex Local Setup — Using OpenAI Codex CLI with Local Models

## Overview

**codex-shim** (`https://github.com/0xSero/codex-shim`) is a local Python/aiohttp proxy server that translates between OpenAI's Codex Responses API and any upstream model backend (OpenAI chat completions, Anthropic Messages, or generic OpenAI-compatible endpoints). This lets you run Codex Desktop and Codex CLI against local models like SGLang, llama.cpp, Ollama, or any API-key-based provider.

### Architecture

```
Codex CLI/Desktop ── /v1/responses ──▶ codex-shim (127.0.0.1:8765)
                                  │
                                  ├── ChatGPT passthrough (gpt-5.5)
                                  │       └─▶ chatgpt.com/backend-api/codex/responses
                                  │
                                  ├── OpenAI chat completions
                                  │       └─▶ base_url/chat/completions
                                  │
                                  └── Anthropic Messages
                                          └─▶ base_url/messages
```

The shim:
- Runs on `127.0.0.1:8765` by default (loopback only — not Internet-facing)
- Translates streaming Responses API ↔ chat completions / Anthropic Messages
- Preserves tool calls, reasoning blocks, parallel function calls, and image inputs
- Supports an optional smart auto-router for cost-optimized per-task routing

---

## Installation

### 1. Clone and install codex-shim

```bash
git clone https://github.com/0xSero/codex-shim ~/codex-shim
cd ~/codex-shim
python3 -m pip install --user -e .
```

This installs the `codex-shim` command and the `aiohttp` dependency.

### 2. Make sure codex CLI is installed

```bash
npm install -g @openai/codex
# or check existing:
which codex && codex --version
```

### 3. Optional: symlink helper scripts

```bash
mkdir -p ~/.local/bin
ln -sf "$PWD/bin/codex-app" ~/.local/bin/codex-app
ln -sf "$PWD/bin/codex-model" ~/.local/bin/codex-model
export PATH="$HOME/.local/bin:$PATH"
```

---

## Configuration

### Create `~/.codex-shim/models.json`

This file defines which models the shim can route to:

```json
{
  "models": [
    {
      "model": "Qwen3.6-27B-FP8-DFLASH",
      "display_name": "SGLang Qwen3.6-27B-FP8 (GPU0)",
      "provider": "generic-chat-completion-api",
      "base_url": "http://127.0.0.1:8000/v1",
      "api_key": "your-api-key",
      "max_context_limit": 131072,
      "no_image_support": true
    },
    {
      "model": "gemma-4-12b-it-UD-Q4_K_XL",
      "display_name": "llama.cpp Gemma4-12B MTP (GPU1)",
      "provider": "generic-chat-completion-api",
      "base_url": "http://127.0.0.1:8001/v1",
      "api_key": "your-api-key",
      "max_context_limit": 8192,
      "no_image_support": true
    }
  ]
}
```

### Supported provider values

| Provider | Upstream API |
|---|---|
| `openai` | OpenAI `/v1/chat/completions` |
| `generic-chat-completion-api` | Any OpenAI-shaped chat completions endpoint (SGLang, llama.cpp, Ollama, LiteLLM, vLLM, etc.) |
| `anthropic` | Anthropic `/v1/messages` |

### Model fields

| Field | Description |
|---|---|
| `model` | Model ID sent to the upstream (required) |
| `display_name` | Human-readable name shown in Codex picker |
| `provider` | One of `openai`, `generic-chat-completion-api`, `anthropic` |
| `base_url` | Full URL to the upstream API base (required) |
| `api_key` | API key for the upstream (use `api_key_env` for env var reference) |
| `max_context_limit` | Context window limit in tokens |
| `no_image_support` | Set `true` for text-only models so Codex doesn't send images |
| `extra_headers` | Extra headers forwarded to the upstream |

### Environment variable for API keys

Use `api_key_env` instead of `api_key` to reference an environment variable:

```json
{
  "model": "my-model",
  "provider": "generic-chat-completion-api",
  "base_url": "http://127.0.0.1:8000/v1",
  "api_key_env": "LOCAL_API_KEY"
}
```

---

## Running the shim

### Start the daemon

```bash
cd ~/codex-shim
codex-shim generate   # Generate catalog and config files
codex-shim start      # Start background daemon on 127.0.0.1:8765
codex-shim list       # Show available model slugs
codex-shim status     # Health check
```

### Point Codex at the shim

#### Option A: Enable shim globally (modifies `~/.codex/config.toml`)

```bash
codex-shim enable              # Writes managed config block
codex-shim model use qwen3-6-27b-fp8-dflash  # Set default model
codex-shim app .               # Launch Codex Desktop
```

To revert:
```bash
codex-shim disable
```

#### Option B: One-off CLI usage (no config modification)

```bash
codex-shim codex -- "inspect this repo and summarize the architecture"
```

#### Option C: Manual config override

Edit `~/.codex/config.toml`:
```toml
[model_providers.codex_shim]
url = "http://127.0.0.1:8765/v1"
default_model = "qwen3-6-27b-fp8-dflash"

[model_providers.codex_shim.capabilities]
wire_api = "responses"
```

### Managing models

```bash
codex-model list                    # Show available models
codex-model qwen3-6-27b-fp8-dflash  # Switch to this model
codex-app                           # Relaunch Codex Desktop
```

### Stopping

```bash
codex-shim stop      # Stop the daemon
codex-shim disable   # Remove config changes + stop daemon
```

---

## Local model backends

### SGLang (port 8000)

SGLang exposes an OpenAI-compatible API. Configure it as:

```json
{
  "model": "<your-sglang-model-id>",
  "display_name": "SGLang Model",
  "provider": "generic-chat-completion-api",
  "base_url": "http://127.0.0.1:8000/v1",
  "api_key": "your-key"
}
```

### llama.cpp server (port 8001)

llama.cpp's built-in server also speaks OpenAI-compatible chat completions:

```json
{
  "model": "<your-llama-cpp-model-id>",
  "display_name": "llama.cpp Model",
  "provider": "generic-chat-completion-api",
  "base_url": "http://127.0.0.1:8001/v1",
  "api_key": "your-key"
}
```

### Ollama (port 11434)

```json
{
  "model": "llama3.2",
  "display_name": "Ollama Llama 3.2",
  "provider": "generic-chat-completion-api",
  "base_url": "http://127.0.0.1:11434/v1",
  "api_key": "ollama"
}
```

Or even shorter — the shim auto-detects Ollama:

```json
{ "models": [ "llama3.2" ] }
```

### LiteLLM proxy

If you're using LiteLLM as a local proxy, configure each backend as:

```json
{
  "model": "litellm-model-name",
  "display_name": "Model via LiteLLM",
  "provider": "generic-chat-completion-api",
  "base_url": "http://127.0.0.1:4000/v1",
  "api_key": "sk-litellm"
}
```

---

## Debugging and troubleshooting

### Diagnostics

```bash
codex-shim doctor   # Read-only diagnostic report (Python, deps, Codex, settings, health)
codex-shim status   # Health probe + model count
```

### Logs

```bash
tail -f .codex-shim/shim.log    # Request summaries (no API keys in logs)
cat ~/.codex-shim/models.json   # Your settings file
```

### Model picker (web UI)

Open `http://127.0.0.1:8765/picker` in your browser to see available models and switch between them.

### Common issues

| Problem | Fix |
|---|---|
| Shim won't start | Run `codex-shim doctor`, check if port 8765 is in use |
| `codex-shim list` says "No models" | Create `~/.codex-shim/models.json` or run `codex login` for ChatGPT passthrough |
| Codex shows only "default" | Run `codex-shim generate`, then `codex-shim enable` |
| Model 404 error | Regenerate after editing settings: `codex-shim generate` |
| Upstream 401/403 | Check API key in models.json or env var |
| Tool calls become text | The upstream model may not support tool calls well; check shim.log |
| Streaming hangs | Verify upstream streams outside Codex, then `codex-shim restart` |
| Port 8765 busy | Use `codex-shim --port 8766 start` and `codex-shim --port 8766 app .` |

---

## Alternative approaches (without codex-shim)

### Using `--api-host` with Codex CLI

The Codex CLI accepts environment variables and config overrides for API routing. You can set:

```toml
# ~/.codex/config.toml
[model_providers.local]
url = "http://127.0.0.1:8000/v1"
```

However, this only works if your local endpoint speaks the **Responses API** natively. Most local servers (SGLang, llama.cpp, Ollama) only speak chat completions — which is why `codex-shim` exists to translate.

### LiteLLM as a standalone proxy

LiteLLM can expose a Responses-like endpoint, but it doesn't translate between Responses and chat completions the way codex-shim does. You would need:
1. A LiteLLM proxy on one port
2. Codex configured to talk to it
3. Manual translation of the request/response format

`codex-shim` handles all of this automatically.

### Direct llama.cpp / SGLang API calls

Without any shim, you can interact with local models directly via HTTP/curl, but you lose the Codex agent loop (tool calls, shell execution, file editing, multi-step reasoning).

---

## Commands reference

```
codex-shim generate                    # Regenerate catalog/config
codex-shim start                       # Start background daemon
codex-shim enable                      # Start daemon + write config
codex-shim status                      # Health check
codex-shim doctor                      # Diagnostics report
codex-shim stop                        # Stop daemon
codex-shim disable                     # Remove config + stop daemon
codex-shim restart                     # Stop + regenerate + start
codex-shim list                        # List model slugs and routes
codex-shim model list                  # List picker-available slugs
codex-shim model use <slug>            # Set default model
codex-shim codex -- <args>             # One-off CLI run
codex-shim app [path]                  # Launch Codex Desktop

codex-app [path]                       # Shortcut for codex-shim app
codex-model [list|<slug>]              # Shortcut for codex-shim model
```

### Global flags
- `--settings <path>` — Use a custom settings file instead of `~/.codex-shim/models.json`
- `--port <port>` — Use a custom port instead of 8765

---

## License

MIT — see the codex-shim repo. Codex Desktop is a trademark of OpenAI. This project is unaffiliated.
