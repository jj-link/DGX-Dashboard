# Comprehensive Survey: Routing Codex CLI to Local LLM Models

**Date:** 2026-06-18  
**Purpose:** Compare ALL available tools for routing OpenAI Codex CLI to local LLM models.

## 1. Purpose-Built Codex Shims (Direct Solutions)

### 1A. codex-shim by 0xSero (sybil-solutions/codex-shim)

- **GitHub:** https://github.com/sybil-solutions/codex-shim
- **Stars:** ~963
- **Language:** Python 3.11+ (aiohttp)
- **License:** MIT
- **Last commit:** 2026-06-18 (very active, 20+ recent commits)

**What it does:** Full-featured local proxy server exposing OpenAI Responses-compatible endpoints. Codex points at the shim; the shim routes to matching upstream and translates streaming responses back. Also patches Codex Desktop model picker (macOS) to show BYOK models.

**Supported backends:**
- OpenAI-compatible Chat Completions (SGLang, vLLM, Ollama, LM Studio, LocalAI, llama.cpp server)
- Anthropic Messages API
- ChatGPT passthrough (uses ~/.codex/auth.json tokens)
- Cursor/Composer passthrough (via cursor-agent login)
- Auto Router (AI-classifier-based smart routing between models)

**Setup:**
```
git clone https://github.com/sybil-solutions/codex-shim ~/.codex-shim
cd ~/.codex-shim && pip install --user -e .
codex-shim generate  # creates models.json + config
codex-shim start
codex-shim link      # links ~/.codex/config.toml
```
Config: `~/.codex-shim/models.json` (JSON format)

**Key features:**
- Full streaming SSE (token-level realtime)
- Tool calling: function_call, apply_patch, web_search, computer_use
- Responses-to-Chat-to-Anthropic full bidirectional translation
- ChatGPT/Cursor subscription passthrough (no extra API keys)
- Auto Router — per-task classifier-based model selection
- Context compaction (/v1/responses/compact)
- Web-based model picker UI (GET /picker)
- Image support (input_image, computer_call_output, visual feedback)
- Host header validation (DNS rebinding protection)
- Access logs with trace IDs
- codex-shim doctor diagnostics
- Windows, macOS, Linux, WSL support

**Known limitations:**
- macOS picker patch requires codesign (macOS only)
- BYOK models lack native Codex-hosted tools (function-call fallbacks)
- Image generation requires ChatGPT passthrough
- Python-based (not as performant as Rust/C++ at extreme throughput)

**VERDICT: BEST OVERALL** — most feature-complete, actively maintained, handles most edge cases, multi-backend support.
