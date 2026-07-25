# codex-shim Feature Idea: Provider Abstraction

## Current State
codex-shim uses a flat `models.json` where each model entry specifies its own `base_url`:
```json
{
  "models": [
    {
      "model": "Qwen3.6-27B-FP8-DFLASH",
      "provider": "generic-chat-completion-api",
      "base_url": "http://127.0.0.1:8000/v1",
      "api_key": "***"
    }
  ]
}
```

## Hermes Agent's Better Pattern
Hermes Agent separates **providers** (infrastructure) from **models** (what runs on them):
```yaml
providers:
  rtx_6000_pro:
    name: RTX 6000 Pro
    api: http://localhost:8000/v1
    context_length: 262144
    default_model: Qwen3.6-27B-FP8-DFLASH
```

Benefits:
- Define infrastructure once, add models without repeating URLs
- Model discovery from `/v1/models` endpoint
- Switch models without editing config
- Cleaner separation of concerns

## Proposed Enhancement for codex-shim
Add a `providers` block to `models.json`:
```json
{
  "providers": {
    "sglang-gpu0": {
      "base_url": "http://127.0.0.1:8000/v1",
      "api_key": "***"
    }
  },
  "models": [
    {
      "model": "Qwen3.6-27B-FP8-DFLASH",
      "provider_ref": "sglang-gpu0",
      "display_name": "Qwen3.6-27B"
    }
  ]
}
```

Or support auto-discovery:
```json
{
  "providers": {
    "sglang": {
      "base_url": "http://127.0.0.1:8000/v1",
      "discover_models": true
    }
  }
}
```

## Repo
https://github.com/sybil-solutions/codex-shim
