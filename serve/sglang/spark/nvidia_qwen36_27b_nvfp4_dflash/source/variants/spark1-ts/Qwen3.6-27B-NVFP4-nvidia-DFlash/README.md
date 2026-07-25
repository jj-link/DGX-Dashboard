# Qwen3.6-27B-NVFP4-DFlash on SGLang

This runtime serves `unsloth/Qwen3.6-27B-NVFP4` with SGLang and uses
`z-lab/Qwen3.6-27B-DFlash` as the speculative draft model.

## Files

- `docker-compose.yml`: SGLang server definition.
- `.env`: local secrets and runtime values.
- `.env.example`: template for `.env`.
- `scripts/up.sh`: start the server in the foreground.
- `scripts/down.sh`: stop and remove the server container.
- `scripts/logs.sh`: follow server logs.
- `scripts/test.sh`: smoke-test the OpenAI-compatible API.

## First Run

Edit `.env` and set a valid Hugging Face token:

```bash
HF_TOKEN=hf_...
```

Then run:

```bash
./scripts/up.sh
```

In another shell:

```bash
./scripts/test.sh
```
