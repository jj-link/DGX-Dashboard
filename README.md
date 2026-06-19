# DGX Spark Dashboard

Live monitoring dashboard for DGX Spark (GB10) — GPU hardware stats + inference server metrics.

## Features

- **GPU stats**: utilization gauge, memory usage, temperature, power draw, clock speeds
- **Inference servers**: supports SGLang, vLLM, llama.cpp — shows throughput, latency, queue depth, KV cache, model info
- **System**: CPU load, RAM usage, uptime
- **Auto-refresh**: configurable poll interval (default 3s)
- **Dark theme**: terminal-friendly, no external dependencies
- **Benchmarks dashboard**: 5 view modes for benchmark result analysis (see below)

## Quick start

```bash
cd dgx-dashboard
pip install -r requirements.txt

# Edit config.ini — add your inference servers
nano config.ini

# Run
python server.py
```

Open `http://<spark-ip>:9000` in your browser.

## Config

Edit `config.ini`:

```ini
[server]
host = 0.0.0.0
port = 9000
refresh_interval = 3
auth_user = admin          # optional
auth_password = changeme   # optional

[inference_servers]
sglang-main = sglang,http://localhost:8000
sglang-draft = sglang,http://localhost:8001
vllm = vllm,http://localhost:8080
local = llamacpp,http://localhost:8080
```

Server types: `sglang`, `vllm`, `llamacpp`

### Benchmarks

Add a `[benchmarks]` section to `config.ini` with paths to your benchmark result directories:

```ini
[benchmarks]
results_dir = /path/to/benchmark/results
aider_benchmarks_dir = /path/to/benchmark/aider/tmp.benchmarks
```

**Oneshot data** (`results_dir`):
- `cross-agent-oneshot-*.json` — cross-agent runs with opencode results per model/language
- `*-oneshot-*.json` — per-model runs for quantization/token cost views

**Multi-turn data** (`aider_benchmarks_dir`):
- `*-aiderdkr-*/` sweep directories with `_stats.yml` and `.aider.results.json` files

Benchmarks data is cached for 5 minutes. The "Benchmarks" tab stops live polling automatically.

**5 sub-views:**
1. **Oneshot** — opencode pass rate per model and language from cross-agent runs
2. **Multi-turn** — aider sweep pass@1/pass@2 with per-language breakdowns
3. **Language Heatmap** — color-coded pass rates across models and languages
4. **Quantization Impact** — per-model pass rates to compare quantization levels
5. **Token Cost** — prompt/completion token averages and totals per model

## Running as a service

Copy `dashboard.service` to `/etc/systemd/system/`, adjust paths, then:

```bash
systemctl daemon-reload
systemctl enable --now dashboard
```

## Endpoints queried

| Server   | Endpoints                                              |
|-----------|--------------------------------------------------------|
| SGLang    | `/v1/models`, `/health`, `/get_server_info`, `/metrics` |
| vLLM      | `/v1/models`, `/stats`, `/metrics`                     |
| llama.cpp | `/v1/models`, `/info`, `/stats`                        |

GPU stats come from `nvidia-smi --query-gpu`.
