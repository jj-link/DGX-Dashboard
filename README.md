# DGX Spark Dashboard

Live monitoring dashboard for DGX Spark (GB10) — GPU hardware stats + inference server metrics.

## Features

- **GPU stats**: utilization gauge, memory usage, temperature, power draw, clock speeds
- **Inference servers**: supports SGLang, vLLM, llama.cpp — shows throughput, latency, queue depth, KV cache, model info
- **System**: CPU load, RAM usage, uptime
- **Auto-refresh**: configurable poll interval (default 3s)
- **Dark theme**: terminal-friendly, no external dependencies

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
