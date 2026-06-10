#!/usr/bin/env python3
"""DGX Spark Dashboard - GPU + inference server monitoring."""

import configparser
import os
import re
import socket
import subprocess
import sys
import time
import concurrent.futures
from datetime import datetime, timezone

import requests
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)
app.jinja_env.auto_reload = True

# ── Config ───────────────────────────────────────────────────────────────────

CONFIG_PATH = os.environ.get("DASHBOARD_CONFIG",
                             os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"))
config = configparser.ConfigParser()
config.read(CONFIG_PATH)

HOST = config.get("server", "host", fallback="0.0.0.0")
PORT = config.getint("server", "port", fallback=9000)
REFRESH = config.getint("server", "refresh_interval", fallback=3)
AUTH_USER = config.get("server", "auth_user", fallback="")
AUTH_PASS = config.get("server", "auth_password", fallback="")

# Parse inference servers
INFERENCE_SERVERS = {}
if config.has_section("inference_servers"):
    for name, value in config.items("inference_servers"):
        parts = value.split(",")
        if len(parts) == 2:
            INFERENCE_SERVERS[name] = {
                "type": parts[0].strip(),
                "url": parts[1].strip().rstrip("/"),
            }

# Parse remote hosts (for SSH-based GPU queries)
REMOTE_HOSTS = {}
if config.has_section("remote_hosts"):
    for name, value in config.items("remote_hosts"):
        # Format: ssh_user@hostname_or_ip
        REMOTE_HOSTS[name] = value.strip()


# ── GPU stats ────────────────────────────────────────────────────────────────

GPU_QUERY_FIELDS = (
    "index,name,uuid,driver_version,"
    "utilization.gpu,utilization.memory,"
    "memory.used,memory.total,"
    "temperature.gpu,temperature.memory,"
    "power.draw,power.limit,"
    "clocks.current.graphics,clocks.current.memory,"
    "vbios_version,compute_mode"
)


def run_nvidia_smi(query_fields, header=False):
    """Run nvidia-smi locally, return raw text or None."""
    try:
        fmt = "csv" if header else "csv,noheader"
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query_fields}", f"--format={fmt}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def run_ssh_nvidia_smi(host, query_fields, header=False):
    """Run nvidia-smi on a remote host via SSH, return raw text or None."""
    try:
        fmt = "csv" if header else "csv,noheader"
        cmd = f"nvidia-smi --query-gpu={query_fields} --format={fmt}"
        result = subprocess.run(
            ["ssh", host, cmd],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        pass
    return None


def parse_gpu_stats():
    """Collect GPU stats via nvidia-smi (local + remote async)."""
    gpus = []

    # Local GPUs (fast, synchronous)
    count_out = run_nvidia_smi("count")
    if count_out:
        try:
            int(count_out.strip().split("\n")[0])
        except ValueError:
            pass
        else:
            raw = run_nvidia_smi(GPU_QUERY_FIELDS, header=False)
            hdr_out = run_nvidia_smi(GPU_QUERY_FIELDS, header=True)
            if raw and hdr_out:
                headers = [h.strip() for h in hdr_out.split("\n")[0].split(",")]
                raw_lines = [l for l in raw.split("\n") if l.strip() and not l.strip().startswith("index")]
                for line in raw_lines:
                    if not line.strip():
                        continue
                    values = [v.strip() for v in line.split(",")]
                    gpu = {"host": "local"}
                    for i, h in enumerate(headers):
                        if i >= len(values):
                            continue
                        val = values[i]
                        if val == "[N/A]" or val == "N/A":
                            val = None
                        else:
                            val = re.sub(r"\s+[A-Za-z%]+$", "", val)
                            try:
                                val = int(val)
                            except ValueError:
                                try:
                                    val = float(val)
                                except ValueError:
                                    pass
                        gpu[h] = val
                    gpus.append(gpu)

    # Remote GPUs (async via SSH)
    if REMOTE_HOSTS:
        def fetch_remote_gpus(host_name, host_addr):
            raw = run_ssh_nvidia_smi(host_addr, GPU_QUERY_FIELDS, header=False)
            hdr = run_ssh_nvidia_smi(host_addr, GPU_QUERY_FIELDS, header=True)
            if not raw or not hdr:
                return []
            headers = [h.strip() for h in hdr.split("\n")[0].split(",")]
            result = []
            for line in raw.split("\n"):
                if not line.strip():
                    continue
                values = [v.strip() for v in line.split(",")]
                gpu = {"host": host_name}
                for i, h in enumerate(headers):
                    if i >= len(values):
                        continue
                    val = values[i]
                    if val == "[N/A]" or val == "N/A":
                        val = None
                    else:
                        val = re.sub(r"\s+[A-Za-z%]+$", "", val)
                        try:
                            val = int(val)
                        except ValueError:
                            try:
                                val = float(val)
                            except ValueError:
                                pass
                    gpu[h] = val
                result.append(gpu)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(REMOTE_HOSTS)) as executor:
            futures = {executor.submit(fetch_remote_gpus, name, addr): name
                      for name, addr in REMOTE_HOSTS.items()}
            for future in concurrent.futures.as_completed(futures, timeout=10):
                try:
                    remote_gpus = future.result()
                    gpus.extend(remote_gpus)
                except (concurrent.futures.TimeoutError, Exception):
                    pass

    return gpus


# ── Prometheus metrics parser ────────────────────────────────────────────────

def parse_prometheus(text):
    """Extract useful values from Prometheus metrics text."""
    m = {}

    # SGLang
    for pat, key in [
        (r"sglang:num_generated_tokens_total\s+(\d+)", "generation_tokens"),
        (r"sglang:prompt_tokens_total\{[^}]*\}\s+([eE\d.]+)", "prompt_tokens"),
        (r"sglang:inter_token_latency_ms_mean\s+([\d.]+)", "inter_token_latency_ms"),
        (r"sglang:running_requests\s+(\d+)", "running_requests"),
        (r"sglang:gen_throughput\s+([\d.]+)", "throughput"),
        (r"sglang:cached_token_info\s+([\d.]+)", "kv_cache_hit_rate"),
        (r"sglang:mean_time_to_first_token_ms_mean\s+([\d.]+)", "ttft_ms"),
    ]:
        match = re.search(pat, text)
        if match:
            m[key] = float(match.group(1))

    # vLLM
    for pat, key in [
        (r"vllm:prompt_tokens_total\s+(\d+)", "prompt_tokens"),
        (r"vllm:generation_tokens_total\s+(\d+)", "generation_tokens"),
        (r"vllm:gpu_cache_usage_perc\s+([\d.]+)", "kv_cache_usage"),
        (r"vllm:num_requests_running\s+(\d+)", "running_requests"),
        (r"vllm:num_requests_waiting\s+(\d+)", "queued_requests"),
        (r"vllm:time_per_output_token_seconds_mean\s+([\d.]+)", "itl_s"),
        (r"vllm:time_to_first_token_seconds_mean\s+([\d.]+)", "ttft_s"),
    ]:
        match = re.search(pat, text)
        if match:
            m[key] = float(match.group(1))

    return m


# ── Inference server queries ─────────────────────────────────────────────────

def query_server(name, cfg):
    """Query one inference server. Returns stats dict."""
    stype = cfg["type"]
    url = cfg["url"]
    result = {
        "name": name, "type": stype, "url": url,
        "online": False, "error": None,
        "models": [], "stats": {},
    }
    try:
        if stype == "sglang":
            _query_sglang(url, result, name)
        elif stype == "vllm":
            _query_vllm(url, result)
        elif stype == "llamacpp":
            _query_llamacpp(url, result)
        else:
            result["error"] = f"Unknown type: {stype}"
    except requests.exceptions.ConnectionError:
        result["error"] = "Connection refused"
    except requests.exceptions.Timeout:
        result["error"] = "Timeout"
    except Exception as e:
        result["error"] = str(e)[:200]
    return result


def _get_json(url, path, timeout=5):
    r = requests.get(f"{url}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def _parse_sglang_logs(host=None):
    """Parse latest SGLang prefill batch log lines from Docker for input throughput."""
    try:
        if host:
            cmd = ["ssh", host, "docker", "ps", "--format", "{{.Names}}"]
        else:
            cmd = ["docker", "ps", "--format", "{{.Names}}"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return {}
        containers = [c.strip() for c in r.stdout.strip().split("\n") if "sglang" in c.lower()]
        if not containers:
            return {}
        container = containers[0]

        if host:
            cmd = ["ssh", host, "docker", "logs", "--since", "5m", container]
        else:
            cmd = ["docker", "logs", "--since", "5m", container]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return {}

        logs = r.stdout + r.stderr
        now = datetime.now(timezone.utc)
        stale_threshold = 15  # seconds

        prefill_pattern = re.compile(
            r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*Prefill batch.*input throughput \(token/s\):\s*([\d.]+)"
        )
        last_prefill = None
        for line in logs.split("\n"):
            m = prefill_pattern.search(line)
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                age = (now - ts).total_seconds()
                if age < stale_threshold:
                    last_prefill = float(m.group(2))
        return {"input_throughput": last_prefill} if last_prefill is not None else {}
    except Exception:
        pass
    return {}


# Cache for static endpoints (avoid waking GPU unnecessarily)
_sglang_cache = {}  # key: server name, value: {models, online, stats, loads}
_sglang_cache_time = {}  # key: server name, value: timestamp
_sglang_loads_time = {}  # key: server name, value: timestamp
_SGLANG_CACHE_TTL = 30  # seconds
_SGLANG_LOADS_TTL = 3  # seconds — /v1/loads is in-memory, poll at refresh rate

def _query_sglang(url, result, server_name=None):
    global _sglang_cache_time, _sglang_loads_time
    name = server_name or url
    now = time.time()
    
    # Models + server info (static, cache for 30s)
    if now - _sglang_cache_time.get(name, 0) < _SGLANG_CACHE_TTL:
        cached = _sglang_cache.get(name, {})
        result["models"] = cached.get("models", [])
        result["online"] = cached.get("online", False)
        # Only set static fields, don't overwrite dynamic /v1/loads data
        for key in ["max_total_tokens", "context_len", "version", "max_prefill_tokens", "max_running_requests"]:
            if key in cached.get("stats", {}):
                result["stats"][key] = cached["stats"][key]
    else:
        try:
            data = _get_json(url, "/v1/models")
            result["models"] = [x.get("id", "?") for x in data.get("data", [])]
            result["online"] = True

            try:
                health = _get_json(url, "/health")
                if "server_info" in health:
                    info = health["server_info"]
                    result["stats"].update({
                        "max_total_tokens": info.get("max_total_tokens"),
                        "context_len": info.get("context_len"),
                        "version": info.get("version", ""),
                    })
            except Exception:
                pass

            try:
                info = _get_json(url, "/server_info")
                result["stats"].update({
                    "max_total_num_tokens": info.get("max_total_num_tokens"),
                    "max_prefill_tokens": info.get("max_prefill_tokens"),
                    "max_running_requests": info.get("max_running_requests"),
                    "context_len": info.get("context_len"),
                    "version": info.get("version", result["stats"].get("version", "")),
                })
            except Exception:
                pass

            # Cache static data
            _sglang_cache[name] = {
                "models": result["models"],
                "online": result["online"],
                "stats": dict(result["stats"]),
            }
            _sglang_cache_time[name] = now
        except Exception:
            pass

    # v1/loads — real-time metrics (cache for 30s to avoid waking GPU)
    if now - _sglang_loads_time.get(name, 0) < _SGLANG_LOADS_TTL:
        # Restore cached loads data
        cached = _sglang_cache.get(name, {})
        loads_cached = cached.get("loads", {})
        result["stats"].update(loads_cached)
    else:
        try:
            loads = _get_json(url, "/v1/loads")
            result["stats"]["version"] = loads.get("version", result["stats"].get("version", ""))
            
            # Per-rank details (first rank is enough for single-GPU)
            for ld in loads.get("loads", []):
                result["stats"]["running_requests"] = ld.get("num_running_reqs")
                result["stats"]["queued_requests"] = ld.get("num_waiting_reqs")
                result["stats"]["token_usage"] = ld.get("token_usage")
                result["stats"]["max_total_num_tokens"] = ld.get("max_total_num_tokens")
                result["stats"]["max_running_requests"] = ld.get("max_running_requests")
                result["stats"]["gen_throughput"] = ld.get("gen_throughput")
                result["stats"]["cache_hit_rate"] = ld.get("cache_hit_rate")
                spec = ld.get("speculative", {})
                result["stats"]["spec_accept_length"] = spec.get("accept_length")
                result["stats"]["spec_accept_rate"] = spec.get("accept_rate")
                mem = ld.get("memory", {})
                result["stats"]["weight_gb"] = mem.get("weight_gb")
                result["stats"]["kv_cache_gb"] = mem.get("kv_cache_gb")
                result["stats"]["token_capacity"] = mem.get("token_capacity")
                break  # first rank is enough
            _sglang_loads_time[name] = now
            # Cache loads data
            _sglang_cache.setdefault(name, {})["loads"] = {
                "running_requests": result["stats"].get("running_requests"),
                "queued_requests": result["stats"].get("queued_requests"),
                "token_usage": result["stats"].get("token_usage"),
                "max_total_num_tokens": result["stats"].get("max_total_num_tokens"),
                "max_running_requests": result["stats"].get("max_running_requests"),
                "gen_throughput": result["stats"].get("gen_throughput"),
                "cache_hit_rate": result["stats"].get("cache_hit_rate"),
                "spec_accept_length": result["stats"].get("spec_accept_length"),
                "spec_accept_rate": result["stats"].get("spec_accept_rate"),
                "weight_gb": result["stats"].get("weight_gb"),
                "kv_cache_gb": result["stats"].get("kv_cache_gb"),
                "token_capacity": result["stats"].get("token_capacity"),
            }
        except Exception:
            pass

    # Parse Docker logs for input throughput
    log_host = None
    if server_name and server_name in REMOTE_HOSTS:
        log_host = REMOTE_HOSTS[server_name]
    log_metrics = _parse_sglang_logs(log_host)
    if log_metrics:
        result["stats"]["input_throughput"] = log_metrics.get("input_throughput")
    else:
        result["stats"]["input_throughput"] = 0

    # Clear throughput if no active generation
    if result["stats"].get("running_requests", 0) == 0:
        result["stats"]["gen_throughput"] = 0
        result["stats"]["input_throughput"] = 0


def _query_vllm(url, result):
    # Models
    data = _get_json(url, "/v1/models")
    result["models"] = [x.get("id", "?") for x in data.get("data", [])]
    result["online"] = True

    # Stats
    try:
        stats = _get_json(url, "/stats")
        result["stats"].update({
            "gpu_cache_usage_perc": stats.get("gpu_cache_usage_perc"),
            "gpu_swap_usage_perc": stats.get("gpu_swap_usage_perc"),
            "num_running_reqs": stats.get("num_running_reqs"),
            "num_swapped_reqs": stats.get("num_swapped_reqs"),
            "num_waiting_reqs": stats.get("num_waiting_reqs"),
        })
        tu = stats.get("token_usage", {})
        if tu:
            result["stats"]["prompt_tokens"] = tu.get("prompt_tokens")
            result["stats"]["generation_tokens"] = tu.get("generation_tokens")
    except Exception:
        pass

    # Metrics
    try:
        r = requests.get(f"{url}/metrics", timeout=5)
        r.raise_for_status()
        m = parse_prometheus(r.text)
        result["stats"].update(m)
    except Exception:
        pass


def _query_llamacpp(url, result):
    # Models
    try:
        data = _get_json(url, "/v1/models")
        result["models"] = [x.get("id", "?") for x in data.get("data", [])]
    except Exception:
        pass

    # Info
    try:
        info = _get_json(url, "/info")
        result["online"] = True
        model = info.get("model", {})
        result["stats"].update({
            "model_name": model.get("name", ""),
            "description": model.get("description", ""),
            "n_ctx": info.get("usage", {}).get("n_ctx", 0),
            "n_batch": info.get("usage", {}).get("n_batch", 0),
            "n_gpu_layers": info.get("usage", {}).get("n_gpu_layers", 0),
            "mlock": info.get("main", {}).get("mlock", False),
            "mmap": info.get("main", {}).get("mmap", False),
        })
    except Exception:
        pass

    # Stats
    try:
        stats = _get_json(url, "/stats")
        result["stats"].update({
            "tokens_predicted": stats.get("tokens_predicted", 0),
            "tokens_per_second": stats.get("tokens_per_second", 0),
            "queue_predicting": stats.get("queue_predicting", 0),
            "queue_remaining": stats.get("queue_remaining", 0),
        })
    except Exception:
        pass


# ── System stats ─────────────────────────────────────────────────────────────

def get_system_stats():
    stats = {
        "hostname": socket.gethostname(),
        "uptime": "",
        "cpu_count": os.cpu_count() or 0,
        "mem_total": 0, "mem_used": 0, "mem_available": 0,
        "load_avg": [],
    }
    try:
        with open("/proc/uptime") as f:
            secs = float(f.read().split()[0])
            d, rem = divmod(int(secs), 86400)
            h, rem = divmod(rem, 3600)
            m, _ = divmod(rem, 60)
            stats["uptime"] = f"{d}d {h}h {m}m"
    except Exception:
        pass
    try:
        mi = {}
        with open("/proc/meminfo") as f:
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    mi[p[0].rstrip(":")] = int(p[1])
        stats["mem_total"] = mi.get("MemTotal", 0) * 1024
        stats["mem_available"] = mi.get("MemAvailable", 0) * 1024
        stats["mem_used"] = stats["mem_total"] - stats["mem_available"]
    except Exception:
        pass
    try:
        stats["load_avg"] = list(os.getloadavg())
    except Exception:
        pass
    return stats


# ── Auth ─────────────────────────────────────────────────────────────────────

def check_auth():
    if not AUTH_USER or not AUTH_PASS:
        return True
    auth = request.authorization
    if not auth:
        return False
    return auth.username == AUTH_USER and auth.password == AUTH_PASS


@app.before_request
def before_req():
    if not check_auth() and (AUTH_USER and AUTH_PASS):
        return Response('Unauthorized', 401,
                        {"WWW-Authenticate": 'Basic realm="Login Required"'})


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", refresh_interval=REFRESH)


@app.route("/api/stats")
def api_stats():
    # Run GPU stats and server queries in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        gpu_future = executor.submit(parse_gpu_stats)
        server_futures = {executor.submit(query_server, n, c): n
                         for n, c in INFERENCE_SERVERS.items()}
        sys_future = executor.submit(get_system_stats)

        # Collect results with timeouts
        gpus = []
        try:
            gpus = gpu_future.result(timeout=15)
        except (concurrent.futures.TimeoutError, Exception):
            pass

        servers = []
        for future in concurrent.futures.as_completed(server_futures, timeout=10):
            try:
                servers.append(future.result())
            except (concurrent.futures.TimeoutError, Exception):
                pass

        system = {}
        try:
            system = sys_future.result(timeout=5)
        except (concurrent.futures.TimeoutError, Exception):
            pass

    return jsonify({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpus": gpus,
        "servers": servers,
        "system": system,
    })


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Dashboard: http://{HOST}:{PORT}")
    print(f"Monitoring {len(INFERENCE_SERVERS)} inference server(s)")
    app.run(host=HOST, port=PORT, threaded=True)
