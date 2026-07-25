"""Inference endpoint monitoring for SGLang, vLLM, and llama.cpp."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping

import requests

from dgx_dashboard.config import InferenceServerSettings


SGLANG_STATIC_TTL = 30
SGLANG_LOADS_TTL = 3


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class InferenceAdapters:
    get: Callable[..., requests.Response] = requests.get
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
    clock: Callable[[], float] = time.time
    utc_now: Callable[[], datetime] = _utc_now


def parse_prometheus(text: str) -> dict[str, float]:
    """Extract dashboard metrics from SGLang and vLLM Prometheus text."""

    metrics: dict[str, float] = {}
    patterns = [
        (r"sglang:num_generated_tokens_total\s+(\d+)", "generation_tokens"),
        (r"sglang:prompt_tokens_total\{[^}]*\}\s+([eE\d.]+)", "prompt_tokens"),
        (r"sglang:inter_token_latency_ms_mean\s+([\d.]+)", "inter_token_latency_ms"),
        (r"sglang:running_requests\s+(\d+)", "running_requests"),
        (r"sglang:gen_throughput\s+([\d.]+)", "throughput"),
        (r"sglang:cached_token_info\s+([\d.]+)", "kv_cache_hit_rate"),
        (r"sglang:mean_time_to_first_token_ms_mean\s+([\d.]+)", "ttft_ms"),
        (r"vllm:prompt_tokens_total\s+(\d+)", "prompt_tokens"),
        (r"vllm:generation_tokens_total\s+(\d+)", "generation_tokens"),
        (r"vllm:gpu_cache_usage_perc\s+([\d.]+)", "kv_cache_usage"),
        (r"vllm:num_requests_running\s+(\d+)", "running_requests"),
        (r"vllm:num_requests_waiting\s+(\d+)", "queued_requests"),
        (r"vllm:time_per_output_token_seconds_mean\s+([\d.]+)", "itl_s"),
        (r"vllm:time_to_first_token_seconds_mean\s+([\d.]+)", "ttft_s"),
    ]
    for pattern, key in patterns:
        match = re.search(pattern, text)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


class InferenceMonitor:
    def __init__(
        self,
        remote_hosts: Mapping[str, str],
        adapters: InferenceAdapters | None = None,
    ) -> None:
        self._remote_hosts = remote_hosts
        self._adapters = adapters or InferenceAdapters()
        self._cache: dict[str, dict[str, object]] = {}
        self._static_cache_time: dict[str, float] = {}
        self._loads_cache_time: dict[str, float] = {}

    def _get_json(self, url: str, path: str, timeout: int = 5) -> dict[str, object]:
        response = self._adapters.get(f"{url}{path}", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"{path} returned a non-object JSON value")
        return payload

    def _parse_sglang_logs(self, host: str | None = None) -> dict[str, float]:
        try:
            if host:
                command = ["ssh", host, "docker", "ps", "--format", "{{.Names}}"]
            else:
                command = ["docker", "ps", "--format", "{{.Names}}"]
            completed = self._adapters.run(
                command,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if completed.returncode != 0:
                return {}
            containers = [
                name.strip()
                for name in completed.stdout.splitlines()
                if "sglang" in name.lower()
            ]
            if not containers:
                return {}

            if host:
                command = ["ssh", host, "docker", "logs", "--since", "5m", containers[0]]
            else:
                command = ["docker", "logs", "--since", "5m", containers[0]]
            completed = self._adapters.run(
                command,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if completed.returncode != 0:
                return {}

            pattern = re.compile(
                r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*"
                r"Prefill batch.*input throughput \(token/s\):\s*([\d.]+)"
            )
            last_prefill: float | None = None
            now = self._adapters.utc_now()
            for line in (completed.stdout + completed.stderr).splitlines():
                match = pattern.search(line)
                if not match:
                    continue
                timestamp = datetime.strptime(
                    match.group(1),
                    "%Y-%m-%d %H:%M:%S",
                ).replace(tzinfo=timezone.utc)
                if (now - timestamp).total_seconds() < 15:
                    last_prefill = float(match.group(2))
            return {"input_throughput": last_prefill} if last_prefill is not None else {}
        except Exception:
            return {}

    def _query_sglang(
        self,
        url: str,
        result: dict[str, object],
        server_name: str,
    ) -> None:
        now = self._adapters.clock()
        stats = result["stats"]
        assert isinstance(stats, dict)

        if now - self._static_cache_time.get(server_name, 0) < SGLANG_STATIC_TTL:
            cached = self._cache.get(server_name, {})
            result["models"] = cached.get("models", [])
            result["online"] = cached.get("online", False)
            cached_stats = cached.get("stats", {})
            if isinstance(cached_stats, dict):
                for key in (
                    "max_total_tokens",
                    "context_len",
                    "version",
                    "max_prefill_tokens",
                    "max_running_requests",
                ):
                    if key in cached_stats:
                        stats[key] = cached_stats[key]
        else:
            try:
                models = self._get_json(url, "/v1/models")
                data = models.get("data", [])
                if not isinstance(data, list):
                    raise ValueError("/v1/models data must be a list")
                result["models"] = [
                    item.get("id", "?") if isinstance(item, dict) else "?" for item in data
                ]
                result["online"] = True

                try:
                    health = self._get_json(url, "/health")
                    info = health.get("server_info")
                    if isinstance(info, dict):
                        stats.update(
                            {
                                "max_total_tokens": info.get("max_total_tokens"),
                                "context_len": info.get("context_len"),
                                "version": info.get("version", ""),
                            }
                        )
                except Exception:
                    pass

                try:
                    info = self._get_json(url, "/server_info")
                    stats.update(
                        {
                            "max_total_num_tokens": info.get("max_total_num_tokens"),
                            "max_prefill_tokens": info.get("max_prefill_tokens"),
                            "max_running_requests": info.get("max_running_requests"),
                            "context_len": info.get("context_len"),
                            "version": info.get("version", stats.get("version", "")),
                        }
                    )
                except Exception:
                    pass

                self._cache[server_name] = {
                    "models": result["models"],
                    "online": result["online"],
                    "stats": dict(stats),
                }
                self._static_cache_time[server_name] = now
            except Exception:
                pass

        if now - self._loads_cache_time.get(server_name, 0) < SGLANG_LOADS_TTL:
            cached_loads = self._cache.get(server_name, {}).get("loads", {})
            if isinstance(cached_loads, dict):
                stats.update(cached_loads)
        else:
            try:
                loads = self._get_json(url, "/v1/loads")
                stats["version"] = loads.get("version", stats.get("version", ""))
                rank_loads = loads.get("loads", [])
                if isinstance(rank_loads, list):
                    for load in rank_loads:
                        if not isinstance(load, dict):
                            continue
                        stats["running_requests"] = load.get("num_running_reqs")
                        stats["queued_requests"] = load.get("num_waiting_reqs")
                        stats["token_usage"] = load.get("token_usage")
                        stats["max_total_num_tokens"] = load.get("max_total_num_tokens")
                        stats["max_running_requests"] = load.get("max_running_requests")
                        stats["gen_throughput"] = load.get("gen_throughput")
                        stats["cache_hit_rate"] = load.get("cache_hit_rate")
                        speculative = load.get("speculative", {})
                        if isinstance(speculative, dict):
                            stats["spec_accept_length"] = speculative.get("accept_length")
                            stats["spec_accept_rate"] = speculative.get("accept_rate")
                        memory = load.get("memory", {})
                        if isinstance(memory, dict):
                            stats["weight_gb"] = memory.get("weight_gb")
                            stats["kv_cache_gb"] = memory.get("kv_cache_gb")
                            stats["token_capacity"] = memory.get("token_capacity")
                        break
                self._loads_cache_time[server_name] = now
                self._cache.setdefault(server_name, {})["loads"] = {
                    key: stats.get(key)
                    for key in (
                        "running_requests",
                        "queued_requests",
                        "token_usage",
                        "max_total_num_tokens",
                        "max_running_requests",
                        "gen_throughput",
                        "cache_hit_rate",
                        "spec_accept_length",
                        "spec_accept_rate",
                        "weight_gb",
                        "kv_cache_gb",
                        "token_capacity",
                    )
                }
            except Exception:
                pass

        log_metrics = self._parse_sglang_logs(self._remote_hosts.get(server_name))
        stats["input_throughput"] = log_metrics.get("input_throughput", 0)
        if stats.get("running_requests", 0) == 0:
            stats["gen_throughput"] = 0
            stats["input_throughput"] = 0

    def _query_vllm(self, url: str, result: dict[str, object]) -> None:
        models = self._get_json(url, "/v1/models")
        data = models.get("data", [])
        result["models"] = [
            item.get("id", "?") if isinstance(item, dict) else "?"
            for item in data
            if isinstance(data, list)
        ]
        result["online"] = True
        stats_result = result["stats"]
        assert isinstance(stats_result, dict)
        try:
            stats = self._get_json(url, "/stats")
            stats_result.update(
                {
                    "gpu_cache_usage_perc": stats.get("gpu_cache_usage_perc"),
                    "gpu_swap_usage_perc": stats.get("gpu_swap_usage_perc"),
                    "num_running_reqs": stats.get("num_running_reqs"),
                    "num_swapped_reqs": stats.get("num_swapped_reqs"),
                    "num_waiting_reqs": stats.get("num_waiting_reqs"),
                }
            )
            token_usage = stats.get("token_usage", {})
            if isinstance(token_usage, dict) and token_usage:
                stats_result["prompt_tokens"] = token_usage.get("prompt_tokens")
                stats_result["generation_tokens"] = token_usage.get("generation_tokens")
        except Exception:
            pass
        try:
            response = self._adapters.get(f"{url}/metrics", timeout=5)
            response.raise_for_status()
            stats_result.update(parse_prometheus(response.text))
        except Exception:
            pass

    def _query_llamacpp(self, url: str, result: dict[str, object]) -> None:
        try:
            models = self._get_json(url, "/v1/models")
            data = models.get("data", [])
            if isinstance(data, list):
                result["models"] = [
                    item.get("id", "?") if isinstance(item, dict) else "?" for item in data
                ]
        except Exception:
            pass

        stats_result = result["stats"]
        assert isinstance(stats_result, dict)
        try:
            info = self._get_json(url, "/info")
            result["online"] = True
            model = info.get("model", {})
            usage = info.get("usage", {})
            main = info.get("main", {})
            stats_result.update(
                {
                    "model_name": model.get("name", "") if isinstance(model, dict) else "",
                    "description": model.get("description", "") if isinstance(model, dict) else "",
                    "n_ctx": usage.get("n_ctx", 0) if isinstance(usage, dict) else 0,
                    "n_batch": usage.get("n_batch", 0) if isinstance(usage, dict) else 0,
                    "n_gpu_layers": usage.get("n_gpu_layers", 0) if isinstance(usage, dict) else 0,
                    "mlock": main.get("mlock", False) if isinstance(main, dict) else False,
                    "mmap": main.get("mmap", False) if isinstance(main, dict) else False,
                }
            )
        except Exception:
            pass
        try:
            stats = self._get_json(url, "/stats")
            stats_result.update(
                {
                    "tokens_predicted": stats.get("tokens_predicted", 0),
                    "tokens_per_second": stats.get("tokens_per_second", 0),
                    "queue_predicting": stats.get("queue_predicting", 0),
                    "queue_remaining": stats.get("queue_remaining", 0),
                }
            )
        except Exception:
            pass

    def query(
        self,
        name: str,
        settings: InferenceServerSettings,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "name": name,
            "type": settings.kind,
            "url": settings.url,
            "online": False,
            "error": None,
            "models": [],
            "stats": {},
        }
        try:
            if settings.kind == "sglang":
                self._query_sglang(settings.url, result, name)
            elif settings.kind == "vllm":
                self._query_vllm(settings.url, result)
            elif settings.kind == "llamacpp":
                self._query_llamacpp(settings.url, result)
            else:
                result["error"] = f"Unknown type: {settings.kind}"
        except requests.exceptions.ConnectionError:
            result["error"] = "Connection refused"
        except requests.exceptions.Timeout:
            result["error"] = "Timeout"
        except Exception as error:
            result["error"] = str(error)[:200]
        return result
