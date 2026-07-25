"""Hardware-free monitoring parser and failure contracts."""

from __future__ import annotations

from pathlib import Path
import subprocess

import requests

from dgx_dashboard.config import InferenceServerSettings
from dgx_dashboard.monitoring.gpu import GpuAdapters, GpuMonitor, parse_gpu_csv
from dgx_dashboard.monitoring.inference import (
    InferenceAdapters,
    InferenceMonitor,
    parse_prometheus,
)
from dgx_dashboard.monitoring.system import SystemAdapters, SystemMonitor


def test_gpu_csv_preserves_literal_headers_and_types():
    header = "index, name, utilization.gpu [%], memory.used [MiB], temperature.memory"
    raw = "0, NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 42 %, 1234 MiB, [N/A]"

    assert parse_gpu_csv(raw, header, "local") == [
        {
            "host": "local",
            "index": 0,
            "name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            "utilization.gpu [%]": 42,
            "memory.used [MiB]": 1234,
            "temperature.memory": None,
        }
    ]



def test_gpu_monitor_keeps_local_data_when_a_remote_times_out():
    calls = []
    payload = "index, name, memory.used [MiB]\n0, Test GPU, 1234 MiB\n"

    def run(argv, **kwargs):
        calls.append((argv, kwargs["timeout"]))
        if argv[0] == "ssh" and argv[1] == "spark2-ts":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, stdout=payload, stderr="")

    monitor = GpuMonitor(
        {"spark1": "spark1-ts", "spark2": "spark2-ts"},
        GpuAdapters(run=run),
    )
    result = monitor.collect()

    assert sorted(gpu["host"] for gpu in result) == ["local", "spark1"]
    assert len(calls) == 3
    assert all("--format=csv" in call[-1] for call, _timeout in calls)
    assert max(timeout for _call, timeout in calls) <= 10

def test_prometheus_parser_preserves_metric_names():
    metrics = parse_prometheus(
        """
sglang:num_generated_tokens_total 123
sglang:running_requests 2
sglang:gen_throughput 45.5
vllm:num_requests_waiting 3
vllm:gpu_cache_usage_perc 0.75
"""
    )
    assert metrics == {
        "generation_tokens": 123.0,
        "running_requests": 2.0,
        "throughput": 45.5,
        "queued_requests": 3.0,
        "kv_cache_usage": 0.75,
    }


def test_inference_connection_failure_keeps_server_object_shape():
    def refuse(*_args, **_kwargs):
        raise requests.exceptions.ConnectionError("offline")

    monitor = InferenceMonitor({}, InferenceAdapters(get=refuse))
    result = monitor.query(
        "local",
        InferenceServerSettings(kind="vllm", url="http://127.0.0.1:8000"),
    )

    assert result == {
        "name": "local",
        "type": "vllm",
        "url": "http://127.0.0.1:8000",
        "online": False,
        "error": "Connection refused",
        "models": [],
        "stats": {},
    }


def test_system_monitor_uses_injected_proc_reader():
    files = {
        "uptime": "93784.0 0.0\n",
        "meminfo": "MemTotal: 1000 kB\nMemAvailable: 250 kB\n",
    }

    def read_text(path: Path) -> str:
        return files[path.name]

    monitor = SystemMonitor(
        SystemAdapters(
            hostname=lambda: "unit-host",
            cpu_count=lambda: 8,
            load_average=lambda: (1.0, 2.0, 3.0),
            read_text=read_text,
        )
    )
    assert monitor.collect() == {
        "hostname": "unit-host",
        "uptime": "1d 2h 3m",
        "cpu_count": 8,
        "mem_total": 1024000,
        "mem_used": 768000,
        "mem_available": 256000,
        "load_avg": [1.0, 2.0, 3.0],
    }
