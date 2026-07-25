"""Shared deterministic fixtures for DGX Dashboard tests."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest

from dgx_dashboard import create_app
from dgx_dashboard.config import (
    BenchmarkSettings,
    DashboardSettings,
    InferenceServerSettings,
    ServerSettings,
)


@pytest.fixture
def settings(tmp_path: Path) -> DashboardSettings:
    results = tmp_path / "benchmark-results"
    return DashboardSettings(
        source=tmp_path / "config.ini",
        server=ServerSettings(
            host="127.0.0.1",
            port=9000,
            refresh_interval=3,
            auth_user="",
            auth_password="",
        ),
        inference_servers=MappingProxyType(
            {
                "local": InferenceServerSettings(
                    kind="sglang",
                    url="http://127.0.0.1:8000",
                )
            }
        ),
        remote_hosts=MappingProxyType({}),
        benchmarks=BenchmarkSettings(
            results_dir=results,
            aider_benchmarks_dir=tmp_path / "aider-benchmarks",
            result_index_path=tmp_path / "result-index.json",
        ),
    )


@pytest.fixture
def stats_payload() -> dict[str, object]:
    return {
        "timestamp": "2026-07-25T14:00:00+00:00",
        "gpus": [],
        "servers": [
            {
                "name": "local",
                "type": "sglang",
                "url": "http://127.0.0.1:8000",
                "online": False,
                "error": "Connection refused",
                "models": [],
                "stats": {},
            }
        ],
        "system": {
            "hostname": "test-host",
            "uptime": "1d 2h 3m",
            "cpu_count": 16,
            "mem_total": 1024,
            "mem_used": 512,
            "mem_available": 512,
            "load_avg": [0.1, 0.2, 0.3],
        },
    }


@pytest.fixture
def benchmark_payload() -> dict[str, object]:
    return {
        "oneshot_table": [],
        "multiturn_lang_table": [],
        "quant_comparison_table": [],
        "comparison_table": [],
        "token_cost_table": [],
        "multiturn_leaderboard": [],
        "oneshot_leaderboard": {},
        "multiturn_summary": [],
    }


class _StaticMonitoring:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def collect(self) -> dict[str, object]:
        return self._payload


class _StaticBenchmarks:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def get(self) -> dict[str, object]:
        return self._payload


@pytest.fixture
def app(
    settings: DashboardSettings,
    stats_payload: dict[str, object],
    benchmark_payload: dict[str, object],
):
    application = create_app(
        settings,
        monitoring=_StaticMonitoring(stats_payload),
        benchmarks=_StaticBenchmarks(benchmark_payload),
    )
    application.config.update(TESTING=True, DEBUG=False)
    return application


@pytest.fixture
def client(app):
    return app.test_client()
