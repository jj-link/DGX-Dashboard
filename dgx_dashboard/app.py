"""Flask application composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask

from dgx_dashboard.benchmarks.results import BenchmarkResults
from dgx_dashboard.config import DashboardSettings, load_config
from dgx_dashboard.monitoring import MonitoringAdapters, MonitoringService
from dgx_dashboard.monitoring.gpu import GpuAdapters, GpuMonitor
from dgx_dashboard.monitoring.inference import InferenceAdapters, InferenceMonitor
from dgx_dashboard.monitoring.system import SystemAdapters, SystemMonitor
from dgx_dashboard.web.routes import BenchmarkReader, MonitoringReader, create_blueprint


_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppAdapters:
    gpu: GpuAdapters = field(default_factory=GpuAdapters)
    inference: InferenceAdapters = field(default_factory=InferenceAdapters)
    system: SystemAdapters = field(default_factory=SystemAdapters)
    monitoring: MonitoringAdapters = field(default_factory=MonitoringAdapters)


def create_app(
    settings: DashboardSettings | None = None,
    *,
    config_path: str | Path | None = None,
    adapters: AppAdapters | None = None,
    monitoring: MonitoringReader | None = None,
    benchmarks: BenchmarkReader | None = None,
) -> Flask:
    """Create an isolated Flask app with injectable I/O services."""

    if settings is not None and config_path is not None:
        raise ValueError("pass settings or config_path, not both")
    resolved_settings = settings or load_config(config_path)
    resolved_adapters = adapters or AppAdapters()

    if monitoring is None:
        gpu_monitor = GpuMonitor(resolved_settings.remote_hosts, resolved_adapters.gpu)
        inference_monitor = InferenceMonitor(
            resolved_settings.remote_hosts,
            resolved_adapters.inference,
        )
        system_monitor = SystemMonitor(resolved_adapters.system)
        monitoring = MonitoringService(
            resolved_settings.inference_servers,
            gpu_monitor,
            inference_monitor,
            system_monitor,
            resolved_adapters.monitoring,
        )

    if benchmarks is None:
        benchmarks = BenchmarkResults(
            resolved_settings.benchmarks,
            _REPO_ROOT / "benchmark_static.json",
        )

    app = Flask(
        __name__,
        template_folder=str(_REPO_ROOT / "templates"),
        static_folder=str(_REPO_ROOT / "static"),
        static_url_path="/static",
    )
    app.jinja_env.auto_reload = True
    app.config["DASHBOARD_SETTINGS"] = resolved_settings
    app.extensions["dgx_dashboard.monitoring"] = monitoring
    app.extensions["dgx_dashboard.benchmarks"] = benchmarks
    app.register_blueprint(create_blueprint(resolved_settings, monitoring, benchmarks))
    return app
