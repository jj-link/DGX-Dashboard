"""Composed monitoring service used by the read API."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Mapping

from dgx_dashboard.config import InferenceServerSettings
from dgx_dashboard.monitoring.gpu import GpuMonitor
from dgx_dashboard.monitoring.inference import InferenceMonitor
from dgx_dashboard.monitoring.system import SystemMonitor


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class MonitoringAdapters:
    executor: Callable[..., concurrent.futures.Executor] = concurrent.futures.ThreadPoolExecutor
    utc_now: Callable[[], datetime] = _utc_now


class MonitoringService:
    def __init__(
        self,
        inference_servers: Mapping[str, InferenceServerSettings],
        gpu: GpuMonitor,
        inference: InferenceMonitor,
        system: SystemMonitor,
        adapters: MonitoringAdapters | None = None,
    ) -> None:
        self._inference_servers = inference_servers
        self._gpu = gpu
        self._inference = inference
        self._system = system
        self._adapters = adapters or MonitoringAdapters()

    def collect(self) -> dict[str, object]:
        with self._adapters.executor(max_workers=4) as executor:
            gpu_future = executor.submit(self._gpu.collect)
            server_futures = [
                executor.submit(self._inference.query, name, settings)
                for name, settings in self._inference_servers.items()
            ]
            system_future = executor.submit(self._system.collect)

            try:
                gpus = gpu_future.result(timeout=15)
            except Exception:
                gpus = []

            servers: list[dict[str, object]] = []
            try:
                for future in concurrent.futures.as_completed(server_futures, timeout=10):
                    try:
                        servers.append(future.result())
                    except Exception:
                        pass
            except concurrent.futures.TimeoutError:
                pass

            try:
                system = system_future.result(timeout=5)
            except Exception:
                system = {}

        return {
            "timestamp": self._adapters.utc_now().isoformat(),
            "gpus": gpus,
            "servers": servers,
            "system": system,
        }
