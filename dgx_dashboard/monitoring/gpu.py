"""Local and remote NVIDIA GPU monitoring."""

from __future__ import annotations

import concurrent.futures
import subprocess
from dataclasses import dataclass
from typing import Callable, Mapping


GPU_QUERY_FIELDS = (
    "index,name,uuid,driver_version,"
    "utilization.gpu,utilization.memory,"
    "memory.used,memory.total,"
    "temperature.gpu,temperature.memory,"
    "power.draw,power.limit,"
    "clocks.current.graphics,clocks.current.memory,"
    "vbios_version,compute_mode"
)
_VALUE_SUFFIXES = (
    " %",
    " KiB",
    " MiB",
    " GiB",
    " MB",
    " GB",
    " W",
    " C",
    " kHz",
    " MHz",
    " GHz",
)


@dataclass(frozen=True)
class GpuAdapters:
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
    executor: Callable[..., concurrent.futures.Executor] = concurrent.futures.ThreadPoolExecutor


def _coerce_value(raw: str) -> object:
    value = raw.strip()
    if value in {"[N/A]", "N/A"}:
        return None
    for suffix in _VALUE_SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_gpu_csv(raw: str, header: str, host: str) -> list[dict[str, object]]:
    """Parse ``nvidia-smi`` CSV while preserving its literal header keys."""

    headers = [value.strip() for value in header.splitlines()[0].split(",")]
    gpus: list[dict[str, object]] = []
    for line in raw.splitlines():
        if not line.strip() or line.strip().startswith("index"):
            continue
        values = line.split(",")
        gpu: dict[str, object] = {"host": host}
        for index, key in enumerate(headers):
            if index < len(values):
                gpu[key] = _coerce_value(values[index])
        gpus.append(gpu)
    return gpus


class GpuMonitor:
    def __init__(
        self,
        remote_hosts: Mapping[str, str],
        adapters: GpuAdapters | None = None,
    ) -> None:
        self._remote_hosts = remote_hosts
        self._adapters = adapters or GpuAdapters()

    def _query(self, argv: list[str], timeout: int) -> str | None:
        try:
            completed = self._adapters.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
        output = completed.stdout.strip()
        return output or None

    def _local(self, query_fields: str, *, header: bool = False) -> str | None:
        output_format = "csv" if header else "csv,noheader"
        return self._query(
            ["nvidia-smi", f"--query-gpu={query_fields}", f"--format={output_format}"],
            10,
        )

    def _remote(
        self,
        host: str,
        query_fields: str,
        *,
        header: bool = False,
    ) -> str | None:
        output_format = "csv" if header else "csv,noheader"
        command = f"nvidia-smi --query-gpu={query_fields} --format={output_format}"
        return self._query(["ssh", host, command], 8)

    def _collect_remote(self, name: str, host: str) -> list[dict[str, object]]:
        raw = self._remote(host, GPU_QUERY_FIELDS, header=True)
        if not raw:
            return []
        return parse_gpu_csv(raw, raw, name)

    def collect(self) -> list[dict[str, object]]:
        gpus: list[dict[str, object]] = []
        raw = self._local(GPU_QUERY_FIELDS, header=True)
        if raw:
            gpus.extend(parse_gpu_csv(raw, raw, "local"))

        if not self._remote_hosts:
            return gpus

        with self._adapters.executor(max_workers=len(self._remote_hosts)) as executor:
            futures = [
                executor.submit(self._collect_remote, name, host)
                for name, host in self._remote_hosts.items()
            ]
            try:
                for future in concurrent.futures.as_completed(futures, timeout=10):
                    try:
                        gpus.extend(future.result())
                    except Exception:
                        pass
            except concurrent.futures.TimeoutError:
                pass
        return gpus
