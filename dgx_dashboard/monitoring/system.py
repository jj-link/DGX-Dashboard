"""Host system monitoring."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def _load_average() -> tuple[float, ...]:
    try:
        return os.getloadavg()
    except (AttributeError, OSError):
        return ()


@dataclass(frozen=True)
class SystemAdapters:
    hostname: Callable[[], str] = socket.gethostname
    cpu_count: Callable[[], int | None] = os.cpu_count
    load_average: Callable[[], tuple[float, ...]] = _load_average
    read_text: Callable[[Path], str] = _read_text


class SystemMonitor:
    def __init__(
        self,
        adapters: SystemAdapters | None = None,
        *,
        proc_root: Path = Path("/proc"),
    ) -> None:
        self._adapters = adapters or SystemAdapters()
        self._proc_root = proc_root

    def collect(self) -> dict[str, object]:
        stats: dict[str, object] = {
            "hostname": self._adapters.hostname(),
            "uptime": "",
            "cpu_count": self._adapters.cpu_count() or 0,
            "mem_total": 0,
            "mem_used": 0,
            "mem_available": 0,
            "load_avg": [],
        }
        try:
            seconds = float(self._adapters.read_text(self._proc_root / "uptime").split()[0])
            days, remainder = divmod(int(seconds), 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, _ = divmod(remainder, 60)
            stats["uptime"] = f"{days}d {hours}h {minutes}m"
        except (OSError, ValueError, IndexError):
            pass

        try:
            memory: dict[str, int] = {}
            for line in self._adapters.read_text(self._proc_root / "meminfo").splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    memory[parts[0].rstrip(":")] = int(parts[1])
            total = memory.get("MemTotal", 0) * 1024
            available = memory.get("MemAvailable", 0) * 1024
            stats["mem_total"] = total
            stats["mem_available"] = available
            stats["mem_used"] = total - available
        except (OSError, ValueError):
            pass

        try:
            stats["load_avg"] = list(self._adapters.load_average())
        except OSError:
            pass
        return stats
