"""Control catalog, serving probes, and durable operation facade."""

from __future__ import annotations

import concurrent.futures
import os
import re
import selectors
import signal
import subprocess
import time
from dataclasses import replace
from typing import Any

from dgx_dashboard.control.catalog import ServingCatalog
from dgx_dashboard.control.commands import CommandBuilder, CommandSpec
from dgx_dashboard.control.manager import RunManager


class AdapterError(RuntimeError):
    """Raised when a bounded serving adapter fails."""


class ControlService:
    def __init__(
        self,
        catalog: ServingCatalog,
        commands: CommandBuilder,
        manager: RunManager,
        *,
        popen_factory=subprocess.Popen,
    ) -> None:
        self.catalog = catalog
        self.commands = commands
        self.manager = manager
        self._popen = popen_factory

    def catalog_payload(self) -> dict[str, object]:
        return self.catalog.public()

    def serving_status(self) -> dict[str, object]:
        selections = self.manager.latest_serving_recipes()
        statuses: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(self.catalog.targets))) as executor:
            futures = {
                executor.submit(self._status_one, target, *selections[target]): target
                for target in self.catalog.targets
                if target in selections
            }
            for future, target in futures.items():
                try:
                    statuses[target] = future.result(timeout=35)
                except Exception:
                    statuses[target] = {"target": target, "state": "error", "error": "status_probe_failed"}
        for target in self.catalog.targets:
            statuses.setdefault(target, {"target": target, "state": "untracked", "error": None})
        return {
            "targets": [statuses[target] for target in self.catalog.targets],
            "reconciliation": self.manager.reconciliation,
        }

    def serving_logs(self, target: str, engine: str, artifact: str, lines: int) -> dict[str, object]:
        recipe = self.catalog.get(target, engine, artifact)
        command = replace(self.commands.serving(recipe, "logs", lines=lines), timeout=30)
        returncode, output, truncated = self._run_bounded(command, max_bytes=262_144)
        if returncode != 0:
            raise AdapterError("serving log adapter failed")
        return {
            "target": target,
            "engine": engine,
            "artifact": artifact,
            "lines": lines,
            "text": output.decode("utf-8", errors="replace"),
            "truncated": truncated,
        }

    def _status_one(self, target: str, engine: str, artifact: str) -> dict[str, Any]:
        try:
            recipe = self.catalog.get(target, engine, artifact)
        except KeyError:
            return {"target": target, "state": "error", "error": "recipe_removed"}
        command = replace(self.commands.serving(recipe, "status"), timeout=30)
        returncode, output, truncated = self._run_bounded(command, max_bytes=65_536)
        states = [match.decode("ascii") for match in re.findall(rb"\bstate=([a-z]+)\b", output)]
        endpoints = {
            match.decode("ascii")
            for match in re.findall(rb"\bendpoint=(https?://[0-9.]+:[0-9]+/v1)\b", output)
        }
        endpoint = next(iter(endpoints)) if len(endpoints) == 1 else None
        if returncode != 0 or truncated or not states:
            state = "error"
            error = "status_probe_failed"
        elif all(value == "running" for value in states):
            state, error = "running", None
        elif all(value == "absent" for value in states):
            state, error = "absent", None
        else:
            state, error = "mixed", None
        return {
            "target": target,
            "engine": engine,
            "artifact": artifact,
            "served": recipe.served,
            "state": state,
            "ready": state == "running",
            "endpoint": endpoint,
            "error": error,
        }

    def _run_bounded(self, command: CommandSpec, *, max_bytes: int) -> tuple[int, bytes, bool]:
        try:
            process = self._popen(
                list(command.argv),
                cwd=command.cwd,
                env=dict(command.environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
                shell=False,
            )
        except OSError:
            return -1, b"", False
        if process.stdout is None:
            self._stop(process)
            return -1, b"", False

        output = bytearray()
        truncated = False
        deadline = time.monotonic() + command.timeout
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._stop(process)
                    return -1, bytes(output), truncated
                events = selector.select(timeout=min(remaining, 0.25))
                if events:
                    chunk = os.read(process.stdout.fileno(), min(65_536, max_bytes + 1 - len(output)))
                    if not chunk:
                        break
                    output.extend(chunk)
                    if len(output) > max_bytes:
                        del output[max_bytes:]
                        truncated = True
                        self._stop(process)
                        return -1, bytes(output), True
                elif process.poll() is not None:
                    break
            try:
                returncode = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self._stop(process)
                returncode = -1
            return int(returncode), bytes(output), truncated
        finally:
            selector.close()
            process.stdout.close()

    @staticmethod
    def _stop(process: Any) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
