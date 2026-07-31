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

from dgx_dashboard.control.catalog import ServeRecipe, ServingCatalog
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
        selected_keys: dict[str, tuple[str, str, str, str | None]] = {
            target: (target, engine, artifact, launch_profile)
            for target, (engine, artifact, launch_profile) in selections.items()
        }
        probe_keys = set(selected_keys.values())
        for recipe in self.catalog.recipes():
            for launch_profile in recipe.launch_profiles:
                probe_keys.add((recipe.target, recipe.engine, recipe.artifact, launch_profile.name))

        results: dict[tuple[str, str, str, str | None], dict[str, Any]] = {}
        if probe_keys:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(probe_keys))) as executor:
                futures = {
                    executor.submit(self._status_one, *key): key
                    for key in probe_keys
                }
                for future, key in futures.items():
                    try:
                        results[key] = future.result(timeout=35)
                    except Exception:
                        results[key] = {
                            "target": key[0],
                            "engine": key[1],
                            "artifact": key[2],
                            "launch_profile": key[3],
                            "state": "error",
                            "ready": False,
                            "error": "status_probe_failed",
                        }

        statuses: dict[str, dict[str, Any]] = {}
        for target in self.catalog.targets:
            candidates = [result for key, result in results.items() if key[0] == target]
            running = [result for result in candidates if result.get("state") == "running"]
            split_profiles = [
                result
                for result in candidates
                if result.get("launch_profile") is not None and result.get("state") == "mixed"
            ]
            if len(running) > 1 or split_profiles:
                conflicting = running + split_profiles
                statuses[target] = {
                    "target": target,
                    "state": "conflict",
                    "ready": False,
                    "error": "profile_rank_mismatch" if split_profiles else "multiple_profiles_running",
                    "running_profiles": [
                        {
                            key: result.get(key)
                            for key in ("engine", "artifact", "launch_profile", "served", "state")
                        }
                        for result in sorted(
                            conflicting,
                            key=lambda item: (
                                str(item.get("engine")),
                                str(item.get("artifact")),
                                str(item.get("launch_profile")),
                            ),
                        )
                    ],
                }
            elif running:
                statuses[target] = running[0]
            elif target in selected_keys:
                statuses[target] = results[selected_keys[target]]
            else:
                statuses[target] = {"target": target, "state": "untracked", "error": None}
        return {
            "targets": [statuses[target] for target in self.catalog.targets],
            "reconciliation": self.manager.reconciliation,
        }

    def serving_logs(
        self,
        target: str,
        engine: str,
        artifact: str,
        launch_profile: str | None,
        lines: int,
    ) -> dict[str, object]:
        recipe = self.catalog.get(target, engine, artifact)
        command = replace(
            self.commands.serving(recipe, "logs", launch_profile, lines=lines),
            timeout=30,
        )
        returncode, output, truncated = self._run_bounded(command, max_bytes=262_144)
        if returncode != 0:
            raise AdapterError("serving log adapter failed")
        return {
            "target": target,
            "engine": engine,
            "artifact": artifact,
            "launch_profile": launch_profile,
            "lines": lines,
            "text": output.decode("utf-8", errors="replace"),
            "truncated": truncated,
        }

    def _status_one(
        self,
        target: str,
        engine: str,
        artifact: str,
        launch_profile: str | None,
    ) -> dict[str, Any]:
        try:
            recipe = self.catalog.get(target, engine, artifact)
        except KeyError:
            return {
                "target": target,
                "engine": engine,
                "artifact": artifact,
                "launch_profile": launch_profile,
                "state": "error",
                "ready": False,
                "error": "recipe_removed",
            }
        try:
            command = replace(
                self.commands.serving(recipe, "status", launch_profile),
                timeout=30,
            )
        except ValueError:
            return {
                "target": target,
                "engine": engine,
                "artifact": artifact,
                "launch_profile": launch_profile,
                "state": "error",
                "ready": False,
                "error": "profile_removed",
            }
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
            "launch_profile": launch_profile,
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
