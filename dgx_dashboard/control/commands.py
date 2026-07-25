"""Fixed argv and environment construction for control-plane operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from dgx_dashboard.config import BenchmarkSettings, ControlSettings
from dgx_dashboard.control.catalog import ServeRecipe
from dgx_dashboard.control.requests import OperationRequest


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    timeout: int


@dataclass(frozen=True)
class OperationPlan:
    command: CommandSpec
    verify: CommandSpec | None
    cleanup: tuple[CommandSpec, ...]
    benchmark_label: str | None


class CommandBuilder:
    """Build only the closed commands exposed by the typed HTTP contract."""

    def __init__(
        self,
        control: ControlSettings,
        benchmarks: BenchmarkSettings,
        *,
        service_user: str = "workbench",
    ) -> None:
        self.control = control
        self.benchmarks = benchmarks
        self.root = control.wrapper_root
        self._base_environment = MappingProxyType(
            {
                "PATH": "/home/workbench/.local/bin:/usr/lib/wsl/lib:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "HOME": f"/home/{service_user}",
                "USER": service_user,
                "LOGNAME": service_user,
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONUNBUFFERED": "1",
                "TMPDIR": str(control.state_dir.parent / "tmp"),
                "POLYGLOT_ROOT": str(control.polyglot_root),
                "BENCHMARK_RESULTS_ROOT": str(benchmarks.results_dir),
            }
        )

    @property
    def base_environment(self) -> Mapping[str, str]:
        return self._base_environment

    def plan(self, operation: OperationRequest, run_id: str) -> OperationPlan:
        if operation.kind == "serving":
            if operation.recipe is None or operation.action is None:
                raise ValueError("serving operation is incomplete")
            command = self.serving(operation.recipe, operation.action)
            verify = self.serving(operation.recipe, "verify") if operation.action == "start" else None
            cleanup = (
                self.serving(operation.recipe, "stop"),
                self.serving(operation.recipe, "status"),
            ) if operation.action == "start" else ()
            return OperationPlan(command=command, verify=verify, cleanup=cleanup, benchmark_label=None)

        if operation.options is None:
            raise ValueError("benchmark operation is incomplete")
        command = self.benchmark(operation.target, operation.options, run_id)
        return OperationPlan(
            command=command,
            verify=None,
            cleanup=(),
            benchmark_label=f"io.dgx-dashboard.run-id={run_id}",
        )

    def serving(self, recipe: ServeRecipe, action: str, *, lines: int | None = None) -> CommandSpec:
        if action not in {"start", "status", "logs", "verify", "stop"}:
            raise ValueError("unsupported serving action")
        argv = [str(self.root / "serve.sh"), recipe.target, recipe.engine, recipe.artifact]
        if action != "start":
            argv.append(action)
        if action == "logs":
            if lines is None or not 1 <= lines <= 1000:
                raise ValueError("invalid serving log line count")
            argv.append(str(lines))
        environment = dict(self._base_environment)
        if action == "start":
            environment.update({"DETACH": "1", "KEEP": "1", "RESTART_POLICY": "unless-stopped"})
        return CommandSpec(
            argv=tuple(argv),
            cwd=self.root,
            environment=MappingProxyType(environment),
            timeout=self.control.serving_timeout if action in {"start", "verify"} else 300,
        )

    def benchmark(self, target: str, options: Mapping[str, object], run_id: str) -> CommandSpec:
        argv = [str(self.root / "benchmark.sh"), target, "oneshot"]
        option_flags = (
            ("lang", "--lang"),
            ("num_tests", "--num-tests"),
            ("keywords", "--keywords"),
            ("max_tokens", "--max-tokens"),
            ("temperature", "--temperature"),
            ("timeout", "--timeout"),
            ("test_timeout", "--test-timeout"),
            ("concurrency", "--concurrency"),
            ("reasoning", "--reasoning"),
            ("reasoning_effort", "--reasoning-effort"),
        )
        for name, flag in option_flags:
            value = options.get(name)
            if value is None:
                continue
            if name == "keywords":
                if not value:
                    continue
                value = ",".join(str(item) for item in value)
            argv.extend((flag, str(value)))
        environment = dict(self._base_environment)
        environment["DGX_DASHBOARD_RUN_ID"] = run_id
        return CommandSpec(
            argv=tuple(argv),
            cwd=self.root,
            environment=MappingProxyType(environment),
            timeout=self.control.benchmark_timeout,
        )
