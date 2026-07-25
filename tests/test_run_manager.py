"""Durable run state, exclusion, cancellation, and cleanup contracts."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from dgx_dashboard.config import BenchmarkSettings, ControlSettings
from dgx_dashboard.control.catalog import ServeRecipe
from dgx_dashboard.control.commands import CommandBuilder
from dgx_dashboard.control.manager import RunConflict, RunManager
from dgx_dashboard.control.requests import OperationRequest


class _Catalog:
    targets = ("local", "spark2", "cluster")

    def __init__(self, root: Path) -> None:
        self.recipes = {
            target: ServeRecipe(
                target=target,
                engine="vllm",
                artifact=f"recipe_{target}",
                profile="cluster" if target == "cluster" else ("rtx6000" if target == "local" else "spark"),
                served=f"served-{target}",
                container_name=f"container-{target}",
                package=root,
            )
            for target in self.targets
        }

    def get(self, target, engine, artifact):
        recipe = self.recipes.get(target)
        if recipe is None or (engine, artifact) != (recipe.engine, recipe.artifact):
            raise KeyError("unknown recipe")
        return recipe


def _make_builder(tmp_path: Path, *, benchmark_timeout: int = 30, serving_timeout: int = 30):
    root = tmp_path / "repo"
    root.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    corpus = tmp_path / "polyglot"
    corpus.mkdir()
    state = tmp_path / "runs"
    (tmp_path / "tmp").mkdir()
    control = ControlSettings(
        enabled=True,
        allowed_origin="http://100.64.1.2:9000",
        wrapper_root=root,
        state_dir=state,
        polyglot_root=corpus,
        targets=("local", "spark2", "cluster"),
        retention=20,
        serving_timeout=serving_timeout,
        benchmark_timeout=benchmark_timeout,
    )
    benchmarks = BenchmarkSettings(
        results_dir=results,
        aider_benchmarks_dir=tmp_path / "aider",
        result_index_path=tmp_path / "result-index.json",
    )
    return root, state, CommandBuilder(control, benchmarks), _Catalog(root)


def _write_script(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _benchmark_operation(target: str = "local") -> OperationRequest:
    resources = {f"target:{target}"} if target != "cluster" else {"target:spark2", "target:spark3"}
    resources.add("benchmark-worker")
    request = {
        "kind": "benchmark",
        "benchmark": "oneshot",
        "target": target,
        "options": {"lang": "python", "num_tests": 1},
    }
    return OperationRequest(
        kind="benchmark",
        target=target,
        resources=frozenset(resources),
        public=request,
        options=request["options"],
    )


def _serving_operation(catalog: _Catalog, target: str, action: str = "start") -> OperationRequest:
    recipe = catalog.recipes[target]
    resources = {
        "local": {"target:local"},
        "spark2": {"target:spark2"},
        "cluster": {"target:spark2", "target:spark3"},
    }[target]
    request = {
        "kind": "serving",
        "action": action,
        "target": target,
        "engine": recipe.engine,
        "artifact": recipe.artifact,
    }
    return OperationRequest(
        kind="serving",
        target=target,
        resources=frozenset(resources),
        public=request,
        recipe=recipe,
        action=action,
    )


def _wait_terminal(manager: RunManager, run_id: str, timeout: float = 8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = manager.get(run_id)
        if record["state"] in {
            "succeeded", "failed", "timed_out", "cancelled", "interrupted", "launch_failed", "cleanup_failed"
        }:
            return record
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not become terminal")


class _DockerCleanup:
    def __init__(self, container_ids: bytes = b"") -> None:
        self.calls = []
        self.container_ids = container_ids

    def __call__(self, argv, **_kwargs):
        self.calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=self.container_ids)


def test_success_persists_safe_metadata_direct_log_and_result_links(tmp_path, monkeypatch):
    root, state, commands, catalog = _make_builder(tmp_path)
    _write_script(
        root / "benchmark.sh",
        "printf 'target=%s run=%s secret=%s\\n' \"$1\" \"$DGX_DASHBOARD_RUN_ID\" \"${SHOULD_NOT_LEAK-unset}\"\n",
    )
    _write_script(root / "serve.sh", "exit 0\n")
    monkeypatch.setenv("SHOULD_NOT_LEAK", "ambient-secret")
    manager = RunManager(
        state,
        catalog,
        commands,
        retention=20,
        result_resolver=lambda _run_id: [
            {"label": "local · now", "url": "/api/benchmarks/results/c2FmZS5qc29u"}
        ],
    )

    submitted = manager.submit(_benchmark_operation())
    record = _wait_terminal(manager, submitted["id"])
    assert record["state"] == "succeeded"
    assert record["results"] == [
        {"label": "local · now", "url": "/api/benchmarks/results/c2FmZS5qc29u"}
    ]

    run_dir = state / submitted["id"]
    metadata = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    serialized = json.dumps(metadata)
    assert not {"pid", "argv", "cwd", "environment"} & set(metadata)
    assert "ambient-secret" not in serialized
    assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((run_dir / "meta.json").stat().st_mode) == 0o640
    assert stat.S_IMODE((run_dir / "output.log").stat().st_mode) == 0o640

    first = manager.read_log(submitted["id"], 0, 12)
    second = manager.read_log(submitted["id"], first["next_offset"], 65_536)
    assert first["next_offset"] <= second["next_offset"]
    assert "secret=unset" in first["data"] + second["data"]


def test_global_benchmark_exclusion_reports_occupant_and_cancel_cleans_exact_label(tmp_path):
    root, state, commands, catalog = _make_builder(tmp_path)
    _write_script(root / "benchmark.sh", "printf 'started\\n'\nexec sleep 30\n")
    _write_script(root / "serve.sh", "exit 0\n")
    cleanup = _DockerCleanup()
    manager = RunManager(
        state,
        catalog,
        commands,
        retention=20,
        run_factory=cleanup,
        cancel_grace=0.2,
    )

    first = manager.submit(_benchmark_operation("local"))
    with pytest.raises(RunConflict) as conflict:
        manager.submit(_benchmark_operation("spark2"))
    assert conflict.value.occupying_run_id == first["id"]

    assert manager.cancel(first["id"])["state"] == "cancel_requested"
    terminal = _wait_terminal(manager, first["id"])
    assert terminal["state"] == "cancelled"
    list_call = cleanup.calls[0]
    assert list_call[-1] == f"label=io.dgx-dashboard.run-id={first['id']}"
    assert "io.dgx-dashboard.run-id" in list_call[-1]


def test_cluster_lease_conflicts_with_spark2_mutation(tmp_path):
    root, state, commands, catalog = _make_builder(tmp_path)
    _write_script(
        root / "serve.sh",
        "action=${4:-start}\nif [[ $action == start ]]; then exec sleep 30; fi\nprintf 'container=x state=absent\\n'\n",
    )
    _write_script(root / "benchmark.sh", "exit 0\n")
    manager = RunManager(state, catalog, commands, retention=20, cancel_grace=0.2)

    cluster = manager.submit(_serving_operation(catalog, "cluster"))
    with pytest.raises(RunConflict) as conflict:
        manager.submit(_serving_operation(catalog, "spark2", "stop"))
    assert conflict.value.occupying_run_id == cluster["id"]
    manager.cancel(cluster["id"])
    assert _wait_terminal(manager, cluster["id"])["state"] == "cancelled"
    log = manager.read_log(cluster["id"], 0, 65_536)["data"]
    assert "container=x state=absent" in log


def test_timeout_terminates_and_runs_benchmark_label_cleanup(tmp_path):
    root, state, commands, catalog = _make_builder(tmp_path, benchmark_timeout=1)
    _write_script(root / "benchmark.sh", "exec sleep 30\n")
    _write_script(root / "serve.sh", "exit 0\n")
    cleanup = _DockerCleanup()
    manager = RunManager(
        state,
        catalog,
        commands,
        retention=20,
        run_factory=cleanup,
        cancel_grace=0.1,
    )

    submitted = manager.submit(_benchmark_operation())
    terminal = _wait_terminal(manager, submitted["id"])
    assert terminal["state"] == "timed_out"
    assert terminal["error_code"] == "operation_timeout"
    assert cleanup.calls[0][-1] == f"label=io.dgx-dashboard.run-id={submitted['id']}"


def test_serving_start_runs_exact_verify_followup(tmp_path):
    root, state, commands, catalog = _make_builder(tmp_path)
    _write_script(
        root / "serve.sh",
        "action=${4:-start}\nprintf 'action=%s target=%s artifact=%s\\n' \"$action\" \"$1\" \"$3\"\n",
    )
    _write_script(root / "benchmark.sh", "exit 0\n")
    manager = RunManager(state, catalog, commands, retention=20)

    submitted = manager.submit(_serving_operation(catalog, "local"))
    assert _wait_terminal(manager, submitted["id"])["state"] == "succeeded"
    output = manager.read_log(submitted["id"], 0, 65_536)["data"]
    assert "action=start target=local artifact=recipe_local" in output
    assert "action=verify target=local artifact=recipe_local" in output


def test_restart_reconciles_benchmark_and_removes_only_labeled_containers(tmp_path):
    root, state, commands, catalog = _make_builder(tmp_path)
    _write_script(root / "benchmark.sh", "exit 0\n")
    _write_script(root / "serve.sh", "exit 0\n")
    run_id = str(uuid.uuid4())
    run_dir = state / run_id
    run_dir.mkdir(parents=True, mode=0o700)
    (run_dir / "output.log").write_bytes(b"")
    record = {
        "schema": 1,
        "id": run_id,
        "kind": "benchmark",
        "state": "running",
        "request": dict(_benchmark_operation().public),
        "resources": ["benchmark-worker", "target:local"],
        "created_at": "2026-07-25T12:00:00+00:00",
        "started_at": "2026-07-25T12:00:01+00:00",
        "finished_at": None,
        "exit_code": None,
        "error_code": None,
        "results": [],
    }
    (run_dir / "meta.json").write_text(json.dumps(record), encoding="utf-8")
    cleanup = _DockerCleanup(b"abcdef123456\n")

    manager = RunManager(state, catalog, commands, retention=20, run_factory=cleanup)
    reconciled = manager.get(run_id)
    assert reconciled["state"] == "interrupted"
    assert manager.reconciliation == [{"id": run_id, "state": "interrupted"}]
    assert cleanup.calls[0][-1] == f"label=io.dgx-dashboard.run-id={run_id}"
    assert cleanup.calls[1][:4] == ("docker", "container", "rm", "--force")
