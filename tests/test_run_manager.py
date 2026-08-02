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
from dgx_dashboard.control.catalog import LaunchProfile, ServeRecipe
from dgx_dashboard.control.commands import CommandBuilder
from dgx_dashboard.control.manager import RunConflict, RunManager
from dgx_dashboard.control.preflight import TargetUnavailable
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
        "launch_profile": None,
    }
    return OperationRequest(
        kind="serving",
        target=target,
        resources=frozenset(resources),
        public=request,
        recipe=recipe,
        action=action,
        launch_profile=None,
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


def test_operation_preflight_rejects_before_durable_state(tmp_path):
    _root, state, commands, catalog = _make_builder(tmp_path)
    checked = []

    def reject(operation):
        checked.append(operation.target)
        raise TargetUnavailable(operation.target)

    manager = RunManager(
        state,
        catalog,
        commands,
        retention=20,
        operation_preflight=reject,
    )

    with pytest.raises(TargetUnavailable) as unavailable:
        manager.submit(_serving_operation(catalog, "spark2"))

    assert unavailable.value.target == "spark2"
    assert checked == ["spark2"]
    assert manager.list(10) == []
    assert list(state.iterdir()) == []


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


def test_failed_start_does_not_hide_last_successful_serving_recipe(tmp_path):
    root, state, commands, catalog = _make_builder(tmp_path)
    _write_script(
        root / "serve.sh",
        "action=${4:-start}\n"
        "if [[ $action == start && $3 == recipe_conflict ]]; then exit 125; fi\n"
        "if [[ $action == status ]]; then printf 'container=x state=absent\\n'; fi\n",
    )
    _write_script(root / "benchmark.sh", "exit 0\n")
    manager = RunManager(state, catalog, commands, retention=20)

    active = manager.submit(_serving_operation(catalog, "local"))
    assert _wait_terminal(manager, active["id"])["state"] == "succeeded"

    conflict_recipe = ServeRecipe(
        target="local",
        engine="vllm",
        artifact="recipe_conflict",
        profile="rtx6000",
        served="served-conflict",
        container_name="container-conflict",
        package=root,
    )
    conflict_request = {
        "kind": "serving",
        "action": "start",
        "target": "local",
        "engine": conflict_recipe.engine,
        "artifact": conflict_recipe.artifact,
        "launch_profile": None,
    }
    conflict = manager.submit(
        OperationRequest(
            kind="serving",
            target="local",
            resources=frozenset({"target:local"}),
            public=conflict_request,
            recipe=conflict_recipe,
            action="start",
            launch_profile=None,
        )
    )
    assert _wait_terminal(manager, conflict["id"])["state"] == "failed"
    assert manager.latest_serving_recipes() == {"local": ("vllm", "recipe_local", None)}

def test_launch_profile_is_persisted_and_restored_as_serving_identity(tmp_path):
    root, state, commands, catalog = _make_builder(tmp_path)
    recipe = ServeRecipe(
        target="cluster",
        engine="vllm",
        artifact="recipe_cluster",
        profile="cluster",
        served="served-cluster",
        container_name="container-cluster",
        package=root,
        launch_profiles=(
            LaunchProfile("quality", "fp8_ds_mla", 1_048_576, 6, 3),
            LaunchProfile("throughput", "nvfp4_ds_mla", 350_000, 12, 5),
        ),
        default_launch_profile="quality",
    )
    catalog.recipes["cluster"] = recipe
    _write_script(
        root / "serve.sh",
        '[[ "${CLUSTER_PROFILE:-}" == throughput ]] || exit 9\n',
    )
    _write_script(root / "benchmark.sh", "exit 0\n")
    request = {
        "kind": "serving",
        "action": "start",
        "target": "cluster",
        "engine": "vllm",
        "artifact": "recipe_cluster",
        "launch_profile": "throughput",
    }
    operation = OperationRequest(
        kind="serving",
        target="cluster",
        resources=frozenset({"target:spark2", "target:spark3"}),
        public=request,
        recipe=recipe,
        action="start",
        launch_profile="throughput",
    )

    manager = RunManager(state, catalog, commands, retention=20)
    submitted = manager.submit(operation)
    completed = _wait_terminal(manager, submitted["id"])
    assert completed["state"] == "succeeded"
    assert completed["request"] == request
    assert manager.latest_serving_recipes() == {
        "cluster": ("vllm", "recipe_cluster", "throughput"),
    }

    restored = RunManager(state, catalog, commands, retention=20)
    assert restored.latest_serving_recipes() == {
        "cluster": ("vllm", "recipe_cluster", "throughput"),
    }



def test_serving_verification_does_not_block_run_reads(tmp_path):
    root, state, commands, catalog = _make_builder(tmp_path)
    _write_script(
        root / "serve.sh",
        "action=${4:-start}\n"
        "printf 'action=%s\\n' \"$action\"\n"
        "if [[ $action == verify ]]; then sleep 2; fi\n"
        "if [[ $action == stop || $action == status ]]; then printf 'container=x state=absent\\n'; fi\n",
    )
    _write_script(root / "benchmark.sh", "exit 0\n")
    manager = RunManager(state, catalog, commands, retention=20, cancel_grace=0.1)

    submitted = manager.submit(_serving_operation(catalog, "local"))
    deadline = time.monotonic() + 1
    while "action=verify" not in manager.read_log(submitted["id"], 0, 65_536)["data"]:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    started = time.monotonic()
    snapshot = manager.get(submitted["id"])
    assert time.monotonic() - started < 0.5
    assert snapshot["state"] == "running"
    manager.cancel(submitted["id"])
    assert _wait_terminal(manager, submitted["id"])["state"] == "cancelled"


def test_serving_verify_failure_runs_stop_and_status_cleanup(tmp_path):
    root, state, commands, catalog = _make_builder(tmp_path)
    _write_script(
        root / "serve.sh",
        "action=${4:-start}\n"
        "printf 'action=%s\\n' \"$action\"\n"
        "if [[ $action == verify ]]; then exit 7; fi\n"
        "if [[ $action == stop || $action == status ]]; then printf 'container=x state=absent\\n'; fi\n",
    )
    _write_script(root / "benchmark.sh", "exit 0\n")
    manager = RunManager(state, catalog, commands, retention=20)

    submitted = manager.submit(_serving_operation(catalog, "local"))
    terminal = _wait_terminal(manager, submitted["id"])
    assert terminal["state"] == "failed"
    assert terminal["error_code"] == "process_exit"
    output = manager.read_log(submitted["id"], 0, 65_536)["data"]
    assert "action=verify" in output
    assert "action=stop" in output
    assert "action=status" in output

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


def test_restart_preserves_retired_recipe_history_without_relaunching(tmp_path):
    root, state, commands, catalog = _make_builder(tmp_path)
    marker = tmp_path / "serve-called"
    _write_script(root / "serve.sh", f"touch {marker}\n")
    _write_script(root / "benchmark.sh", "exit 0\n")
    state.mkdir()

    def write_record(run_state: str, created_at: str) -> str:
        run_id = str(uuid.uuid4())
        run_dir = state / run_id
        run_dir.mkdir(mode=0o700)
        (run_dir / "output.log").write_bytes(b"historical output\n")
        terminal = run_state == "cleanup_failed"
        record = {
            "schema": 1,
            "id": run_id,
            "kind": "serving",
            "state": run_state,
            "request": {
                "kind": "serving",
                "action": "start",
                "target": "local",
                "engine": "vllm",
                "artifact": "retired_recipe",
            },
            "resources": ["target:local"],
            "created_at": created_at,
            "started_at": created_at,
            "finished_at": created_at if terminal else None,
            "exit_code": 255 if terminal else None,
            "error_code": "cleanup_failed" if terminal else None,
            "results": [],
        }
        (run_dir / "meta.json").write_text(json.dumps(record), encoding="utf-8")
        return run_id

    terminal_id = write_record("cleanup_failed", "2026-07-25T12:00:00+00:00")
    running_id = write_record("running", "2026-07-25T12:01:00+00:00")
    manager = RunManager(state, catalog, commands, retention=20)

    assert manager.get(terminal_id)["state"] == "cleanup_failed"
    assert manager.get(running_id)["state"] == "interrupted"
    assert manager.reconciliation == [{"id": running_id, "state": "interrupted"}]
    assert not marker.exists()
