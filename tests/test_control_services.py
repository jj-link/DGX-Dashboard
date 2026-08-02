"""Hardware-free serving catalog, command, and preflight contracts."""

from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from dgx_dashboard.control.catalog import LaunchProfile, ServeRecipe, ServingCatalog
from dgx_dashboard.control.commands import CommandBuilder
from dgx_dashboard.control.preflight import ControlPreflight, PreflightError, TargetUnavailable
from dgx_dashboard.control.requests import (
    OperationRequest,
    RequestValidationError,
    validate_operation,
    validate_persisted_operation,
)
from dgx_dashboard.control.service import ControlService


_METADATA_KEYS = (
    "IMAGE", "MODEL", "MODEL_REVISION", "MODEL_HOST_PATH", "SERVED", "DRAFTER",
    "DRAFTER_REVISION", "DRAFTER_HOST_PATH", "TOKENIZER", "TOKENIZER_REVISION",
    "TOKENIZER_HOST_PATH", "CONTAINER_NAME",
)


def _executable(path: Path, content: str = "#!/usr/bin/env bash\nexit 0\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _metadata_record() -> bytes:
    values = {
        "IMAGE": "registry.example/model@sha256:" + "1" * 64,
        "MODEL": "owner/model",
        "MODEL_REVISION": "2" * 40,
        "MODEL_HOST_PATH": "/models/model",
        "SERVED": "served-model",
        "DRAFTER": "",
        "DRAFTER_REVISION": "",
        "DRAFTER_HOST_PATH": "",
        "TOKENIZER": "",
        "TOKENIZER_REVISION": "",
        "TOKENIZER_HOST_PATH": "",
        "CONTAINER_NAME": "served-model-container",
    }
    fields = []
    for key in _METADATA_KEYS:
        fields.extend((key, values[key]))
    return b"\0".join(value.encode("utf-8") for value in fields) + b"\0"


class _ParserRunner:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=_metadata_record(), stderr=b"")


def _catalog_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _executable(root / "control" / "tools" / "parse-runtime-env.py")
    package = root / "control" / "serve" / "vllm" / "rtx6000" / "model_a"
    package.mkdir(parents=True)
    (package / "runtime.env").write_text("validated by injected parser\n", encoding="utf-8")
    _executable(package / "serve.sh")
    return root


def test_catalog_uses_strict_parser_and_exposes_no_private_metadata(tmp_path):
    root = _catalog_root(tmp_path)
    parser = _ParserRunner()
    catalog = ServingCatalog(root, ("local",), run_parser=parser)

    assert catalog.targets == ("local",)
    assert catalog.get("local", "vllm", "model_a").container_name == "served-model-container"
    assert catalog.public() == {
        "enabled": True,
        "targets": ["local"],
        "recipes": [
            {
                "target": "local",
                "engine": "vllm",
                "artifact": "model_a",
                "profile": "rtx6000",
                "served": "served-model",
                "launch_profiles": [],
                "default_launch_profile": None,
            }
        ],
    }
    _, kwargs = parser.calls[0]
    assert set(kwargs["env"]) == {"PATH", "LANG", "LC_ALL"}
    assert kwargs["stdin"] == subprocess.DEVNULL
    with pytest.raises(KeyError):
        catalog.get("local", "vllm", "../model_a")


def test_catalog_exposes_validated_cluster_launch_profiles(tmp_path):
    root = _catalog_root(tmp_path)
    package = root / "control" / "serve" / "cluster" / "vllm" / "model_cluster"
    package.mkdir(parents=True)
    (package / "runtime.env").write_text("validated by injected parser\n", encoding="utf-8")
    for action in ("start", "status", "logs", "verify", "stop"):
        _executable(package / f"{action}.sh")
    profiles = package / "profiles"
    profiles.mkdir()
    (profiles / "default").write_text("quality\n", encoding="utf-8")
    for name, cache, context, sequences, tokens in (
        ("balanced", "nvfp4_ds_mla", 1_048_576, 6, 3),
        ("quality", "fp8_ds_mla", 1_048_576, 6, 3),
        ("throughput", "nvfp4_ds_mla", 350_000, 12, 5),
    ):
        (profiles / f"{name}.env").write_text(
            f"KV_CACHE_DTYPE={cache}\n"
            f"MAX_MODEL_LEN={context}\n"
            f"MAX_NUM_SEQS={sequences}\n"
            f"MTP_NUM_TOKENS={tokens}\n",
            encoding="utf-8",
        )

    catalog = ServingCatalog(root, ("local", "cluster"), run_parser=_ParserRunner())
    recipe = catalog.get("cluster", "vllm", "model_cluster")
    assert recipe.default_launch_profile == "quality"
    assert [profile.public() for profile in recipe.launch_profiles] == [
        {
            "name": "balanced",
            "kv_cache_dtype": "nvfp4_ds_mla",
            "context_length": 1_048_576,
            "max_sequences": 6,
            "speculative_tokens": 3,
        },
        {
            "name": "quality",
            "kv_cache_dtype": "fp8_ds_mla",
            "context_length": 1_048_576,
            "max_sequences": 6,
            "speculative_tokens": 3,
        },
        {
            "name": "throughput",
            "kv_cache_dtype": "nvfp4_ds_mla",
            "context_length": 350_000,
            "max_sequences": 12,
            "speculative_tokens": 5,
        },
    ]


def _recipe(root: Path, target: str = "local") -> ServeRecipe:
    return ServeRecipe(
        target=target,
        engine="vllm",
        artifact="model_a",
        profile="rtx6000" if target == "local" else "spark",
        served="served-model",
        container_name="served-model-container",
        package=root,
    )


class _RequestCatalog:
    targets = ("local", "spark2", "cluster")

    def __init__(self, root: Path) -> None:
        self.recipe = _recipe(root)

    def get(self, target, engine, artifact):
        if (target, engine, artifact) != self.recipe.key:
            raise KeyError
        return self.recipe


class _ProfileCatalog:
    targets = ("cluster",)

    def __init__(self, root: Path) -> None:
        self.recipe = ServeRecipe(
            target="cluster",
            engine="vllm",
            artifact="model_cluster",
            profile="cluster",
            served="served-cluster",
            container_name="served-cluster-container",
            package=root,
            launch_profiles=(
                LaunchProfile("balanced", "nvfp4_ds_mla", 1_048_576, 6, 3),
                LaunchProfile("quality", "fp8_ds_mla", 1_048_576, 6, 3),
                LaunchProfile("throughput", "nvfp4_ds_mla", 350_000, 12, 5),
            ),
            default_launch_profile="quality",
        )

    def get(self, target, engine, artifact):
        if (target, engine, artifact) != self.recipe.key:
            raise KeyError
        return self.recipe

    def recipes(self):
        return (self.recipe,)


def test_typed_requests_build_fixed_argv_and_allowlisted_environment(settings, tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    control_settings = replace(
        settings.control,
        enabled=True,
        wrapper_root=root,
        state_dir=tmp_path / "runs",
        polyglot_root=tmp_path / "polyglot",
        targets=("local", "spark2", "cluster"),
    )
    benchmark_settings = replace(
        settings.benchmarks,
        results_dir=tmp_path / "results",
        result_index_path=tmp_path / "result-index.json",
    )
    commands = CommandBuilder(control_settings, benchmark_settings)
    catalog = _RequestCatalog(root)
    monkeypatch.setenv("DASHBOARD_AUTH_PASSWORD", "must-not-forward")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")

    serving = validate_operation(
        {"kind": "serving", "action": "start", "target": "local", "engine": "vllm", "artifact": "model_a", "launch_profile": None},
        catalog,
    )
    plan = commands.plan(serving, "00000000-0000-4000-8000-000000000001")
    assert plan.command.argv == (str(root / "serve.sh"), "local", "vllm", "model_a")
    assert {key: plan.command.environment[key] for key in ("DETACH", "KEEP", "RESTART_POLICY")} == {
        "DETACH": "1", "KEEP": "1", "RESTART_POLICY": "unless-stopped"
    }
    assert "/usr/lib/wsl/lib" in plan.command.environment["PATH"].split(":")
    assert "DASHBOARD_AUTH_PASSWORD" not in plan.command.environment
    assert "SSH_AUTH_SOCK" not in plan.command.environment
    assert plan.verify is not None and plan.verify.argv[-1] == "verify"
    assert plan.command.timeout == control_settings.serving_timeout
    assert plan.verify.timeout == control_settings.serving_timeout
    assert [command.argv[-1] for command in plan.cleanup] == ["stop", "status"]

    benchmark = validate_operation(
        {
            "kind": "benchmark",
            "benchmark": "oneshot",
            "target": "spark2",
            "options": {
                "lang": None,
                "num_tests": 1,
                "keywords": ["arrays", "parsing"],
                "temperature": 0.5,
                "reasoning_effort": None,
            },
        },
        catalog,
    )
    benchmark_plan = commands.plan(benchmark, "00000000-0000-4000-8000-000000000002")
    assert benchmark_plan.command.argv == (
        str(root / "benchmark.sh"), "spark2", "oneshot",
        "--num-tests", "1", "--keywords", "arrays,parsing", "--temperature", "0.5",
    )
    assert "--lang" not in benchmark_plan.command.argv
    assert benchmark_plan.benchmark_label == "io.dgx-dashboard.run-id=00000000-0000-4000-8000-000000000002"


def test_profiled_serving_requests_propagate_exact_lifecycle_identity(settings, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    control_settings = replace(
        settings.control,
        enabled=True,
        wrapper_root=root,
        state_dir=tmp_path / "runs",
        polyglot_root=tmp_path / "polyglot",
        targets=("cluster",),
    )
    commands = CommandBuilder(control_settings, settings.benchmarks)
    catalog = _ProfileCatalog(root)
    payload = {
        "kind": "serving",
        "action": "start",
        "target": "cluster",
        "engine": "vllm",
        "artifact": "model_cluster",
        "launch_profile": "throughput",
    }

    operation = validate_operation(payload, catalog)
    plan = commands.plan(operation, "00000000-0000-4000-8000-000000000003")
    assert operation.public == payload
    assert operation.launch_profile == "throughput"
    commands_in_plan = [plan.command, plan.verify, *plan.cleanup]
    assert all(command is not None for command in commands_in_plan)
    assert all(command.environment["CLUSTER_PROFILE"] == "throughput" for command in commands_in_plan if command)

    with pytest.raises(RequestValidationError, match="launch_profile is required"):
        validate_operation({**payload, "launch_profile": None}, catalog)
    with pytest.raises(RequestValidationError, match="not supported"):
        validate_operation({**payload, "launch_profile": "latency"}, catalog)
    with pytest.raises(RequestValidationError, match="missing field"):
        validate_operation({key: value for key, value in payload.items() if key != "launch_profile"}, catalog)

    legacy = {key: value for key, value in payload.items() if key != "launch_profile"}
    persisted = validate_persisted_operation(legacy, catalog)
    assert persisted.public == legacy
    assert persisted.launch_profile == "quality"


def test_status_discovers_manual_profiles_and_reports_conflicts(tmp_path, monkeypatch):
    catalog = _ProfileCatalog(tmp_path)

    class Manager:
        reconciliation = []

        @staticmethod
        def latest_serving_recipes():
            return {}

    service = ControlService(catalog, object(), Manager())

    def one_running(_target, _engine, _artifact, launch_profile):
        return {
            "target": "cluster",
            "engine": "vllm",
            "artifact": "model_cluster",
            "launch_profile": launch_profile,
            "served": "served-cluster",
            "state": "running" if launch_profile == "throughput" else "absent",
            "ready": launch_profile == "throughput",
            "endpoint": "http://100.64.0.8:8888/v1" if launch_profile == "throughput" else None,
            "error": None,
        }

    monkeypatch.setattr(service, "_status_one", one_running)
    status = service.serving_status()["targets"][0]
    assert status["state"] == "running"
    assert status["launch_profile"] == "throughput"

    def conflicting(_target, _engine, _artifact, launch_profile):
        result = one_running(_target, _engine, _artifact, launch_profile)
        if launch_profile == "quality":
            result["state"] = "running"
            result["ready"] = True
        return result

    monkeypatch.setattr(service, "_status_one", conflicting)
    status = service.serving_status()["targets"][0]
    assert status["state"] == "conflict"
    assert status["error"] == "multiple_profiles_running"
    assert [item["launch_profile"] for item in status["running_profiles"]] == ["quality", "throughput"]

    def split_ranks(_target, _engine, _artifact, launch_profile):
        result = one_running(_target, _engine, _artifact, launch_profile)
        result["state"] = "mixed" if launch_profile == "balanced" else "absent"
        result["ready"] = False
        return result

    monkeypatch.setattr(service, "_status_one", split_ranks)
    status = service.serving_status()["targets"][0]
    assert status["state"] == "conflict"
    assert status["error"] == "profile_rank_mismatch"
    assert [item["launch_profile"] for item in status["running_profiles"]] == ["balanced"]


def test_request_validation_rejects_bool_numeric_and_arbitrary_surface(tmp_path):
    catalog = _RequestCatalog(tmp_path)
    base = {"kind": "benchmark", "benchmark": "oneshot", "target": "local"}
    with pytest.raises(RequestValidationError, match="must be an integer"):
        validate_operation({**base, "options": {"concurrency": True}}, catalog)
    with pytest.raises(RequestValidationError, match="unknown field"):
        validate_operation({**base, "options": {"out_dir": "/tmp"}}, catalog)
    with pytest.raises(RequestValidationError, match="invalid filter"):
        validate_operation({**base, "options": {"keywords": ["x,y"]}}, catalog)


def test_persisted_requests_allow_only_well_formed_retired_recipes(tmp_path):
    catalog = _RequestCatalog(tmp_path)
    request = {
        "kind": "serving",
        "action": "start",
        "target": "local",
        "engine": "vllm",
        "artifact": "retired_recipe",
    }
    with pytest.raises(RequestValidationError, match="unknown serving recipe"):
        validate_operation({**request, "launch_profile": None}, catalog)

    persisted = validate_persisted_operation(request, catalog)
    assert persisted.public == request
    assert persisted.recipe is None
    assert persisted.resources == frozenset({"target:local"})

    with pytest.raises(RequestValidationError, match="artifact must be a recipe name"):
        validate_persisted_operation({**request, "artifact": "../escape"}, catalog)


class _PreflightCatalog:
    def __init__(self, targets):
        self.targets = tuple(targets)


class _PreflightRunner:
    def __init__(self, unavailable_hosts=()) -> None:
        self.calls = []
        self.unavailable_hosts = frozenset(unavailable_hosts)

    def __call__(self, argv, **_kwargs):
        self.calls.append(tuple(argv))
        stdout = b""
        returncode = 0
        if argv[0] == "nvidia-smi":
            stdout = b"NVIDIA RTX PRO 6000 Blackwell Workstation Edition\n"
        elif argv[0] == "ssh" and argv[-2] in self.unavailable_hosts:
            returncode = 255
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=b"")


def _preflight_settings(settings, tmp_path, targets=("local",)):
    root = tmp_path / "repo"
    root.mkdir()
    _executable(root / "serve.sh")
    _executable(root / "benchmark.sh")
    state = tmp_path / "runs"
    state.mkdir()
    (tmp_path / "tmp").mkdir()
    results = tmp_path / "results"
    results.mkdir()
    corpus = tmp_path / "polyglot"
    for language in ("cpp", "go", "java", "javascript", "python", "rust"):
        (corpus / language / "exercises").mkdir(parents=True)
    return replace(
        settings,
        server=replace(
            settings.server,
            host="100.64.1.2",
            auth_user="operator",
            auth_password="secret",
        ),
        remote_hosts=MappingProxyType(
            {
                "spark1": "test-spark1",
                "spark2": "test-spark2",
                "spark3": "test-spark3",
            }
        ),
        benchmarks=replace(settings.benchmarks, results_dir=results),
        control=replace(
            settings.control,
            enabled=True,
            allowed_origin="http://100.64.1.2:9000",
            wrapper_root=root,
            state_dir=state,
            polyglot_root=corpus,
            targets=tuple(targets),
        ),
    )


def _preflight_operation(target, kind="serving"):
    return OperationRequest(
        kind=kind,
        target=target,
        resources=frozenset(),
        public={},
    )


def test_enabled_preflight_does_not_probe_target_hardware(settings, tmp_path):
    configured = _preflight_settings(
        settings,
        tmp_path,
        targets=("local", "spark1", "spark2", "spark3", "cluster"),
    )
    catalog = _PreflightCatalog(configured.control.targets)
    commands = CommandBuilder(configured.control, configured.benchmarks)
    runner = _PreflightRunner(unavailable_hosts={"test-spark1"})

    ControlPreflight(configured, catalog, commands, runner=runner).validate()

    assert not {"docker", "nvidia-smi", "ssh"} & {call[0] for call in runner.calls}


def test_operation_preflight_checks_only_selected_target(settings, tmp_path):
    configured = _preflight_settings(
        settings,
        tmp_path,
        targets=("local", "spark1", "spark2", "spark3", "cluster"),
    )
    catalog = _PreflightCatalog(configured.control.targets)
    commands = CommandBuilder(configured.control, configured.benchmarks)
    runner = _PreflightRunner()
    preflight = ControlPreflight(configured, catalog, commands, runner=runner)

    preflight.validate_operation(_preflight_operation("spark2"))

    assert len(runner.calls) == 1
    ssh = runner.calls[0]
    rendered = " ".join(ssh)
    assert ssh[-2:] == ("test-spark2", "exec true")
    assert "StrictHostKeyChecking=yes" in rendered
    assert "ForwardAgent=no" in rendered
    assert "ClearAllForwardings=yes" in rendered
    assert "RequestTTY=no" in rendered

    local_runner = _PreflightRunner()
    ControlPreflight(configured, catalog, commands, runner=local_runner).validate_operation(
        _preflight_operation("local")
    )
    assert [call[0] for call in local_runner.calls] == ["docker", "nvidia-smi"]

    cluster_runner = _PreflightRunner()
    ControlPreflight(configured, catalog, commands, runner=cluster_runner).validate_operation(
        _preflight_operation("cluster", kind="benchmark")
    )
    assert [call[0] for call in cluster_runner.calls] == ["docker", "ssh", "ssh"]
    assert {call[-2] for call in cluster_runner.calls[1:]} == {"test-spark2", "test-spark3"}


def test_operation_preflight_rejects_only_unavailable_target(settings, tmp_path):
    configured = _preflight_settings(
        settings,
        tmp_path,
        targets=("spark1", "spark2"),
    )
    catalog = _PreflightCatalog(configured.control.targets)
    commands = CommandBuilder(configured.control, configured.benchmarks)
    preflight = ControlPreflight(
        configured,
        catalog,
        commands,
        runner=_PreflightRunner(unavailable_hosts={"test-spark1"}),
    )

    with pytest.raises(TargetUnavailable) as unavailable:
        preflight.validate_operation(_preflight_operation("spark1"))
    assert unavailable.value.target == "spark1"

    preflight.validate_operation(_preflight_operation("spark2"))


def test_enabled_preflight_rejects_wildcard_binding(settings, tmp_path):
    configured = _preflight_settings(settings, tmp_path)
    configured = replace(configured, server=replace(configured.server, host="0.0.0.0"))
    catalog = _PreflightCatalog(configured.control.targets)
    commands = CommandBuilder(configured.control, configured.benchmarks)
    with pytest.raises(PreflightError, match="wildcard"):
        ControlPreflight(configured, catalog, commands, runner=_PreflightRunner()).validate()
