"""Hardware-free serving catalog, command, and preflight contracts."""

from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from dgx_dashboard.control.catalog import ServeRecipe, ServingCatalog
from dgx_dashboard.control.commands import CommandBuilder
from dgx_dashboard.control.preflight import ControlPreflight, PreflightError
from dgx_dashboard.control.requests import RequestValidationError, validate_operation


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
            }
        ],
    }
    _, kwargs = parser.calls[0]
    assert set(kwargs["env"]) == {"PATH", "LANG", "LC_ALL"}
    assert kwargs["stdin"] == subprocess.DEVNULL
    with pytest.raises(KeyError):
        catalog.get("local", "vllm", "../model_a")


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
        {"kind": "serving", "action": "start", "target": "local", "engine": "vllm", "artifact": "model_a"},
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


def test_request_validation_rejects_bool_numeric_and_arbitrary_surface(tmp_path):
    catalog = _RequestCatalog(tmp_path)
    base = {"kind": "benchmark", "benchmark": "oneshot", "target": "local"}
    with pytest.raises(RequestValidationError, match="must be an integer"):
        validate_operation({**base, "options": {"concurrency": True}}, catalog)
    with pytest.raises(RequestValidationError, match="unknown field"):
        validate_operation({**base, "options": {"out_dir": "/tmp"}}, catalog)
    with pytest.raises(RequestValidationError, match="invalid filter"):
        validate_operation({**base, "options": {"keywords": ["x,y"]}}, catalog)


class _PreflightCatalog:
    def __init__(self, targets):
        self.targets = tuple(targets)


class _PreflightRunner:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, argv, **_kwargs):
        self.calls.append(tuple(argv))
        stdout = b""
        if argv[0] == "nvidia-smi":
            stdout = b"NVIDIA RTX PRO 6000 Blackwell Workstation Edition\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=b"")


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


def test_enabled_preflight_checks_gpu_and_hardened_remote_ssh(settings, tmp_path):
    configured = _preflight_settings(settings, tmp_path, targets=("local", "spark2"))
    catalog = _PreflightCatalog(configured.control.targets)
    commands = CommandBuilder(configured.control, configured.benchmarks)
    runner = _PreflightRunner()

    ControlPreflight(configured, catalog, commands, runner=runner).validate()
    assert any(call[0] == "nvidia-smi" for call in runner.calls)
    ssh = next(call for call in runner.calls if call[0] == "ssh")
    rendered = " ".join(ssh)
    assert "StrictHostKeyChecking=yes" in rendered
    assert "ForwardAgent=no" in rendered
    assert "ClearAllForwardings=yes" in rendered
    assert "RequestTTY=no" in rendered
    assert ssh[-2:] == ("spark2-ts", "exec true")


def test_enabled_preflight_rejects_wildcard_binding(settings, tmp_path):
    configured = _preflight_settings(settings, tmp_path)
    configured = replace(configured, server=replace(configured.server, host="0.0.0.0"))
    catalog = _PreflightCatalog(configured.control.targets)
    commands = CommandBuilder(configured.control, configured.benchmarks)
    with pytest.raises(PreflightError, match="wildcard"):
        ControlPreflight(configured, catalog, commands, runner=_PreflightRunner()).validate()
