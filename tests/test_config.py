"""Startup configuration validation contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from dgx_dashboard.config import ConfigError, load_config


VALID_CONFIG = """\
[server]
host = 127.0.0.1
port = 9000
refresh_interval = 3
auth_user =
auth_password =

[inference_servers]
local = sglang,http://127.0.0.1:8000
spark1 = vllm,http://100.64.1.2:8000

[remote_hosts]
spark1 = spark1-ts

[benchmarks]
results_dir = {results}
aider_benchmarks_dir = {aider}
"""


def write_config(tmp_path: Path, content: str | None = None) -> Path:
    path = tmp_path / "dashboard.ini"
    if content is None:
        content = VALID_CONFIG.format(
            results=tmp_path / "benchmark-results",
            aider=tmp_path / "aider-benchmarks",
        )
    path.write_text(content, encoding="utf-8")
    return path


def test_load_config_returns_typed_validated_settings(tmp_path):
    source = write_config(tmp_path)
    settings = load_config(source)

    assert settings.source == source
    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 9000
    assert settings.inference_servers["local"].kind == "sglang"
    assert settings.inference_servers["spark1"].url == "http://100.64.1.2:8000"
    assert settings.remote_hosts == {"spark1": "spark1-ts"}
    assert settings.benchmarks.result_index_path == tmp_path / "result-index.json"
    with pytest.raises(TypeError):
        settings.remote_hosts["spark2"] = "spark2-ts"


def test_explicit_path_wins_over_environment(tmp_path):
    source = write_config(tmp_path)
    settings = load_config(source, environ={"DASHBOARD_CONFIG": "/missing/config.ini"})
    assert settings.source == source


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("auth_user =", "auth_user and auth_password must both be set"),
        ("local = sglang,http://127.0.0.1:8000", "must be TYPE,URL"),
        ("port = 9000", "port must be between 1 and 65535"),
        ("results_dir =", "results_dir must be an absolute path"),
    ],
)
def test_invalid_configuration_fails_explicitly(tmp_path, replacement, message):
    content = VALID_CONFIG.format(
        results=tmp_path / "benchmark-results",
        aider=tmp_path / "aider-benchmarks",
    )
    if replacement == "auth_user =":
        content = content.replace("auth_user =", "auth_user = operator")
    elif replacement.startswith("local"):
        content = content.replace(replacement, "local = sglang")
    elif replacement.startswith("port"):
        content = content.replace(replacement, "port = 70000")
    else:
        content = content.replace(f"results_dir = {tmp_path / 'benchmark-results'}", "results_dir = relative")
    source = write_config(tmp_path, content)

    with pytest.raises(ConfigError, match=message):
        load_config(source)


def test_unknown_option_is_not_silently_ignored(tmp_path):
    content = VALID_CONFIG.format(
        results=tmp_path / "benchmark-results",
        aider=tmp_path / "aider-benchmarks",
    ).replace("refresh_interval = 3", "refresh_interval = 3\nrefresh_intervl = 4")
    source = write_config(tmp_path, content)
    with pytest.raises(ConfigError, match=r"unknown \[server\] option.*refresh_intervl"):
        load_config(source)


def test_missing_configuration_names_the_path(tmp_path):
    source = tmp_path / "missing.ini"
    with pytest.raises(ConfigError, match=r"missing\.ini: configuration file does not exist"):
        load_config(source)
