"""Validated dashboard configuration loading."""

from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlsplit


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config.ini"
_ALLOWED_SECTIONS = {"server", "inference_servers", "remote_hosts", "benchmarks"}
_SERVER_KEYS = {"host", "port", "refresh_interval", "auth_user", "auth_password"}
_BENCHMARK_KEYS = {"results_dir", "aider_benchmarks_dir"}
_SERVER_TYPES = {"sglang", "vllm", "llamacpp"}
_SSH_HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]*\Z")


class ConfigError(ValueError):
    """Raised when dashboard configuration is missing or unsafe."""


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    refresh_interval: int
    auth_user: str
    auth_password: str


@dataclass(frozen=True)
class InferenceServerSettings:
    kind: str
    url: str


@dataclass(frozen=True)
class BenchmarkSettings:
    results_dir: Path
    aider_benchmarks_dir: Path
    result_index_path: Path


@dataclass(frozen=True)
class DashboardSettings:
    source: Path
    server: ServerSettings
    inference_servers: Mapping[str, InferenceServerSettings]
    remote_hosts: Mapping[str, str]
    benchmarks: BenchmarkSettings


def _fail(source: Path, message: str) -> ConfigError:
    return ConfigError(f"{source}: {message}")


def _required(parser: configparser.ConfigParser, source: Path, section: str, key: str) -> str:
    if not parser.has_option(section, key):
        raise _fail(source, f"missing [{section}] {key}")
    return parser.get(section, key).strip()


def _parse_int(
    parser: configparser.ConfigParser,
    source: Path,
    section: str,
    key: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = _required(parser, source, section, key)
    try:
        value = int(raw)
    except ValueError as error:
        raise _fail(source, f"[{section}] {key} must be an integer") from error
    if not minimum <= value <= maximum:
        raise _fail(source, f"[{section}] {key} must be between {minimum} and {maximum}")
    return value


def _absolute_path(source: Path, key: str, raw: str) -> Path:
    if not raw:
        raise _fail(source, f"[benchmarks] {key} must not be empty")
    value = Path(raw).expanduser()
    if not value.is_absolute():
        raise _fail(source, f"[benchmarks] {key} must be an absolute path")
    return value.resolve(strict=False)


def _parse_server(source: Path, parser: configparser.ConfigParser) -> ServerSettings:
    unknown = set(parser.options("server")) - _SERVER_KEYS
    if unknown:
        raise _fail(source, f"unknown [server] option(s): {', '.join(sorted(unknown))}")

    host = _required(parser, source, "server", "host")
    if not host or any(character.isspace() for character in host):
        raise _fail(source, "[server] host must be a non-empty address without whitespace")
    port = _parse_int(parser, source, "server", "port", minimum=1, maximum=65535)
    refresh = _parse_int(
        parser,
        source,
        "server",
        "refresh_interval",
        minimum=1,
        maximum=3600,
    )
    auth_user = _required(parser, source, "server", "auth_user")
    auth_password = _required(parser, source, "server", "auth_password")
    if bool(auth_user) != bool(auth_password):
        raise _fail(source, "[server] auth_user and auth_password must both be set or both be empty")
    return ServerSettings(host, port, refresh, auth_user, auth_password)


def _parse_inference_servers(
    source: Path,
    parser: configparser.ConfigParser,
) -> Mapping[str, InferenceServerSettings]:
    servers: dict[str, InferenceServerSettings] = {}
    for name, raw in parser.items("inference_servers"):
        name = name.strip()
        if not name or any(character.isspace() for character in name):
            raise _fail(source, "[inference_servers] names must not be empty or contain whitespace")
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 2 or not all(parts):
            raise _fail(source, f"[inference_servers] {name} must be TYPE,URL")
        kind, url = parts
        if kind not in _SERVER_TYPES:
            raise _fail(
                source,
                f"[inference_servers] {name} has unsupported type {kind!r}; "
                f"expected one of {', '.join(sorted(_SERVER_TYPES))}",
            )
        parsed = urlsplit(url)
        try:
            parsed_port = parsed.port
        except ValueError as error:
            raise _fail(source, f"[inference_servers] {name} has an invalid URL port") from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or parsed_port is None
        ):
            raise _fail(
                source,
                f"[inference_servers] {name} URL must be an HTTP(S) origin with an explicit port",
            )
        servers[name] = InferenceServerSettings(kind=kind, url=url.rstrip("/"))
    return MappingProxyType(servers)


def _parse_remote_hosts(
    source: Path,
    parser: configparser.ConfigParser,
    servers: Mapping[str, InferenceServerSettings],
) -> Mapping[str, str]:
    hosts: dict[str, str] = {}
    for name, raw in parser.items("remote_hosts"):
        host = raw.strip()
        if name not in servers:
            raise _fail(source, f"[remote_hosts] {name} has no matching inference server")
        if not host or host.startswith("-") or not _SSH_HOST_RE.fullmatch(host):
            raise _fail(source, f"[remote_hosts] {name} has an invalid SSH host")
        hosts[name] = host
    return MappingProxyType(hosts)


def _parse_benchmarks(source: Path, parser: configparser.ConfigParser) -> BenchmarkSettings:
    unknown = set(parser.options("benchmarks")) - _BENCHMARK_KEYS
    if unknown:
        raise _fail(source, f"unknown [benchmarks] option(s): {', '.join(sorted(unknown))}")
    results_dir = _absolute_path(
        source,
        "results_dir",
        _required(parser, source, "benchmarks", "results_dir"),
    )
    aider_dir = _absolute_path(
        source,
        "aider_benchmarks_dir",
        _required(parser, source, "benchmarks", "aider_benchmarks_dir"),
    )
    return BenchmarkSettings(
        results_dir=results_dir,
        aider_benchmarks_dir=aider_dir,
        result_index_path=results_dir.parent / "result-index.json",
    )


def load_config(
    path: os.PathLike[str] | str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> DashboardSettings:
    """Load and validate one dashboard INI file.

    ``path`` wins over ``DASHBOARD_CONFIG``. Runtime directories may be absent at
    load time, but every configured path must be absolute.
    """

    environment = os.environ if environ is None else environ
    selected = path or environment.get("DASHBOARD_CONFIG") or DEFAULT_CONFIG_PATH
    source = Path(selected).expanduser().resolve(strict=False)
    if not source.is_file():
        raise _fail(source, "configuration file does not exist")

    parser = configparser.ConfigParser(interpolation=None, strict=True)
    try:
        with source.open(encoding="utf-8") as stream:
            parser.read_file(stream)
    except (OSError, UnicodeError, configparser.Error) as error:
        raise _fail(source, f"cannot read configuration: {error}") from error

    if parser.defaults():
        raise _fail(source, "[DEFAULT] options are not supported")
    unknown_sections = set(parser.sections()) - _ALLOWED_SECTIONS
    if unknown_sections:
        raise _fail(source, f"unknown section(s): {', '.join(sorted(unknown_sections))}")
    missing_sections = _ALLOWED_SECTIONS - set(parser.sections())
    if missing_sections:
        raise _fail(source, f"missing section(s): {', '.join(sorted(missing_sections))}")

    server = _parse_server(source, parser)
    inference_servers = _parse_inference_servers(source, parser)
    remote_hosts = _parse_remote_hosts(source, parser, inference_servers)
    benchmarks = _parse_benchmarks(source, parser)
    return DashboardSettings(
        source=source,
        server=server,
        inference_servers=inference_servers,
        remote_hosts=remote_hosts,
        benchmarks=benchmarks,
    )
