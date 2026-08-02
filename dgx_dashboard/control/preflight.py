"""Fail-closed infrastructure preflights for mutation controls."""

from __future__ import annotations

import os
import socket
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from dgx_dashboard.config import DashboardSettings
from dgx_dashboard.control.catalog import ServingCatalog
from dgx_dashboard.control.commands import CommandBuilder
from dgx_dashboard.control.requests import OperationRequest


class PreflightError(RuntimeError):
    """Raised when control-enabled startup is unsafe."""


class TargetUnavailable(RuntimeError):
    """Raised when an operation's target infrastructure cannot be reached."""

    def __init__(self, target: str) -> None:
        super().__init__(f"{target} is unavailable")
        self.target = target


_REMOTE_TARGET_MEMBERS = {
    "spark1": ("spark1",),
    "spark2": ("spark2",),
    "spark3": ("spark3",),
    "cluster": ("spark2", "spark3"),
}

class ControlPreflight:
    def __init__(
        self,
        settings: DashboardSettings,
        catalog: ServingCatalog,
        commands: CommandBuilder,
        *,
        runner=subprocess.run,
    ) -> None:
        self.settings = settings
        self.catalog = catalog
        self.commands = commands
        self._run = runner

    def validate(self) -> None:
        control = self.settings.control
        server = self.settings.server
        if not control.enabled:
            return
        if not server.auth_user or not server.auth_password:
            raise PreflightError("control authentication is not configured")
        self._validate_binding()

        root = control.wrapper_root
        try:
            canonical = root.resolve(strict=True)
        except OSError as error:
            raise PreflightError("the canonical wrapper root is unavailable") from error
        if canonical != root or not root.is_dir():
            raise PreflightError("the wrapper root is not canonical")
        for wrapper in (root / "serve.sh", root / "benchmark.sh"):
            if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
                raise PreflightError("a root control wrapper is unavailable")

        for directory in (
            control.state_dir,
            control.state_dir.parent / "tmp",
            self.settings.benchmarks.results_dir,
            control.polyglot_root,
        ):
            self._validate_writable_directory(directory)
        languages = ("cpp", "go", "java", "javascript", "python", "rust")
        if any(not (control.polyglot_root / language / "exercises").is_dir() for language in languages):
            raise PreflightError("the polyglot benchmark corpus is incomplete")

        self._check(["git", "symbolic-ref", "-q", "HEAD"], "the controller Git checkout is detached")
        status = self._check(["git", "status", "--porcelain", "--untracked-files=normal"], "the controller Git status is unavailable")
        if status.stdout:
            raise PreflightError("the controller Git checkout is dirty")
        self._validate_target_configuration()

    def validate_operation(self, operation: OperationRequest) -> None:
        """Fail closed against only the infrastructure used by one operation."""
        try:
            if operation.kind == "benchmark" or operation.target == "local":
                self._validate_docker()
            if operation.target == "local":
                self._validate_local_gpu()
            else:
                self._validate_remote_target(operation.target)
        except PreflightError as error:
            raise TargetUnavailable(operation.target) from error

    def _validate_target_configuration(self) -> None:
        for target in self.catalog.targets:
            for member in _REMOTE_TARGET_MEMBERS.get(target, ()):
                if member not in self.settings.remote_hosts:
                    raise PreflightError(f"SSH host for {member} is not configured")

    def _validate_docker(self) -> None:
        self._check(["docker", "info"], "Docker access is unavailable", timeout=15)

    def _validate_local_gpu(self) -> None:
        gpu = self._check(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            "the workstation GPU is unavailable",
        )
        if gpu.stdout.decode("utf-8", errors="replace").strip() != (
            "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
        ):
            raise PreflightError("the workstation GPU identity is unexpected")

    def _validate_remote_target(self, target: str) -> None:
        members = _REMOTE_TARGET_MEMBERS.get(target)
        if members is None:
            raise PreflightError("the operation target is unsupported")
        for member in members:
            host = self.settings.remote_hosts.get(member)
            if host is None:
                raise PreflightError(f"SSH host for {member} is not configured")
            self._check(
                [
                    "ssh",
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=yes",
                    "-o",
                    "ForwardAgent=no",
                    "-o",
                    "ClearAllForwardings=yes",
                    "-o",
                    "RequestTTY=no",
                    "-o",
                    "ConnectTimeout=5",
                    "-o",
                    "ConnectionAttempts=1",
                    host,
                    "exec true",
                ],
                f"SSH preflight failed for {host}",
                timeout=8,
            )

    def _validate_binding(self) -> None:
        server = self.settings.server
        parsed = urlsplit(self.settings.control.allowed_origin)
        wildcard_hosts = {"0.0.0.0", "::", "[::]", ""}
        if server.host in wildcard_hosts:
            raise PreflightError("controls cannot bind to a wildcard address")
        if parsed.scheme == "http":
            try:
                origin_addresses = {
                    item[4][0]
                    for item in socket.getaddrinfo(parsed.hostname, parsed.port, type=socket.SOCK_STREAM)
                }
                bind_addresses = {
                    item[4][0]
                    for item in socket.getaddrinfo(server.host, server.port, type=socket.SOCK_STREAM)
                }
            except OSError as error:
                raise PreflightError("the control origin or bind address cannot be resolved") from error
            if parsed.port != server.port or origin_addresses.isdisjoint(bind_addresses):
                raise PreflightError("the HTTP control origin does not match the bind address")
            if not any(address.startswith("100.") for address in bind_addresses):
                raise PreflightError("plaintext controls must bind to a Tailscale IPv4 address")
        elif not server.host.startswith("127.") and server.host != "::1" and server.host != "localhost":
            raise PreflightError("TLS-proxied controls must bind to loopback")

    def _validate_writable_directory(self, directory: Path) -> None:
        try:
            canonical = directory.resolve(strict=True)
        except OSError as error:
            raise PreflightError("a required control data directory is unavailable") from error
        if canonical != directory or not directory.is_dir():
            raise PreflightError("a required control data directory is not canonical")
        probe = directory / f".dashboard-preflight-{uuid.uuid4().hex}"
        descriptor: int | None = None
        try:
            descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.write(descriptor, b"ok\n")
            os.fsync(descriptor)
        except OSError as error:
            raise PreflightError("a required control data directory is not writable") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                probe.unlink(missing_ok=True)
            except OSError:
                pass

    def _check(self, argv: list[str], message: str, *, timeout: int = 10) -> subprocess.CompletedProcess[bytes]:
        try:
            completed = self._run(
                argv,
                cwd=self.commands.root,
                env=dict(self.commands.base_environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                start_new_session=True,
                shell=False,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PreflightError(message) from error
        if completed.returncode != 0:
            raise PreflightError(message)
        return completed
