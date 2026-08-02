"""Durable single-process operation manager with exact resource leases."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping

from dgx_dashboard.control.catalog import ServingCatalog
from dgx_dashboard.control.commands import CommandBuilder, CommandSpec, OperationPlan
from dgx_dashboard.control.requests import (
    OperationRequest,
    RequestValidationError,
    validate_persisted_operation,
)


RUN_STATES = frozenset(
    {
        "queued",
        "running",
        "succeeded",
        "failed",
        "timed_out",
        "cancel_requested",
        "cancelled",
        "interrupted",
        "launch_failed",
        "cleanup_failed",
    }
)
TERMINAL_STATES = frozenset(
    {"succeeded", "failed", "timed_out", "cancelled", "interrupted", "launch_failed", "cleanup_failed"}
)
_RUN_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")
_CONTAINER_ID_RE = re.compile(r"[0-9a-f]{12,64}\Z")
_META_KEYS = {
    "schema",
    "id",
    "kind",
    "state",
    "request",
    "resources",
    "created_at",
    "started_at",
    "finished_at",
    "exit_code",
    "error_code",
    "results",
}
_ERROR_CODES = frozenset(
    {
        "cleanup_failed",
        "interrupted",
        "launch_failed",
        "operation_timeout",
        "persistence_failed",
        "process_exit",
        "verification_launch_failed",
    }
)
_RESULT_URL_RE = re.compile(r"/api/benchmarks/results/[A-Za-z0-9_-]+\Z")


class RunManagerError(RuntimeError):
    """Base class for stable run-manager failures."""


class RunConflict(RunManagerError):
    def __init__(self, occupying_run_id: str) -> None:
        super().__init__("operation conflicts with an active run")
        self.occupying_run_id = occupying_run_id


class RunNotFound(RunManagerError):
    pass


class RunTransitionConflict(RunManagerError):
    pass


class PersistenceFailure(RunManagerError):
    pass


class LaunchFailure(RunManagerError):
    pass


@dataclass
class _ActiveRun:
    operation: OperationRequest
    plan: OperationPlan
    process: Any
    log: BinaryIO
    cancel: threading.Event


PopenFactory = Callable[..., Any]
RunFactory = Callable[..., subprocess.CompletedProcess[bytes]]
ResultResolver = Callable[[str], list[dict[str, str]]]
OperationPreflight = Callable[[OperationRequest], None]
Terminator = Callable[[Any, float], None]


class RunManager:
    """Reserve resources, launch operations, and persist every state transition."""

    def __init__(
        self,
        state_dir: Path,
        catalog: ServingCatalog,
        commands: CommandBuilder,
        *,
        retention: int,
        popen_factory: PopenFactory = subprocess.Popen,
        run_factory: RunFactory = subprocess.run,
        terminate_process: Terminator | None = None,
        result_resolver: ResultResolver | None = None,
        operation_preflight: OperationPreflight | None = None,
        cancel_grace: float = 10.0,
    ) -> None:
        self.state_dir = state_dir.resolve(strict=False)
        self.catalog = catalog
        self.commands = commands
        self.retention = retention
        self._popen = popen_factory
        self._run = run_factory
        self._terminate = terminate_process or self._terminate_group
        self._result_resolver = result_resolver or (lambda _run_id: [])
        self._operation_preflight = operation_preflight
        self._cancel_grace = cancel_grace
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._active: dict[str, _ActiveRun] = {}
        self._leases: dict[str, str] = {}
        self._reconciliation: list[dict[str, str]] = []
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
        self._load_records()
        self._reconcile_nonterminal()
        with self._lock:
            self._prune_terminal_locked()

    @property
    def reconciliation(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(item) for item in self._reconciliation[-100:]]

    def submit(self, operation: OperationRequest) -> dict[str, Any]:
        if self._operation_preflight is not None:
            self._operation_preflight(operation)
        run_id = str(uuid.uuid4())
        now = self._timestamp()
        record = {
            "schema": 1,
            "id": run_id,
            "kind": operation.kind,
            "state": "queued",
            "request": dict(operation.public),
            "resources": sorted(operation.resources),
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "error_code": None,
            "results": [],
        }
        plan = self.commands.plan(operation, run_id)

        with self._lock:
            occupant = self._find_conflict_locked(operation.resources)
            if occupant is not None:
                raise RunConflict(occupant)
            self._reserve_locked(run_id, operation.resources)
            self._records[run_id] = record
            try:
                self._create_run_files_locked(record)
                log = self._open_log(run_id)
            except Exception as error:
                self._records.pop(run_id, None)
                self._release_locked(run_id)
                self._remove_run_dir(run_id)
                if isinstance(error, PersistenceFailure):
                    raise
                raise PersistenceFailure("cannot create durable run state") from error

            try:
                process = self._spawn(plan.command, log)
            except Exception as error:
                record["state"] = "launch_failed"
                record["finished_at"] = self._timestamp()
                record["error_code"] = "launch_failed"
                try:
                    self._persist_locked(record)
                finally:
                    log.close()
                    self._release_locked(run_id)
                    self._prune_terminal_locked()
                raise LaunchFailure("operation process could not be launched") from error

            record["state"] = "running"
            record["started_at"] = self._timestamp()
            try:
                self._persist_locked(record)
            except PersistenceFailure as error:
                self._terminate(process, self._cancel_grace)
                record["state"] = "failed"
                record["finished_at"] = self._timestamp()
                record["error_code"] = "persistence_failed"
                log.close()
                self._release_locked(run_id)
                raise LaunchFailure("operation was stopped because running state could not be persisted") from error
            active = _ActiveRun(
                operation=operation,
                plan=plan,
                process=process,
                log=log,
                cancel=threading.Event(),
            )
            self._active[run_id] = active
            thread = threading.Thread(
                target=self._watch,
                args=(run_id,),
                name=f"dashboard-run-{run_id[:8]}",
                daemon=True,
            )
            thread.start()
            return self._public_record(record)

    def cancel(self, run_id: str) -> dict[str, Any]:
        self._validate_run_id(run_id)
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                raise RunNotFound("run does not exist")
            if record["state"] in TERMINAL_STATES:
                raise RunTransitionConflict("terminal runs cannot be cancelled")
            if record["state"] == "cancel_requested":
                raise RunTransitionConflict("run cancellation was already requested")
            active = self._active.get(run_id)
            if active is None:
                raise RunTransitionConflict("run is not active in this process")
            previous_state = record["state"]
            record["state"] = "cancel_requested"
            try:
                self._persist_locked(record)
            except PersistenceFailure:
                record["state"] = previous_state
                raise
            active.cancel.set()
            return self._public_record(record)

    def list(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            records = sorted(self._records.values(), key=lambda item: item["created_at"], reverse=True)
            return [self._public_record(record) for record in records[:limit]]

    def get(self, run_id: str) -> dict[str, Any]:
        self._validate_run_id(run_id)
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                raise RunNotFound("run does not exist")
            return self._public_record(record)

    def read_log(self, run_id: str, offset: int, limit: int) -> dict[str, Any]:
        self._validate_run_id(run_id)
        with self._lock:
            if run_id not in self._records:
                raise RunNotFound("run does not exist")
        path = self.state_dir / run_id / "output.log"
        try:
            with path.open("rb") as stream:
                size = os.fstat(stream.fileno()).st_size
                start = min(offset, size)
                stream.seek(start)
                data = stream.read(limit)
                size = os.fstat(stream.fileno()).st_size
        except OSError as error:
            raise PersistenceFailure("run log is unavailable") from error
        next_offset = start + len(data)
        return {
            "offset": start,
            "next_offset": next_offset,
            "data": data.decode("utf-8", errors="replace"),
            "eof": next_offset >= size,
        }

    def latest_serving_recipes(self) -> dict[str, tuple[str, str, str | None]]:
        with self._lock:
            records = sorted(self._records.values(), key=lambda item: item["created_at"], reverse=True)
            latest: dict[str, tuple[str, str, str | None]] = {}
            for record in records:
                request = record["request"]
                target = request["target"]
                if (
                    record["kind"] != "serving"
                    or record["state"] != "succeeded"
                    or target in latest
                ):
                    continue
                operation = validate_persisted_operation(request, self.catalog)
                latest[target] = (
                    request["engine"],
                    request["artifact"],
                    operation.launch_profile,
                )
            return latest

    def _watch(self, run_id: str) -> None:
        with self._lock:
            active = self._active[run_id]
        outcome, exit_code = self._wait_active(active, active.plan.command.timeout)

        if outcome == "exited" and exit_code == 0 and active.plan.verify is not None:
            verification_started = False
            with self._lock:
                record = self._records[run_id]
                if record["state"] == "cancel_requested":
                    outcome = "cancelled"
                else:
                    try:
                        active.process = self._spawn(active.plan.verify, active.log)
                    except Exception:
                        outcome = "verification_launch_failed"
                    else:
                        verification_started = True
            if verification_started:
                outcome, exit_code = self._wait_active(active, active.plan.verify.timeout)

        should_cleanup = outcome in {"cancelled", "timed_out"}
        if (
            active.operation.kind == "serving"
            and active.operation.action == "start"
            and (outcome != "exited" or exit_code != 0)
        ):
            should_cleanup = True
        cleanup_ok = self._cleanup(active) if should_cleanup else True

        resolved_results: list[dict[str, str]] = []
        if active.operation.kind == "benchmark":
            try:
                resolved_results = self._safe_results(self._result_resolver(run_id))
            except Exception:
                resolved_results = []

        with self._lock:
            record = self._records[run_id]
            if not cleanup_ok:
                state, error_code = "cleanup_failed", "cleanup_failed"
            elif outcome == "cancelled" or record["state"] == "cancel_requested":
                state, error_code = "cancelled", None
            elif outcome == "timed_out":
                state, error_code = "timed_out", "operation_timeout"
            elif outcome == "verification_launch_failed":
                state, error_code = "failed", "verification_launch_failed"
            elif exit_code == 0:
                state, error_code = "succeeded", None
            else:
                state, error_code = "failed", "process_exit"
            record["state"] = state
            record["finished_at"] = self._timestamp()
            record["exit_code"] = exit_code
            record["error_code"] = error_code
            if record["kind"] == "benchmark":
                record["results"] = resolved_results
            self._active.pop(run_id, None)
            self._release_locked(run_id)
            try:
                self._persist_locked(record)
            except PersistenceFailure:
                record["state"] = "failed"
                record["error_code"] = "persistence_failed"
            active.log.close()
            self._prune_terminal_locked()

    def _wait_active(self, active: _ActiveRun, timeout: int) -> tuple[str, int | None]:
        deadline = time.monotonic() + timeout
        while True:
            exit_code = active.process.poll()
            if exit_code is not None:
                return "exited", int(exit_code)
            if active.cancel.wait(timeout=0.1):
                self._terminate(active.process, self._cancel_grace)
                exit_code = active.process.poll()
                return "cancelled", int(exit_code) if exit_code is not None else None
            if time.monotonic() >= deadline:
                self._terminate(active.process, self._cancel_grace)
                exit_code = active.process.poll()
                return "timed_out", int(exit_code) if exit_code is not None else None

    def _cleanup(self, active: _ActiveRun) -> bool:
        if active.plan.benchmark_label is not None:
            return self._cleanup_benchmark(active.plan.benchmark_label, active.log)
        if not active.plan.cleanup:
            return True
        if self._run_to_log(active.plan.cleanup[0], active.log) != 0:
            return False
        if len(active.plan.cleanup) != 2:
            return False
        returncode, output = self._run_command_capture(active.plan.cleanup[1], active.log)
        states = re.findall(rb"\bstate=([a-z]+)\b", output)
        return returncode == 0 and bool(states) and all(state == b"absent" for state in states)

    def _cleanup_benchmark(self, label: str, log: BinaryIO) -> bool:
        environment = dict(self.commands.base_environment)
        try:
            completed = self._run(
                ["docker", "container", "ls", "--all", "--quiet", "--filter", f"label={label}"],
                cwd=self.commands.root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
                shell=False,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        output = completed.stdout[:65_536]
        log.write(output)
        if completed.returncode != 0 or len(completed.stdout) > 65_536:
            return False
        container_ids = [line.decode("ascii", errors="ignore") for line in output.splitlines() if line]
        if any(not _CONTAINER_ID_RE.fullmatch(container_id) for container_id in container_ids):
            return False
        if not container_ids:
            return True
        command = CommandSpec(
            argv=("docker", "container", "rm", "--force", *container_ids),
            cwd=self.commands.root,
            environment=self.commands.base_environment,
            timeout=60,
        )
        return self._run_command_capture(command, log)[0] == 0

    def _run_to_log(self, command: CommandSpec, log: BinaryIO) -> int:
        try:
            process = self._spawn(command, log)
        except Exception:
            return -1
        deadline = time.monotonic() + command.timeout
        while process.poll() is None:
            if time.monotonic() >= deadline:
                self._terminate(process, self._cancel_grace)
                return -1
            time.sleep(0.1)
        return int(process.poll())

    def _run_command_capture(self, command: CommandSpec, log: BinaryIO) -> tuple[int, bytes]:
        try:
            completed = self._run(
                list(command.argv),
                cwd=command.cwd,
                env=dict(command.environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
                shell=False,
                timeout=command.timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return -1, b""
        output = completed.stdout[:262_144]
        log.write(output)
        if len(completed.stdout) > 262_144:
            return -1, output
        return int(completed.returncode), output

    def _spawn(self, command: CommandSpec, log: BinaryIO) -> Any:
        return self._popen(
            list(command.argv),
            cwd=command.cwd,
            env=dict(command.environment),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
            shell=False,
        )

    def _load_records(self) -> None:
        try:
            candidates = list(self.state_dir.iterdir())
        except OSError as error:
            raise PersistenceFailure("cannot enumerate durable run state") from error
        for directory in candidates:
            if not _RUN_ID_RE.fullmatch(directory.name):
                continue
            if directory.is_symlink() or not directory.is_dir():
                raise PersistenceFailure("durable run state contains an unsafe run directory")
            meta_path = directory / "meta.json"
            try:
                record = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise PersistenceFailure("durable run metadata is unreadable") from error
            operation = self._validate_record(record, directory.name)
            self._records[directory.name] = record
            if record["state"] not in TERMINAL_STATES:
                for resource in operation.resources:
                    if resource in self._leases:
                        raise PersistenceFailure("durable run metadata contains conflicting nonterminal leases")
                    self._leases[resource] = directory.name

    def _validate_record(self, record: Any, run_id: str) -> OperationRequest:
        if not isinstance(record, dict) or set(record) != _META_KEYS:
            raise PersistenceFailure("durable run metadata has an invalid schema")
        if record.get("schema") != 1 or record.get("id") != run_id or record.get("state") not in RUN_STATES:
            raise PersistenceFailure("durable run metadata has an invalid identity or state")
        try:
            operation = validate_persisted_operation(record.get("request"), self.catalog)
        except RequestValidationError as error:
            raise PersistenceFailure("durable run metadata contains an invalid request") from error
        if record.get("kind") != operation.kind or record.get("resources") != sorted(operation.resources):
            raise PersistenceFailure("durable run metadata contains invalid resources")
        if not isinstance(record.get("created_at"), str):
            raise PersistenceFailure("durable run metadata contains an invalid timestamp")
        if any(record.get(key) is not None and not isinstance(record.get(key), str) for key in ("started_at", "finished_at")):
            raise PersistenceFailure("durable run metadata contains an invalid timestamp")
        exit_code = record.get("exit_code")
        if exit_code is not None and (not isinstance(exit_code, int) or isinstance(exit_code, bool)):
            raise PersistenceFailure("durable run metadata contains an invalid exit code")
        if record.get("error_code") is not None and record.get("error_code") not in _ERROR_CODES:
            raise PersistenceFailure("durable run metadata contains an invalid error code")
        results = record.get("results")
        if not isinstance(results, list) or self._safe_results(results) != results:
            raise PersistenceFailure("durable run metadata contains invalid results")
        return operation

    def _reconcile_nonterminal(self) -> None:
        for run_id in sorted(self._records, key=lambda key: self._records[key]["created_at"]):
            record = self._records[run_id]
            if record["state"] in TERMINAL_STATES:
                continue
            operation = validate_persisted_operation(record["request"], self.catalog)
            new_state = "interrupted"
            if operation.kind == "serving" and operation.recipe is not None:
                if operation.action in {"start", "verify"}:
                    command = self.commands.serving(
                        operation.recipe,
                        "verify",
                        operation.launch_profile,
                    )
                    returncode, _ = self._run_capture(command, run_id)
                    if returncode == 0:
                        new_state = "succeeded"
                elif operation.action == "stop":
                    command = self.commands.serving(
                        operation.recipe,
                        "status",
                        operation.launch_profile,
                    )
                    returncode, output = self._run_capture(command, run_id)
                    states = re.findall(rb"\bstate=([a-z]+)\b", output)
                    if returncode == 0 and states and all(state == b"absent" for state in states):
                        new_state = "succeeded"
            elif operation.kind == "benchmark":
                log = self._open_log(run_id)
                try:
                    if not self._cleanup_benchmark(f"io.dgx-dashboard.run-id={run_id}", log):
                        new_state = "cleanup_failed"
                finally:
                    log.close()
            record["state"] = new_state
            record["finished_at"] = self._timestamp()
            record["error_code"] = None if new_state == "succeeded" else new_state
            self._release_locked(run_id)
            self._persist_locked(record)
            self._reconciliation.append({"id": run_id, "state": new_state})

    def _run_capture(self, command: CommandSpec, run_id: str) -> tuple[int, bytes]:
        try:
            completed = self._run(
                list(command.argv),
                cwd=command.cwd,
                env=dict(command.environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
                shell=False,
                timeout=command.timeout,
                check=False,
            )
            output = completed.stdout[:262_144]
            returncode = completed.returncode if len(completed.stdout) <= 262_144 else -1
        except (OSError, subprocess.TimeoutExpired):
            output = b""
            returncode = -1
        log = self._open_log(run_id)
        try:
            log.write(output)
        finally:
            log.close()
        return int(returncode), output

    def _create_run_files_locked(self, record: Mapping[str, Any]) -> None:
        run_dir = self.state_dir / str(record["id"])
        try:
            run_dir.mkdir(mode=0o700)
            os.chmod(run_dir, 0o700)
            self._persist_locked(record)
            descriptor = os.open(run_dir / "output.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
            os.close(descriptor)
        except OSError as error:
            raise PersistenceFailure("cannot create durable run files") from error

    def _persist_locked(self, record: Mapping[str, Any]) -> None:
        run_dir = self.state_dir / str(record["id"])
        temporary = run_dir / f".meta.{uuid.uuid4().hex}.tmp"
        payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = None
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, run_dir / "meta.json")
            os.chmod(run_dir / "meta.json", 0o640)
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise PersistenceFailure("cannot persist run metadata") from error

    def _open_log(self, run_id: str) -> BinaryIO:
        path = self.state_dir / run_id / "output.log"
        flags = os.O_WRONLY | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
            os.chmod(path, 0o640)
            return os.fdopen(descriptor, "ab", buffering=0)
        except OSError as error:
            raise PersistenceFailure("cannot open durable run log") from error

    def _find_conflict_locked(self, resources: frozenset[str]) -> str | None:
        for resource in sorted(resources):
            occupant = self._leases.get(resource)
            if occupant is not None:
                return occupant
        return None

    def _reserve_locked(self, run_id: str, resources: frozenset[str]) -> None:
        for resource in resources:
            self._leases[resource] = run_id

    def _release_locked(self, run_id: str) -> None:
        for resource, occupant in tuple(self._leases.items()):
            if occupant == run_id:
                del self._leases[resource]

    def _prune_terminal_locked(self) -> None:
        terminal = sorted(
            (record for record in self._records.values() if record["state"] in TERMINAL_STATES),
            key=lambda item: item["created_at"],
            reverse=True,
        )
        for record in terminal[self.retention :]:
            run_id = record["id"]
            if run_id in self._active:
                continue
            self._remove_run_dir(run_id)
            self._records.pop(run_id, None)

    def _remove_run_dir(self, run_id: str) -> None:
        if not _RUN_ID_RE.fullmatch(run_id):
            return
        directory = self.state_dir / run_id
        if directory.is_symlink():
            return
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
        except OSError:
            pass

    @staticmethod
    def _safe_results(results: list[dict[str, str]]) -> list[dict[str, str]]:
        safe: list[dict[str, str]] = []
        for result in results[:100]:
            if not isinstance(result, dict) or set(result) != {"label", "url"}:
                continue
            label = result["label"]
            url = result["url"]
            if (
                isinstance(label, str)
                and 1 <= len(label) <= 200
                and isinstance(url, str)
                and _RESULT_URL_RE.fullmatch(url)
            ):
                safe.append({"label": label, "url": url})
        return safe

    @staticmethod
    def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": record["id"],
            "kind": record["kind"],
            "state": record["state"],
            "request": record["request"],
            "created_at": record["created_at"],
            "started_at": record["started_at"],
            "finished_at": record["finished_at"],
            "exit_code": record["exit_code"],
            "error_code": record["error_code"],
            "results": record["results"],
        }

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise RunNotFound("run does not exist")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _terminate_group(process: Any, grace: float) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
