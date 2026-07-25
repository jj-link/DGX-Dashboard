"""Dashboard read routes and fail-closed control API."""

from __future__ import annotations

import hmac
from typing import Any, Protocol

from flask import Blueprint, Response, jsonify, render_template, request
from werkzeug.exceptions import BadRequest

from dgx_dashboard.config import DashboardSettings
from dgx_dashboard.control.manager import (
    LaunchFailure,
    PersistenceFailure,
    RunConflict,
    RunNotFound,
    RunTransitionConflict,
)
from dgx_dashboard.control.requests import (
    RequestValidationError,
    UnknownRecipeError,
    validate_operation,
)
from dgx_dashboard.control.service import AdapterError, ControlService


class MonitoringReader(Protocol):
    def collect(self) -> dict[str, object]: ...


class BenchmarkReader(Protocol):
    def get(self) -> dict[str, object]: ...

    def run_links(self, run_id: str) -> list[dict[str, str]]: ...

    def read_indexed_result(self, token: str) -> bytes: ...


class ApiProblem(RuntimeError):
    def __init__(self, status: int, code: str, message: str, **details: object) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details


def create_blueprint(
    settings: DashboardSettings,
    monitoring: MonitoringReader,
    benchmarks: BenchmarkReader,
    control: ControlService | None = None,
) -> Blueprint:
    blueprint = Blueprint("dashboard", __name__)

    @blueprint.errorhandler(ApiProblem)
    def api_problem(error: ApiProblem) -> tuple[Response, int]:
        body: dict[str, object] = {
            "code": error.code,
            "error": error.message,
        }
        body.update(error.details)
        return jsonify(body), error.status

    @blueprint.before_app_request
    def require_authentication() -> Response | None:
        expected_user = settings.server.auth_user
        expected_password = settings.server.auth_password
        if not expected_user:
            return None
        authorization = request.authorization
        if authorization is None:
            authenticated = False
        else:
            authenticated = hmac.compare_digest(authorization.username or "", expected_user)
            authenticated &= hmac.compare_digest(authorization.password or "", expected_password)
        if authenticated:
            return None
        return Response(
            "Unauthorized",
            401,
            {"WWW-Authenticate": 'Basic realm="Login Required"'},
        )

    def mutation_json() -> dict[str, Any]:
        if control is None or not settings.control.enabled:
            raise ApiProblem(403, "controls_disabled", "mutation controls are disabled")
        if request.mimetype != "application/json":
            raise ApiProblem(400, "invalid_content_type", "Content-Type must be application/json")
        origin = request.headers.get("Origin", "")
        if not origin or not hmac.compare_digest(origin, settings.control.allowed_origin):
            raise ApiProblem(403, "origin_forbidden", "request Origin is not allowed")
        try:
            payload = request.get_json(cache=True, silent=False)
        except BadRequest as error:
            raise ApiProblem(400, "invalid_json", "request body is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ApiProblem(400, "invalid_request", "request body must be a JSON object")
        return payload

    def query_integer(name: str, default: int, minimum: int, maximum: int) -> int:
        raw = request.args.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError as error:
            raise ApiProblem(400, "invalid_query", f"{name} must be an integer") from error
        if not minimum <= value <= maximum:
            raise ApiProblem(400, "invalid_query", f"{name} must be between {minimum} and {maximum}")
        return value

    @blueprint.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            refresh_interval=settings.server.refresh_interval,
        )

    @blueprint.get("/api/benchmarks")
    def api_benchmarks() -> Response:
        return jsonify(benchmarks.get() or {})

    @blueprint.get("/api/benchmarks/results/<token>")
    def api_benchmark_result(token: str) -> Response:
        try:
            payload = benchmarks.read_indexed_result(token)
        except FileNotFoundError as error:
            raise ApiProblem(404, "result_not_found", "benchmark result does not exist") from error
        except ValueError as error:
            raise ApiProblem(413, "result_too_large", "benchmark result exceeds the response limit") from error
        return Response(payload, 200, content_type="application/json; charset=utf-8")

    @blueprint.get("/api/stats")
    def api_stats() -> Response:
        return jsonify(monitoring.collect())

    @blueprint.get("/api/control/catalog")
    def api_control_catalog() -> Response:
        if control is None:
            return jsonify({"enabled": False, "targets": [], "recipes": []})
        return jsonify(control.catalog_payload())

    @blueprint.get("/api/serving")
    def api_serving() -> Response:
        if control is None:
            return jsonify({"enabled": False, "targets": [], "reconciliation": []})
        return jsonify(control.serving_status())

    @blueprint.get("/api/serving/<target>/<engine>/<artifact>/logs")
    def api_serving_logs(target: str, engine: str, artifact: str) -> Response:
        if control is None:
            raise ApiProblem(403, "controls_disabled", "serving adapters are disabled")
        lines = query_integer("lines", 100, 1, 1000)
        try:
            payload = control.serving_logs(target, engine, artifact, lines)
        except KeyError as error:
            raise ApiProblem(404, "recipe_not_found", "serving recipe does not exist") from error
        except AdapterError as error:
            raise ApiProblem(500, "adapter_failed", "serving log adapter failed") from error
        return jsonify(payload)

    @blueprint.get("/api/runs")
    def api_runs() -> Response:
        limit = query_integer("limit", 50, 1, 100)
        return jsonify([] if control is None else control.manager.list(limit))

    @blueprint.get("/api/runs/<run_id>")
    def api_run(run_id: str) -> Response:
        if control is None:
            raise ApiProblem(404, "run_not_found", "run does not exist")
        try:
            return jsonify(control.manager.get(run_id))
        except RunNotFound as error:
            raise ApiProblem(404, "run_not_found", "run does not exist") from error

    @blueprint.get("/api/runs/<run_id>/log")
    def api_run_log(run_id: str) -> Response:
        if control is None:
            raise ApiProblem(404, "run_not_found", "run does not exist")
        offset = query_integer("offset", 0, 0, 9_223_372_036_854_775_807)
        limit = query_integer("limit", 65_536, 1, 65_536)
        try:
            return jsonify(control.manager.read_log(run_id, offset, limit))
        except RunNotFound as error:
            raise ApiProblem(404, "run_not_found", "run does not exist") from error
        except PersistenceFailure as error:
            raise ApiProblem(500, "log_unavailable", "run log is unavailable") from error

    @blueprint.post("/api/runs")
    def api_create_run() -> tuple[Response, int, dict[str, str]]:
        payload = mutation_json()
        assert control is not None
        try:
            operation = validate_operation(payload, control.catalog)
            record = control.manager.submit(operation)
        except UnknownRecipeError as error:
            raise ApiProblem(404, "recipe_not_found", str(error)) from error
        except RequestValidationError as error:
            raise ApiProblem(400, "invalid_request", str(error)) from error
        except RunConflict as error:
            raise ApiProblem(
                409,
                "resource_conflict",
                "operation conflicts with an active run",
                occupying_run_id=error.occupying_run_id,
            ) from error
        except (LaunchFailure, PersistenceFailure) as error:
            raise ApiProblem(500, "operation_launch_failed", "operation could not be launched durably") from error
        location = f"/api/runs/{record['id']}"
        return jsonify({"id": record["id"], "state": record["state"]}), 202, {"Location": location}

    @blueprint.post("/api/runs/<run_id>/cancel")
    def api_cancel_run(run_id: str) -> tuple[Response, int]:
        payload = mutation_json()
        if payload:
            raise ApiProblem(400, "invalid_request", "cancel request body must be {}")
        assert control is not None
        try:
            record = control.manager.cancel(run_id)
        except RunNotFound as error:
            raise ApiProblem(404, "run_not_found", "run does not exist") from error
        except RunTransitionConflict as error:
            raise ApiProblem(409, "transition_conflict", str(error)) from error
        except PersistenceFailure as error:
            raise ApiProblem(500, "persistence_failed", "cancellation could not be persisted") from error
        return jsonify({"id": record["id"], "state": record["state"]}), 202

    return blueprint
