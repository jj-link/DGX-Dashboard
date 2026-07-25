"""Read-only dashboard routes."""

from __future__ import annotations

import hmac
from typing import Protocol

from flask import Blueprint, Response, jsonify, render_template, request

from dgx_dashboard.config import DashboardSettings


class MonitoringReader(Protocol):
    def collect(self) -> dict[str, object]: ...


class BenchmarkReader(Protocol):
    def get(self) -> dict[str, object]: ...


def create_blueprint(
    settings: DashboardSettings,
    monitoring: MonitoringReader,
    benchmarks: BenchmarkReader,
) -> Blueprint:
    blueprint = Blueprint("dashboard", __name__)

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

    @blueprint.get("/")
    def index() -> str:
        return render_template(
            "index.html",
            refresh_interval=settings.server.refresh_interval,
        )

    @blueprint.get("/api/benchmarks")
    def api_benchmarks() -> Response:
        return jsonify(benchmarks.get() or {})

    @blueprint.get("/api/stats")
    def api_stats() -> Response:
        return jsonify(monitoring.collect())

    return blueprint
