"""Read-route and static-asset smoke contracts."""

from __future__ import annotations

import base64
from dataclasses import replace

from dgx_dashboard import create_app


class _Reader:
    def __init__(self, payload):
        self.payload = payload

    def collect(self):
        return self.payload

    def get(self):
        return self.payload


def test_root_uses_external_assets(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"<style>" not in response.data
    assert b"<script>" not in response.data
    assert b'/static/dashboard.css' in response.data
    assert b'/static/live.js' in response.data
    assert b'/static/benchmarks.js' in response.data
    assert b'data-refresh-interval="3"' in response.data


def test_static_assets_are_served(client):
    for path in ("dashboard.css", "live.js", "benchmarks.js"):
        response = client.get(f"/static/{path}")
        assert response.status_code == 200
        assert response.data


def test_api_stats_preserves_wire_schema(client, stats_payload):
    response = client.get("/api/stats")
    assert response.status_code == 200
    assert response.is_json
    assert response.get_json() == stats_payload
    assert set(response.get_json()) == {"timestamp", "gpus", "servers", "system"}
    assert set(response.get_json()["servers"][0]) == {
        "name",
        "type",
        "url",
        "online",
        "error",
        "models",
        "stats",
    }


def test_api_benchmarks_preserves_wire_schema(client, benchmark_payload):
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    assert response.is_json
    assert response.get_json() == benchmark_payload


def test_basic_auth_applies_to_reads_and_assets(settings, stats_payload, benchmark_payload):
    secured = replace(
        settings,
        server=replace(settings.server, auth_user="operator", auth_password="secret"),
    )
    application = create_app(
        secured,
        monitoring=_Reader(stats_payload),
        benchmarks=_Reader(benchmark_payload),
    )
    client = application.test_client()

    unauthorized = client.get("/api/stats")
    assert unauthorized.status_code == 401
    assert unauthorized.data == b"Unauthorized"
    assert unauthorized.headers["WWW-Authenticate"] == 'Basic realm="Login Required"'

    credentials = base64.b64encode(b"operator:secret").decode("ascii")
    authorized = client.get("/api/stats", headers={"Authorization": f"Basic {credentials}"})
    assert authorized.status_code == 200
