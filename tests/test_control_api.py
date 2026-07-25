"""Authenticated control HTTP policy and schema contracts."""

from __future__ import annotations

import base64
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from dgx_dashboard import create_app
from dgx_dashboard.control.catalog import ServeRecipe
from dgx_dashboard.control.manager import RunConflict, RunNotFound, RunTransitionConflict


class _Monitoring:
    def collect(self):
        return {"timestamp": "now", "gpus": [], "servers": [], "system": {}}


class _Benchmarks:
    def get(self):
        return {}

    def run_links(self, _run_id):
        return []

    def read_indexed_result(self, _token):
        raise FileNotFoundError


class _Catalog:
    targets = ("local", "spark2", "cluster")

    def __init__(self, root: Path) -> None:
        self.recipe = ServeRecipe(
            target="local",
            engine="vllm",
            artifact="model_a",
            profile="rtx6000",
            served="served-model",
            container_name="served-model-container",
            package=root,
        )

    def get(self, target, engine, artifact):
        if (target, engine, artifact) != self.recipe.key:
            raise KeyError("unknown serving recipe")
        return self.recipe

    def public(self):
        return {
            "enabled": True,
            "targets": list(self.targets),
            "recipes": [self.recipe.public()],
        }


class _Manager:
    def __init__(self) -> None:
        self.created = []
        self.conflict = False

    def submit(self, operation):
        if self.conflict:
            raise RunConflict("00000000-0000-4000-8000-000000000001")
        self.created.append(operation)
        return {"id": "00000000-0000-4000-8000-000000000002", "state": "running"}

    def list(self, _limit):
        return []

    def get(self, run_id):
        raise RunNotFound(run_id)

    def read_log(self, run_id, _offset, _limit):
        raise RunNotFound(run_id)

    def cancel(self, run_id):
        if run_id.endswith("3"):
            raise RunTransitionConflict("terminal runs cannot be cancelled")
        if not run_id.endswith("2"):
            raise RunNotFound(run_id)
        return {"id": run_id, "state": "cancel_requested"}


class _Control:
    def __init__(self, root: Path) -> None:
        self.catalog = _Catalog(root)
        self.manager = _Manager()

    def catalog_payload(self):
        return self.catalog.public()

    def serving_status(self):
        return {"targets": [], "reconciliation": []}

    def serving_logs(self, target, engine, artifact, lines):
        self.catalog.get(target, engine, artifact)
        return {"target": target, "engine": engine, "artifact": artifact, "lines": lines, "text": "ok\n", "truncated": False}


def _auth() -> str:
    token = base64.b64encode(b"operator:correct horse").decode("ascii")
    return f"Basic {token}"


def _client(settings, tmp_path):
    secured = replace(
        settings,
        server=replace(settings.server, auth_user="operator", auth_password="correct horse"),
        control=replace(
            settings.control,
            enabled=True,
            allowed_origin="http://127.0.0.1:9000",
            wrapper_root=tmp_path,
        ),
    )
    control = _Control(tmp_path)
    app = create_app(
        secured,
        monitoring=_Monitoring(),
        benchmarks=_Benchmarks(),
        control=control,
    )
    app.config.update(TESTING=True)
    return app.test_client(), control


def _mutation_headers(**extra):
    return {
        "Authorization": _auth(),
        "Origin": "http://127.0.0.1:9000",
        **extra,
    }


def _serving_request():
    return {
        "kind": "serving",
        "action": "start",
        "target": "local",
        "engine": "vllm",
        "artifact": "model_a",
    }


def test_catalog_is_authenticated_and_contains_only_safe_metadata(settings, tmp_path):
    client, _ = _client(settings, tmp_path)
    assert client.get("/api/control/catalog").status_code == 401

    response = client.get("/api/control/catalog", headers={"Authorization": _auth()})
    assert response.status_code == 200
    recipe = response.get_json()["recipes"][0]
    assert set(recipe) == {"target", "engine", "artifact", "profile", "served"}
    assert str(tmp_path) not in response.get_data(as_text=True)


def test_mutation_requires_auth_json_and_exact_origin(settings, tmp_path):
    client, _ = _client(settings, tmp_path)
    payload = _serving_request()

    assert client.post("/api/runs", json=payload).status_code == 401
    assert client.post("/api/runs", json=payload, headers={"Authorization": _auth()}).status_code == 403
    assert client.post(
        "/api/runs",
        data="{}",
        headers=_mutation_headers(**{"Content-Type": "text/plain"}),
    ).status_code == 400
    assert client.post(
        "/api/runs",
        json=payload,
        headers={**_mutation_headers(), "Origin": "http://127.0.0.1:9001"},
    ).status_code == 403


def test_strict_serving_request_returns_202_and_location(settings, tmp_path):
    client, control = _client(settings, tmp_path)
    response = client.post("/api/runs", json=_serving_request(), headers=_mutation_headers())

    assert response.status_code == 202
    assert response.headers["Location"] == "/api/runs/00000000-0000-4000-8000-000000000002"
    assert response.get_json() == {
        "id": "00000000-0000-4000-8000-000000000002",
        "state": "running",
    }
    assert control.manager.created[0].recipe == control.catalog.recipe
    assert "Access-Control-Allow-Origin" not in response.headers


def test_unknown_fields_recipe_and_resource_conflict_are_stable(settings, tmp_path):
    client, control = _client(settings, tmp_path)
    invalid = {**_serving_request(), "argv": ["rm", "-rf"]}
    response = client.post("/api/runs", json=invalid, headers=_mutation_headers())
    assert response.status_code == 400
    assert response.get_json()["code"] == "invalid_request"

    missing = {**_serving_request(), "artifact": "missing"}
    response = client.post("/api/runs", json=missing, headers=_mutation_headers())
    assert response.status_code == 404
    assert response.get_json()["code"] == "recipe_not_found"

    control.manager.conflict = True
    response = client.post("/api/runs", json=_serving_request(), headers=_mutation_headers())
    assert response.status_code == 409
    assert response.get_json()["occupying_run_id"].endswith("1")


def test_cancel_requires_empty_object_and_terminal_conflicts(settings, tmp_path):
    client, _ = _client(settings, tmp_path)
    run_id = "00000000-0000-4000-8000-000000000002"
    assert client.post(
        f"/api/runs/{run_id}/cancel",
        json={"force": True},
        headers=_mutation_headers(),
    ).status_code == 400
    response = client.post(f"/api/runs/{run_id}/cancel", json={}, headers=_mutation_headers())
    assert response.status_code == 202
    assert response.get_json()["state"] == "cancel_requested"

    terminal = "00000000-0000-4000-8000-000000000003"
    assert client.post(f"/api/runs/{terminal}/cancel", json={}, headers=_mutation_headers()).status_code == 409


def test_request_body_and_read_queries_are_bounded(settings, tmp_path):
    client, _ = _client(settings, tmp_path)
    response = client.post(
        "/api/runs",
        data='{"padding":"' + "x" * 17_000 + '"}',
        headers=_mutation_headers(**{"Content-Type": "application/json"}),
    )
    assert response.status_code == 413
    assert response.get_json()["code"] == "request_too_large"

    assert client.get("/api/runs?limit=101", headers={"Authorization": _auth()}).status_code == 400
    response = client.get(
        "/api/serving/local/vllm/model_a/logs?lines=1001",
        headers={"Authorization": _auth()},
    )
    assert response.status_code == 400
