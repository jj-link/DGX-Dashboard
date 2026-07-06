"""Smoke tests — verify routes respond without error."""

import pytest

def test_root(client):
    """GET / should return 200."""
    resp = client.get("/")
    assert resp.status_code == 200

def test_api_stats(client):
    """GET /api/stats should return 200 JSON."""
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    assert resp.is_json

def test_api_benchmarks(client):
    """GET /api/benchmarks should return 200 JSON."""
    resp = client.get("/api/benchmarks")
    assert resp.status_code == 200
    assert resp.is_json
