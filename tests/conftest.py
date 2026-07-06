"""conftest.py — shared fixtures for DGX Dashboard tests."""

import os
from unittest.mock import patch, MagicMock
import pytest

@pytest.fixture
def mock_network():
    """Patch all network/hardware calls so tests don't need live services."""
    mocks = {}

    mock_requests = patch("requests.get", return_value=MagicMock(json=lambda: {}, status_code=200))
    mock_subprocess = patch("subprocess.run", return_value=MagicMock(returncode=0, stdout=b"", stderr=b""))
    mock_socket = patch("socket.socket", side_effect=ConnectionRefusedError("mocked"))

    mocks["requests"] = mock_requests.start()
    mocks["subprocess"] = mock_subprocess.start()
    mocks["socket"] = mock_socket.start()

    yield mocks

    for m in mocks.values():
        m.stop()

@pytest.fixture
def app(mock_network):
    """Create the Flask app for testing."""
    import server
    server.app.config["TESTING"] = True
    server.app.config["DEBUG"] = False
    return server.app

@pytest.fixture
def client(app):
    """Test client for the Flask app."""
    return app.test_client()
