"""Tests for API authentication."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hexa.api import create_app
from hexa.config import Config, AudioConfig, WakeWordConfig, HotkeyConfig, STTConfig, TTSConfig, LLMConfig, MemoryConfig, APIConfig


@pytest.fixture
def test_config() -> Config:
    """Create a test config with known bearer token."""
    return Config(
        api=APIConfig(bearer_token="test-token-123"),
    )


@pytest.fixture
def client(test_config: Config) -> TestClient:
    """Create test client with patched config."""
    import hexa.api
    import hexa.config
    import hexa.router
    import hexa.tools
    import hexa.state
    import hexa.events

    # Reset global instances
    hexa.config._config = test_config
    hexa.router._router = None
    hexa.tools._registry = None
    hexa.state._state_machine = None
    hexa.events._event_bus = None

    # Create fresh registry with test tools
    registry = hexa.tools.ToolRegistry()
    registry.set_permissions({"get_time": "safe"})

    @registry.register
    def get_time() -> str:
        return "12:00 PM"

    hexa.tools._registry = registry

    app = create_app()
    return TestClient(app)


def test_no_token_returns_401(client: TestClient) -> None:
    """Test request without token returns 401."""
    response = client.get("/state")
    assert response.status_code == 401


def test_wrong_token_returns_401(client: TestClient) -> None:
    """Test request with wrong token returns 401."""
    response = client.get("/state", headers={"Authorization": "Bearer wrong-token"})
    assert response.status_code == 401


def test_valid_token_returns_200(client: TestClient) -> None:
    """Test request with valid token returns 200."""
    response = client.get("/state", headers={"Authorization": "Bearer test-token-123"})
    assert response.status_code == 200
    assert response.json()["state"] == "idle"


def test_command_endpoint(client: TestClient) -> None:
    """Test /command endpoint."""
    response = client.post(
        "/command",
        json={"text": "what time is it"},
        headers={"Authorization": "Bearer test-token-123"},
    )
    assert response.status_code == 200
    assert "12:00 PM" in response.json()["reply"]


def test_tools_endpoint(client: TestClient) -> None:
    """Test /tools endpoint."""
    response = client.get("/tools", headers={"Authorization": "Bearer test-token-123"})
    assert response.status_code == 200
    tools = response.json()["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "get_time"


def test_config_endpoint_hides_token(client: TestClient) -> None:
    """Test /config hides bearer token."""
    response = client.get("/config", headers={"Authorization": "Bearer test-token-123"})
    assert response.status_code == 200
    config = response.json()["config"]
    assert config["api"]["bearer_token"] == "***"


def test_wake_endpoint(client: TestClient) -> None:
    """Test /wake endpoint."""
    response = client.post("/wake", headers={"Authorization": "Bearer test-token-123"})
    assert response.status_code == 200
    assert response.json()["status"] == "wake simulated"


def test_stop_endpoint(client: TestClient) -> None:
    """Test /stop endpoint."""
    response = client.post("/stop", headers={"Authorization": "Bearer test-token-123"})
    assert response.status_code == 200
    assert response.json()["status"] == "stopped"


def test_websocket_requires_token(client: TestClient) -> None:
    """Test WebSocket requires valid token."""
    # Connect without token - should fail
    with pytest.raises(Exception):
        with client.websocket_connect("/events") as ws:
            pass

    # Connect with wrong token - should fail
    with pytest.raises(Exception):
        with client.websocket_connect("/events?token=wrong") as ws:
            pass

    # Connect with valid token - should succeed
    with client.websocket_connect("/events?token=test-token-123") as ws:
        # Just verify connection works, don't wait for events
        pass