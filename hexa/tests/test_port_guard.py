"""Tests for the launch-time port guard (fail fast on a busy port)."""
from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest

from hexa.__main__ import port_status


@pytest.fixture
def occupied_port() -> tuple[int, socket.socket]:
    """A live bound listener on an ephemeral port (simulating another app)."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))  # OS picks a free ephemeral port
    listener.listen(1)
    port = listener.getsockname()[1]
    yield port, listener
    listener.close()


class TestPortStatus:
    def test_free_port_reports_available(self) -> None:
        """An unused ephemeral port must be seen as free."""
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        free, holder = port_status("127.0.0.1", port)
        assert free is True
        assert holder is None

    def test_busy_port_reports_unavailable(self, occupied_port) -> None:
        """A port held by another listener must be seen as busy."""
        port, _listener = occupied_port
        free, holder = port_status("127.0.0.1", port)
        assert free is False
        assert holder is not None

    def test_busy_port_by_unknown_app_says_so(self, occupied_port) -> None:
        """When /health does not answer, the holder is 'another application'."""
        port, _listener = occupied_port
        with patch("httpx.get", side_effect=ConnectionError("refused")):
            free, holder = port_status("127.0.0.1", port)
        assert free is False
        assert "another application" in holder

    def test_holder_identifies_running_hexa(self, occupied_port) -> None:
        """A /health payload with service=hexa is recognized as Hexa."""
        port, _listener = occupied_port
        response = MagicMock()
        response.json.return_value = {
            "status": "ok",
            "service": "hexa",
            "version": "0.1.0",
            "uptime_seconds": 42.0,
        }
        with patch("httpx.get", return_value=response):
            free, holder = port_status("127.0.0.1", port)
        assert free is False
        assert "running Hexa instance" in holder
        assert "uptime 42 s" in holder

    def test_holder_probe_never_raises(self, occupied_port) -> None:
        """Unexpected errors during the real health probe must not propagate."""
        port, _listener = occupied_port
        with patch("httpx.get", side_effect=RuntimeError("unexpected")):
            free, holder = port_status("127.0.0.1", port)
        assert free is False
        assert "another application" in holder
