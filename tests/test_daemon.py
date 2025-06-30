"""
Tests for SteadyText daemon functionality.

AIDEV-NOTE: Tests both server and client components, including connection
handling, request processing, and fallback behavior.
"""

import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from steadytext.daemon.client import DaemonClient, use_daemon
from steadytext.daemon.protocol import Request, Response, ErrorResponse
from steadytext import generate, embed, EMBEDDING_DIMENSION


# Skip these tests if pyzmq is not available
pytest.importorskip("zmq")


class TestDaemonProtocol:
    """Test protocol message serialization."""

    def test_request_serialization(self):
        req = Request(method="generate", params={"prompt": "test"})
        json_str = req.to_json()
        restored = Request.from_json(json_str)

        assert restored.method == req.method
        assert restored.params == req.params
        assert restored.id == req.id

    def test_response_serialization(self):
        resp = Response(id="test-id", result="test result")
        json_str = resp.to_json()
        restored = Response.from_json(json_str)

        assert restored.id == resp.id
        assert restored.result == resp.result
        assert restored.error is None

    def test_error_response(self):
        err = ErrorResponse(id="test-id", error="test error")
        assert err.result is None
        assert err.error == "test error"


class TestDaemonClient:
    """Test daemon client functionality."""

    def test_client_no_server(self):
        """Test client behavior when no server is running."""
        client = DaemonClient(port=59999)  # Use unlikely port
        assert not client.connect()

        with pytest.raises(ConnectionError):
            client.generate("test")

    def test_ping_no_server(self):
        """Test ping when no server is running."""
        client = DaemonClient(port=59999)
        assert not client.ping()

    @patch("steadytext.daemon.client.zmq")
    def test_client_mock_connection(self, mock_zmq):
        """Test client with mocked ZeroMQ connection."""
        # Setup mock
        mock_context = MagicMock()
        mock_socket = MagicMock()
        mock_zmq.Context.return_value = mock_context
        mock_context.socket.return_value = mock_socket

        # Mock ping response
        mock_socket.recv.return_value = (
            Response(id="test", result="pong").to_json().encode()
        )

        client = DaemonClient()
        assert client.connect()

        # Test generate
        mock_socket.recv.return_value = (
            Response(id="test", result="generated text").to_json().encode()
        )
        result = client.generate("test prompt")
        assert result == "generated text"

        # Test embed
        mock_socket.recv.return_value = (
            Response(id="test", result=[0.1] * EMBEDDING_DIMENSION).to_json().encode()
        )
        result = client.embed("test text")
        assert isinstance(result, np.ndarray)
        assert result.shape == (EMBEDDING_DIMENSION,)


class TestUseDaemon:
    """Test use_daemon context manager."""

    def test_use_daemon_no_server(self):
        """Test use_daemon when no server is running."""
        with use_daemon(port=59999) as client:
            assert client is None  # No connection

    def test_use_daemon_required(self):
        """Test use_daemon with required=True."""
        with pytest.raises(RuntimeError, match="Daemon connection required"):
            with use_daemon(port=59999, required=True):
                pass

    @patch("steadytext.daemon.client.DaemonClient")
    def test_use_daemon_mock_connection(self, mock_client_class):
        """Test use_daemon with mocked connection."""
        mock_client = MagicMock()
        mock_client.connect.return_value = True
        mock_client.host = "localhost"
        mock_client.port = 5555
        mock_client_class.return_value = mock_client

        with use_daemon() as client:
            assert client is mock_client
            # Verify the daemon client configuration
            assert os.environ.get("STEADYTEXT_DAEMON_HOST") == "localhost"
            assert os.environ.get("STEADYTEXT_DAEMON_PORT") == "5555"


class TestDaemonIntegration:
    """Test daemon integration with main API."""

    @patch("steadytext.get_daemon_client")
    def test_generate_with_daemon(self, mock_get_client):
        """Test generate() using daemon when enabled."""
        mock_client = MagicMock()
        mock_client.generate.return_value = "daemon generated text"
        mock_get_client.return_value = mock_client

        # Enable daemon by removing the DISABLE flag set in conftest.py
        disabled_was_set = "STEADYTEXT_DISABLE_DAEMON" in os.environ
        if disabled_was_set:
            del os.environ["STEADYTEXT_DISABLE_DAEMON"]
        try:
            result = generate("test prompt")
            assert result == "daemon generated text"
            mock_client.generate.assert_called_once()
        finally:
            # Restore original state
            if disabled_was_set:
                os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"

    @patch("steadytext.get_daemon_client")
    def test_generate_daemon_fallback(self, mock_get_client):
        """Test generate() fallback when daemon fails."""
        mock_client = MagicMock()
        mock_client.generate.side_effect = ConnectionError("Daemon not available")
        mock_get_client.return_value = mock_client

        # Enable daemon by removing the DISABLE flag set in conftest.py
        disabled_was_set = "STEADYTEXT_DISABLE_DAEMON" in os.environ
        if disabled_was_set:
            del os.environ["STEADYTEXT_DISABLE_DAEMON"]
        try:
            # Should fall back to direct generation
            result = generate("test prompt")
            assert result is None
        finally:
            # Restore original state
            if disabled_was_set:
                os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"

    @patch("steadytext.get_daemon_client")
    def test_embed_with_daemon(self, mock_get_client):
        """Test embed() using daemon when enabled."""
        mock_embedding = np.random.rand(EMBEDDING_DIMENSION).astype(np.float32)
        mock_client = MagicMock()
        mock_client.embed.return_value = mock_embedding
        mock_get_client.return_value = mock_client

        # Enable daemon by removing the DISABLE flag set in conftest.py
        disabled_was_set = "STEADYTEXT_DISABLE_DAEMON" in os.environ
        if disabled_was_set:
            del os.environ["STEADYTEXT_DISABLE_DAEMON"]
        try:
            result = embed("test text")
            assert np.array_equal(result, mock_embedding)
            mock_client.embed.assert_called_once()
        finally:
            # Restore original state
            if disabled_was_set:
                os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"


# AIDEV-NOTE: Server tests would require actually starting a server,
# which is more complex for unit tests. In a real implementation,
# we'd use integration tests or test containers for full server testing.
