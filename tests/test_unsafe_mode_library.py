"""Tests for unsafe_mode parameter at the library level.

AIDEV-NOTE: These tests verify that the unsafe_mode parameter works correctly
without requiring environment variable manipulation.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import threading

import steadytext


class TestUnsafeModeParameter:
    """Test unsafe_mode parameter functionality."""

    def test_generate_with_unsafe_mode_true(self, monkeypatch):
        """Test that generate() with unsafe_mode=True enables remote models."""
        # Ensure environment variable is not set
        monkeypatch.delenv("STEADYTEXT_UNSAFE_MODE", raising=False)

        # Mock the remote provider
        mock_provider = Mock()
        mock_provider.generate.return_value = "Remote model response"

        with patch(
            "steadytext.providers.registry.get_provider", return_value=mock_provider
        ):
            # This should work with unsafe_mode=True
            result = steadytext.generate(
                "Test prompt", model="openai:gpt-4o-mini", unsafe_mode=True
            )

            assert result == "Remote model response"
            mock_provider.generate.assert_called_once()

    def test_generate_with_unsafe_mode_false(self, monkeypatch):
        """Test that generate() with unsafe_mode=False (default) blocks remote models."""
        # Ensure environment variable is not set
        monkeypatch.delenv("STEADYTEXT_UNSAFE_MODE", raising=False)

        # This should fail without unsafe_mode
        result = steadytext.generate(
            "Test prompt", model="openai:gpt-4o-mini", unsafe_mode=False
        )

        # Should return None when remote model is blocked
        assert result is None

    def test_generate_iter_with_unsafe_mode_true(self, monkeypatch):
        """Test that generate_iter() with unsafe_mode=True enables remote models."""
        # Ensure environment variable is not set
        monkeypatch.delenv("STEADYTEXT_UNSAFE_MODE", raising=False)

        # Mock the remote provider
        mock_provider = Mock()
        mock_provider.generate_iter.return_value = iter(["Hello", " ", "world"])

        with patch(
            "steadytext.providers.registry.get_provider", return_value=mock_provider
        ):
            # This should work with unsafe_mode=True
            tokens = list(
                steadytext.generate_iter(
                    "Test prompt", model="openai:gpt-4o-mini", unsafe_mode=True
                )
            )

            assert tokens == ["Hello", " ", "world"]
            mock_provider.generate_iter.assert_called_once()

    def test_generate_iter_with_unsafe_mode_false(self, monkeypatch):
        """Test that generate_iter() with unsafe_mode=False (default) blocks remote models."""
        # Ensure environment variable is not set
        monkeypatch.delenv("STEADYTEXT_UNSAFE_MODE", raising=False)

        # This should return empty iterator without unsafe_mode
        tokens = list(
            steadytext.generate_iter(
                "Test prompt", model="openai:gpt-4o-mini", unsafe_mode=False
            )
        )

        # Should return empty when remote model is blocked
        assert tokens == []

    def test_unsafe_mode_does_not_affect_local_models(self):
        """Test that unsafe_mode parameter doesn't affect local model usage."""
        # Mock local model
        with patch("steadytext.core.generator._get_generator_instance") as mock_gen:
            mock_instance = Mock()
            mock_instance.generate.return_value = "Local model response"
            mock_gen.return_value = mock_instance

            # Both should work the same for local models
            result1 = steadytext.generate("Test", unsafe_mode=False)
            result2 = steadytext.generate("Test", unsafe_mode=True)

            assert result1 == "Local model response"
            assert result2 == "Local model response"

    def test_environment_variable_still_works(self, monkeypatch):
        """Test that STEADYTEXT_UNSAFE_MODE environment variable still works."""
        # Set environment variable
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock the remote provider
        mock_provider = Mock()
        mock_provider.generate.return_value = "Remote model response"

        with patch(
            "steadytext.providers.registry.get_provider", return_value=mock_provider
        ):
            # Should work without unsafe_mode parameter when env var is set
            result = steadytext.generate("Test prompt", model="openai:gpt-4o-mini")

            assert result == "Remote model response"

    def test_parameter_overrides_environment(self, monkeypatch):
        """Test that unsafe_mode parameter takes precedence over environment variable."""
        # Set environment variable to false
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "false")

        # Mock the remote provider
        mock_provider = Mock()
        mock_provider.generate.return_value = "Remote model response"

        with patch(
            "steadytext.providers.registry.get_provider", return_value=mock_provider
        ):
            # unsafe_mode=True should override the environment variable
            result = steadytext.generate(
                "Test prompt", model="openai:gpt-4o-mini", unsafe_mode=True
            )

            assert result == "Remote model response"


class TestUnsafeModeThreadSafety:
    """Test thread safety of unsafe_mode parameter."""

    def test_concurrent_unsafe_mode_calls(self, monkeypatch):
        """Test that concurrent calls with different unsafe_mode values don't interfere."""
        # Ensure environment variable is not set
        monkeypatch.delenv("STEADYTEXT_UNSAFE_MODE", raising=False)

        results = {}
        lock = threading.Lock()

        def run_with_unsafe_mode(thread_id, unsafe_mode):
            """Run generation with specific unsafe_mode setting."""
            # Check that environment is properly isolated
            initial_env = os.environ.get("STEADYTEXT_UNSAFE_MODE")

            # Mock the provider for unsafe_mode=True
            if unsafe_mode:
                with patch(
                    "steadytext.providers.registry.get_provider"
                ) as mock_get_provider:
                    mock_provider = Mock()
                    mock_provider.generate.return_value = f"Thread {thread_id} unsafe"
                    mock_get_provider.return_value = mock_provider

                    result = steadytext.generate(
                        f"Test prompt {thread_id}",
                        model="openai:gpt-4o-mini",
                        unsafe_mode=unsafe_mode,
                    )
            else:
                # For unsafe_mode=False, remote model should fail
                result = steadytext.generate(
                    f"Test prompt {thread_id}",
                    model="openai:gpt-4o-mini",
                    unsafe_mode=unsafe_mode,
                )

            # Verify environment wasn't changed by other threads
            final_env = os.environ.get("STEADYTEXT_UNSAFE_MODE")
            assert initial_env == final_env, (
                f"Thread {thread_id}: Environment changed from {initial_env} to {final_env}"
            )

            with lock:
                results[thread_id] = result

        # Create threads with different unsafe_mode values
        threads = []
        for i in range(10):
            unsafe = i % 2 == 0  # Even threads use unsafe_mode=True
            t = threading.Thread(target=run_with_unsafe_mode, args=(i, unsafe))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Even threads should have succeeded
        for i in range(0, 10, 2):
            assert results.get(i) == f"Thread {i} unsafe"

        # Odd threads should have returned None
        for i in range(1, 10, 2):
            assert results.get(i) is None

    def test_environment_not_permanently_modified(self, monkeypatch):
        """Test that using unsafe_mode doesn't permanently modify environment."""
        # Start with environment variable not set
        monkeypatch.delenv("STEADYTEXT_UNSAFE_MODE", raising=False)
        original_env = os.environ.get("STEADYTEXT_UNSAFE_MODE")

        # Mock the remote provider
        mock_provider = Mock()
        mock_provider.generate.return_value = "Remote response"

        with patch(
            "steadytext.providers.registry.get_provider", return_value=mock_provider
        ):
            # Use unsafe_mode=True
            steadytext.generate(
                "Test prompt", model="openai:gpt-4o-mini", unsafe_mode=True
            )

        # Environment should be unchanged
        assert os.environ.get("STEADYTEXT_UNSAFE_MODE") == original_env


class TestDaemonIntegration:
    """Test unsafe_mode with daemon client/server."""

    @patch("steadytext.daemon.client.zmq")
    def test_daemon_client_passes_unsafe_mode(self, mock_zmq):
        """Test that daemon client correctly passes unsafe_mode parameter."""
        # Setup mock socket
        mock_socket = MagicMock()
        mock_context = MagicMock()
        mock_context.socket.return_value = mock_socket
        mock_zmq.Context.return_value = mock_context

        # Mock successful ping
        mock_socket.recv.side_effect = [
            b'{"id": "test", "result": "pong", "error": null}',  # ping response
            b'{"id": "test", "result": "Generated text", "error": null}',  # generate response
        ]

        from steadytext.daemon.client import DaemonClient

        client = DaemonClient()

        # Test generate with unsafe_mode
        result = client.generate(
            prompt="Test", model="openai:gpt-4o-mini", unsafe_mode=True
        )

        # Check that request included unsafe_mode
        calls = mock_socket.send.call_args_list
        generate_call = calls[1]  # Second call (after ping)
        request_data = generate_call[0][0].decode()

        assert '"unsafe_mode": true' in request_data
        assert result == "Generated text"

    def test_daemon_client_generate_iter_unsafe_mode(self):
        """Test that daemon client passes unsafe_mode in generate_iter."""
        # This test verifies the parameter is passed but doesn't test the full daemon functionality
        # since that would require a running daemon
        from steadytext.daemon.client import DaemonClient

        # Test that the client would pass unsafe_mode in the request
        client = DaemonClient()

        # Mock the connection to avoid actual daemon requirement
        with patch.object(client, "connect", return_value=False):
            # Should raise ConnectionError when daemon not available
            with pytest.raises(ConnectionError, match="Daemon not available"):
                list(
                    client.generate_iter(
                        prompt="Test", model="openai:gpt-4o-mini", unsafe_mode=True
                    )
                )

        # Verify the parameter is part of the client API
        import inspect

        sig = inspect.signature(client.generate_iter)
        assert "unsafe_mode" in sig.parameters
        assert sig.parameters["unsafe_mode"].default is False
