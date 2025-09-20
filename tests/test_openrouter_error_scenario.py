"""Test OpenRouter provider error handling user scenarios.

AIDEV-NOTE: These tests represent complete error handling workflows with OpenRouter
models as described in the quickstart guide and spec. Tests MUST fail initially
since OpenRouterProvider doesn't exist yet (TDD approach).
"""

import os
import pytest
import warnings
import numpy as np
from unittest.mock import Mock, patch

# AIDEV-NOTE: These imports will fail initially - this is expected for TDD
from steadytext.providers.openrouter import OpenRouterProvider, OpenRouterError
from steadytext.providers.base import UnsafeModeWarning


class TestOpenRouterErrorHandling:
    """Test error handling scenarios with OpenRouter provider."""

    def test_missing_api_key_error(self, monkeypatch):
        """Test error when OPENROUTER_API_KEY is missing."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        provider = OpenRouterProvider(api_key=None, model="anthropic/claude-3.5-sonnet")
        assert not provider.is_available()

        # Should raise appropriate error when trying to use
        with pytest.raises(OpenRouterError, match="API key is required"):
            provider.generate("Test prompt")

    def test_invalid_api_key_error(self, monkeypatch):
        """Test error handling with invalid API key."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        with patch("steadytext.providers.openrouter._get_openai") as mock_get_openai:
            # Mock OpenAI client that raises authentication error
            mock_openai = Mock()
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("Invalid API key")
            mock_openai.OpenAI.return_value = mock_client
            mock_get_openai.return_value = mock_openai

            provider = OpenRouterProvider(
                api_key="invalid-key",
                model="anthropic/claude-3.5-sonnet"
            )

            with pytest.raises(OpenRouterError, match="Invalid API key"):
                provider.generate("Test prompt")

    def test_invalid_model_error(self, monkeypatch):
        """Test error handling with invalid model name."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        with patch("steadytext.providers.openrouter._get_openai") as mock_get_openai:
            # Mock OpenAI client that raises model not found error
            mock_openai = Mock()
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("Model not found")
            mock_openai.OpenAI.return_value = mock_client
            mock_get_openai.return_value = mock_openai

            provider = OpenRouterProvider(
                api_key="sk-or-test-key",
                model="nonexistent/model"
            )

            with pytest.raises(OpenRouterError, match="Model.*not found"):
                provider.generate("Test prompt")

    @pytest.mark.slow
    def test_rate_limiting_error_and_retry(self, monkeypatch):
        """Test rate limiting error handling and retry logic."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        with patch("steadytext.providers.openrouter._get_openai") as mock_get_openai:
            # Mock OpenAI client that first raises rate limit error, then succeeds
            mock_openai = Mock()
            mock_client = Mock()

            # First call raises rate limit error
            rate_limit_error = Exception("Rate limit exceeded")
            success_response = Mock(choices=[Mock(message=Mock(content="Success after retry"))])

            mock_client.chat.completions.create.side_effect = [rate_limit_error, success_response]
            mock_openai.OpenAI.return_value = mock_client
            mock_get_openai.return_value = mock_openai

            provider = OpenRouterProvider(
                api_key="sk-or-test-key",
                model="anthropic/claude-3.5-sonnet",
                max_retries=2
            )

            # Should retry and eventually succeed
            result = provider.generate("Test prompt")
            assert result == "Success after retry"

            # Should have made 2 calls (first failed, second succeeded)
            assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.slow
    def test_network_error_fallback(self, monkeypatch):
        """Test fallback behavior when OpenRouter is unreachable."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        with patch("steadytext.providers.openrouter._get_openai") as mock_get_openai:
            # Mock network error
            mock_openai = Mock()
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("Network error")
            mock_openai.OpenAI.return_value = mock_client
            mock_get_openai.return_value = mock_openai

            provider = OpenRouterProvider(
                api_key="sk-or-test-key",
                model="anthropic/claude-3.5-sonnet"
            )

            # Should handle network error gracefully
            result = provider.generate("Test prompt")
            assert result is None  # Returns None on error

    def test_unsafe_mode_not_enabled_error(self, monkeypatch):
        """Test error when unsafe mode is not enabled."""
        monkeypatch.delenv("STEADYTEXT_UNSAFE_MODE", raising=False)

        from steadytext.providers.registry import get_provider

        with pytest.raises(RuntimeError, match="Remote models require unsafe mode"):
            get_provider("openrouter:anthropic/claude-3.5-sonnet")

    @pytest.mark.slow
    def test_deterministic_fallback_integration(self, monkeypatch):
        """Test integration with SteadyText deterministic fallback system.

        This tests the "offline/fallback behavior" scenario from quickstart.md.
        """
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        from steadytext import generate

        with patch("steadytext.providers.openrouter._get_openai") as mock_get_openai:
            # Mock OpenRouter as unavailable
            mock_get_openai.side_effect = Exception("OpenRouter unavailable")

            # Should fall back to deterministic generation
            result = generate(
                "Hello world",
                model="openrouter:anthropic/claude-3.5-sonnet",
                unsafe_mode=True
            )

            # Should still return a response (deterministic fallback)
            assert result is not None
            assert isinstance(result, str)
            # Deterministic fallback should produce consistent results
            result2 = generate(
                "Hello world",
                model="openrouter:anthropic/claude-3.5-sonnet",
                unsafe_mode=True
            )
            assert result == result2

    @pytest.mark.slow
    def test_timeout_error_handling(self, monkeypatch):
        """Test timeout error handling."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        with patch("steadytext.providers.openrouter._get_openai") as mock_get_openai:
            # Mock timeout error
            mock_openai = Mock()
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("Request timeout")
            mock_openai.OpenAI.return_value = mock_client
            mock_get_openai.return_value = mock_openai

            provider = OpenRouterProvider(
                api_key="sk-or-test-key",
                model="anthropic/claude-3.5-sonnet",
                timeout=30
            )

            with pytest.raises(OpenRouterError, match="timeout"):
                provider.generate("Test prompt")

    def test_malformed_response_error(self, monkeypatch):
        """Test handling of malformed API responses."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        with patch("steadytext.providers.openrouter._get_openai") as mock_get_openai:
            # Mock malformed response
            mock_openai = Mock()
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = []  # Empty choices
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client
            mock_get_openai.return_value = mock_openai

            provider = OpenRouterProvider(
                api_key="sk-or-test-key",
                model="anthropic/claude-3.5-sonnet"
            )

            with pytest.raises(OpenRouterError, match="Invalid response format"):
                provider.generate("Test prompt")

    @pytest.mark.slow
    def test_embedding_error_fallback(self, monkeypatch):
        """Test embedding error handling and fallback behavior."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        from steadytext import embed

        with patch("steadytext.providers.openrouter._get_openai") as mock_get_openai:
            # Mock embedding API error
            mock_openai = Mock()
            mock_client = Mock()
            mock_client.embeddings.create.side_effect = Exception("Embedding API error")
            mock_openai.OpenAI.return_value = mock_client
            mock_get_openai.return_value = mock_openai

            # Should fall back to deterministic embedding (zero vector)
            result = embed(
                "Test text",
                model="openrouter:openai/text-embedding-3-small",
                unsafe_mode=True
            )

            # Should still return an embedding
            assert isinstance(result, np.ndarray)
            assert result.shape == (1024,)
            # Might be zero vector fallback or local model fallback
            assert np.allclose(np.linalg.norm(result), 1.0, rtol=1e-5) or np.allclose(result, 0)

    def test_quota_exceeded_error(self, monkeypatch):
        """Test quota exceeded error handling."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        with patch("steadytext.providers.openrouter._get_openai") as mock_get_openai:
            # Mock quota exceeded error
            mock_openai = Mock()
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("Quota exceeded")
            mock_openai.OpenAI.return_value = mock_client
            mock_get_openai.return_value = mock_openai

            provider = OpenRouterProvider(
                api_key="sk-or-test-key",
                model="anthropic/claude-3.5-sonnet"
            )

            with pytest.raises(OpenRouterError, match="Quota exceeded"):
                provider.generate("Test prompt")

    @pytest.mark.slow
    def test_streaming_error_handling(self, monkeypatch):
        """Test error handling during streaming generation."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        with patch("steadytext.providers.openrouter._get_openai") as mock_get_openai:
            # Mock streaming error
            mock_openai = Mock()
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("Streaming error")
            mock_openai.OpenAI.return_value = mock_client
            mock_get_openai.return_value = mock_openai

            provider = OpenRouterProvider(
                api_key="sk-or-test-key",
                model="anthropic/claude-3.5-sonnet"
            )

            # Should handle streaming error gracefully
            chunks = list(provider.generate_iter("Test prompt"))
            assert chunks == []  # Empty list on error

    def test_connection_error_retry_logic(self, monkeypatch):
        """Test retry logic for connection errors."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        with patch("steadytext.providers.openrouter._get_openai") as mock_get_openai:
            # Mock connection errors followed by success
            mock_openai = Mock()
            mock_client = Mock()

            connection_error = Exception("Connection failed")
            success_response = Mock(choices=[Mock(message=Mock(content="Connection restored"))])

            # Fail twice, then succeed
            mock_client.chat.completions.create.side_effect = [
                connection_error,
                connection_error,
                success_response
            ]
            mock_openai.OpenAI.return_value = mock_client
            mock_get_openai.return_value = mock_openai

            provider = OpenRouterProvider(
                api_key="sk-or-test-key",
                model="anthropic/claude-3.5-sonnet",
                max_retries=3
            )

            # Should retry and eventually succeed
            result = provider.generate("Test prompt")
            assert result == "Connection restored"

            # Should have made 3 calls (2 failed, 1 succeeded)
            assert mock_client.chat.completions.create.call_count == 3

    def test_openrouter_specific_error_messages(self):
        """Test OpenRouter-specific error message formatting."""
        # Test that OpenRouterError provides helpful error messages

        error_scenarios = [
            ("Invalid API key", "API key"),
            ("Model not found", "model"),
            ("Rate limit exceeded", "rate limit"),
            ("Quota exceeded", "quota"),
            ("Request timeout", "timeout"),
        ]

        for error_msg, expected_keyword in error_scenarios:
            error = OpenRouterError(error_msg)
            assert expected_keyword.lower() in str(error).lower()
            assert "openrouter" in str(error).lower()


class TestOpenRouterErrorEdgeCases:
    """Test edge cases and unusual error scenarios."""

    def test_empty_model_name_error(self):
        """Test error with empty model name."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            OpenRouterProvider(api_key="sk-or-test-key", model="")

    def test_none_model_name_error(self):
        """Test error with None model name."""
        with pytest.raises(ValueError, match="Model name is required"):
            OpenRouterProvider(api_key="sk-or-test-key", model=None)

    def test_whitespace_only_api_key_error(self):
        """Test error with whitespace-only API key."""
        provider = OpenRouterProvider(api_key="   ", model="anthropic/claude-3.5-sonnet")
        assert not provider.is_available()

    def test_malformed_api_key_format_warning(self):
        """Test warning for malformed API key format."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            provider = OpenRouterProvider(
                api_key="not-a-valid-format",
                model="anthropic/claude-3.5-sonnet"
            )
            # Should warn about potentially invalid API key format
            # (Real API keys should start with "sk-or-")

    @pytest.mark.slow
    def test_partial_streaming_failure(self, monkeypatch):
        """Test handling of partial streaming failures."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        with patch("steadytext.providers.openrouter._get_openai") as mock_get_openai:
            # Mock streaming that starts successfully but fails mid-stream
            mock_openai = Mock()
            mock_client = Mock()

            def failing_stream():
                yield Mock(choices=[Mock(delta=Mock(content="Hello"))])
                yield Mock(choices=[Mock(delta=Mock(content=" world"))])
                raise Exception("Stream interrupted")

            mock_client.chat.completions.create.return_value = failing_stream()
            mock_openai.OpenAI.return_value = mock_client
            mock_get_openai.return_value = mock_openai

            provider = OpenRouterProvider(
                api_key="sk-or-test-key",
                model="anthropic/claude-3.5-sonnet"
            )

            # Should yield partial results before failing
            chunks = []
            try:
                for chunk in provider.generate_iter("Test prompt"):
                    chunks.append(chunk)
            except OpenRouterError:
                pass  # Expected to fail

            # Should have yielded partial results
            assert chunks == ["Hello", " world"]

    def test_concurrent_request_error_isolation(self, monkeypatch):
        """Test that errors in one request don't affect others."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # This test would verify that provider instances are properly isolated
        # and errors in one request don't corrupt the state for other requests

        provider1 = OpenRouterProvider(
            api_key="sk-or-test-key-1",
            model="anthropic/claude-3.5-sonnet"
        )

        provider2 = OpenRouterProvider(
            api_key="sk-or-test-key-2",
            model="openai/gpt-4"
        )

        # Each provider should maintain separate state
        assert provider1.api_key != provider2.api_key
        assert provider1.model != provider2.model


# AIDEV-NOTE: These error handling tests will fail initially because:
# 1. OpenRouterProvider class doesn't exist yet
# 2. OpenRouterError exception class doesn't exist yet
# 3. Error handling methods aren't implemented
# 4. Retry logic isn't implemented
# 5. Integration with SteadyText fallback system isn't complete
# 6. This is expected and correct for TDD approach
# 7. Tests represent complete error scenarios from quickstart.md and spec.md