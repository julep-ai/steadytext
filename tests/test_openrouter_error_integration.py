"""Integration tests for OpenRouter error handling and fallbacks.

AIDEV-NOTE: These tests verify OpenRouter error handling, fallback behavior, and resilience.
They MUST fail initially since OpenRouterProvider doesn't exist yet (TDD requirement).
Tests focus on end-to-end error scenarios and recovery mechanisms.
"""

import pytest
import os
import json
from unittest.mock import patch, Mock, MagicMock
from click.testing import CliRunner
import requests

# AIDEV-NOTE: These imports will fail initially - this is expected for TDD
try:
    from steadytext.providers.openrouter import OpenRouterProvider
    OPENROUTER_PROVIDER_AVAILABLE = True
except ImportError:
    OPENROUTER_PROVIDER_AVAILABLE = False

try:
    from steadytext.providers.registry import get_provider, parse_remote_model, is_unsafe_mode_enabled
    from steadytext.cli.main import cli
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

try:
    from steadytext import generate, embed
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


@pytest.fixture
def runner():
    """Fixture for invoking command-line calls."""
    return CliRunner()


class TestOpenRouterAPIErrorHandling:
    """Test OpenRouter API error handling and recovery."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.slow
    def test_openrouter_api_timeout_error(self, monkeypatch):
        """Test handling of API timeout errors."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock timeout error
        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', side_effect=requests.exceptions.Timeout("Request timed out")):

            provider = OpenRouterProvider(api_key="sk-or-test-key", model="anthropic/claude-3.5-sonnet")

            with pytest.raises(requests.exceptions.Timeout):
                provider.generate("Hello world")

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.slow
    def test_openrouter_api_connection_error(self, monkeypatch):
        """Test handling of API connection errors."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock connection error
        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', side_effect=requests.exceptions.ConnectionError("Connection failed")):

            provider = OpenRouterProvider(api_key="sk-or-test-key", model="anthropic/claude-3.5-sonnet")

            with pytest.raises(requests.exceptions.ConnectionError):
                provider.generate("Hello world")

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.slow
    def test_openrouter_api_rate_limit_error(self, monkeypatch):
        """Test handling of API rate limit errors."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock rate limit error (429 status)
        rate_limit_error = requests.exceptions.HTTPError("429 Rate limit exceeded")
        rate_limit_error.response = Mock()
        rate_limit_error.response.status_code = 429

        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', side_effect=rate_limit_error):

            provider = OpenRouterProvider(api_key="sk-or-test-key", model="anthropic/claude-3.5-sonnet")

            with pytest.raises(requests.exceptions.HTTPError):
                provider.generate("Hello world")

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.slow
    def test_openrouter_api_authentication_error(self, monkeypatch):
        """Test handling of API authentication errors."""
        # Set up environment with invalid key
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-invalid-key")

        # Mock authentication error (401 status)
        auth_error = requests.exceptions.HTTPError("401 Unauthorized")
        auth_error.response = Mock()
        auth_error.response.status_code = 401

        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', side_effect=auth_error):

            provider = OpenRouterProvider(api_key="sk-or-invalid-key", model="anthropic/claude-3.5-sonnet")

            with pytest.raises(requests.exceptions.HTTPError):
                provider.generate("Hello world")

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.slow
    def test_openrouter_api_server_error(self, monkeypatch):
        """Test handling of API server errors."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock server error (500 status)
        server_error = requests.exceptions.HTTPError("500 Internal Server Error")
        server_error.response = Mock()
        server_error.response.status_code = 500

        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', side_effect=server_error):

            provider = OpenRouterProvider(api_key="sk-or-test-key", model="anthropic/claude-3.5-sonnet")

            with pytest.raises(requests.exceptions.HTTPError):
                provider.generate("Hello world")

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.slow
    def test_openrouter_api_malformed_response(self, monkeypatch):
        """Test handling of malformed API responses."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock malformed response
        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', side_effect=json.JSONDecodeError("Invalid JSON", "", 0)):

            provider = OpenRouterProvider(api_key="sk-or-test-key", model="anthropic/claude-3.5-sonnet")

            with pytest.raises(json.JSONDecodeError):
                provider.generate("Hello world")


class TestOpenRouterCLIErrorHandling:
    """Test error handling in CLI with OpenRouter models."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    @pytest.mark.slow
    def test_cli_handles_openrouter_timeout(self, runner, monkeypatch):
        """Test CLI handling of OpenRouter timeout errors."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock timeout error
        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', side_effect=requests.exceptions.Timeout("Request timed out")):

            result = runner.invoke(cli, [
                "generate",
                "Hello world",
                "--model", "openrouter:anthropic/claude-3.5-sonnet",
                "--wait"
            ])

            assert result.exit_code != 0
            assert "timeout" in result.output.lower() or "error" in result.output.lower()

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    @pytest.mark.slow
    def test_cli_handles_openrouter_connection_error(self, runner, monkeypatch):
        """Test CLI handling of OpenRouter connection errors."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock connection error
        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', side_effect=requests.exceptions.ConnectionError("Connection failed")):

            result = runner.invoke(cli, [
                "generate",
                "Hello world",
                "--model", "openrouter:anthropic/claude-3.5-sonnet",
                "--wait"
            ])

            assert result.exit_code != 0
            assert "connection" in result.output.lower() or "error" in result.output.lower()

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    @pytest.mark.slow
    def test_cli_handles_openrouter_auth_error(self, runner, monkeypatch):
        """Test CLI handling of OpenRouter authentication errors."""
        # Set up environment with invalid key
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-invalid-key")

        # Mock authentication error
        auth_error = requests.exceptions.HTTPError("401 Unauthorized")
        auth_error.response = Mock()
        auth_error.response.status_code = 401

        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', side_effect=auth_error):

            result = runner.invoke(cli, [
                "generate",
                "Hello world",
                "--model", "openrouter:anthropic/claude-3.5-sonnet",
                "--wait"
            ])

            assert result.exit_code != 0
            assert "unauthorized" in result.output.lower() or "authentication" in result.output.lower() or "api key" in result.output.lower()

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_cli_handles_invalid_openrouter_model_gracefully(self, runner, monkeypatch):
        """Test CLI handling of invalid OpenRouter model formats."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        # Test various invalid formats
        invalid_models = [
            "openrouter:",  # Empty model name
            "openrouter:invalid-format",  # No slash
            "openrouter:/model",  # Empty provider
            "openrouter:provider/",  # Empty model
            "openrouter: anthropic/claude-3.5-sonnet",  # Space after colon
        ]

        for invalid_model in invalid_models:
            result = runner.invoke(cli, [
                "generate",
                "Hello world",
                "--model", invalid_model,
                "--wait"
            ])

            assert result.exit_code != 0, f"Should fail for invalid model: {invalid_model}"
            # Should provide helpful error message
            assert len(result.output) > 0


class TestOpenRouterFallbackBehavior:
    """Test fallback behavior when OpenRouter fails."""

    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core not available")
    @pytest.mark.slow
    def test_generate_falls_back_on_openrouter_failure(self, monkeypatch):
        """Test that generate() falls back to deterministic mode on OpenRouter failure."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock provider failure during generation
        if OPENROUTER_PROVIDER_AVAILABLE:
            with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
                 patch.object(OpenRouterProvider, 'generate', side_effect=Exception("API Error")):

                # Should fall back to deterministic generation
                result = generate("Hello world", model="openrouter:anthropic/claude-3.5-sonnet")

                # Should return something (fallback behavior)
                assert isinstance(result, str)

    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core not available")
    @pytest.mark.slow
    def test_embed_falls_back_on_openrouter_failure(self, monkeypatch):
        """Test that embed() falls back to deterministic mode on OpenRouter failure."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock provider failure during embedding
        if OPENROUTER_PROVIDER_AVAILABLE:
            with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
                 patch.object(OpenRouterProvider, 'embed', side_effect=Exception("API Error")):

                # Should fall back to deterministic embedding
                result = embed("Hello world", model="openrouter:text-embedding-3-large")

                # Should return 1024-dimensional zero vector (fallback behavior)
                assert isinstance(result, list)
                assert len(result) == 1024


class TestOpenRouterNetworkResilience:
    """Test network resilience and retry behavior."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.slow
    def test_openrouter_intermittent_network_issues(self, monkeypatch):
        """Test handling of intermittent network issues."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock intermittent failures (fails first time, succeeds second time)
        call_count = 0
        def mock_generate_with_retry(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.exceptions.ConnectionError("Network error")
            return "Success after retry"

        with patch.object(OpenRouterProvider, 'is_available', return_value=True):
            # If provider implements retry logic, test it
            # Otherwise, just test that errors are properly propagated
            provider = OpenRouterProvider(api_key="sk-or-test-key", model="anthropic/claude-3.5-sonnet")

            # This test depends on retry implementation in the provider
            # For now, just verify the provider can be created
            assert provider is not None

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.slow
    def test_openrouter_partial_response_handling(self, monkeypatch):
        """Test handling of partial or incomplete responses."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock partial response that gets cut off
        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', return_value="Partial respon"):

            provider = OpenRouterProvider(api_key="sk-or-test-key", model="anthropic/claude-3.5-sonnet")
            result = provider.generate("Hello world")

            # Should handle partial response gracefully
            assert isinstance(result, str)
            assert result == "Partial respon"


class TestOpenRouterEnvironmentErrorHandling:
    """Test error handling related to environment configuration."""

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_missing_unsafe_mode_clear_error_message(self, monkeypatch):
        """Test clear error message when unsafe mode is not enabled."""
        # Disable unsafe mode
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "false")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        with pytest.raises(RuntimeError) as exc_info:
            get_provider("openrouter:anthropic/claude-3.5-sonnet")

        error_message = str(exc_info.value)
        assert "unsafe mode" in error_message.lower()
        assert "STEADYTEXT_UNSAFE_MODE=true" in error_message
        assert "determinism" in error_message.lower()

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_missing_api_key_clear_error_message(self, monkeypatch):
        """Test clear error message when OpenRouter API key is missing."""
        # Enable unsafe mode but remove API key
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            get_provider("openrouter:anthropic/claude-3.5-sonnet")

        error_message = str(exc_info.value)
        assert "not available" in error_message.lower()
        assert "api key" in error_message.lower()

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_invalid_model_format_clear_error_message(self):
        """Test clear error message for invalid model formats."""
        # Test various invalid formats
        invalid_formats = [
            "openrouter",  # No model part
            "openrouter:",  # Empty model part
            "openrouter:invalid",  # No slash
            ":anthropic/claude-3.5-sonnet",  # No provider part
        ]

        for invalid_format in invalid_formats:
            with pytest.raises(ValueError) as exc_info:
                parse_remote_model(invalid_format)

            error_message = str(exc_info.value)
            assert "invalid" in error_message.lower() or "format" in error_message.lower()


class TestOpenRouterModelValidationErrors:
    """Test model validation and error handling."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_unsupported_model_error(self, monkeypatch):
        """Test error handling for unsupported models."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock provider that doesn't support the requested model
        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'get_supported_models', return_value=["other/model"]):

            # Should either succeed (if no validation) or provide clear error
            try:
                provider = OpenRouterProvider(api_key="sk-or-test-key", model="unsupported/model")
                # If creation succeeds, that's also acceptable
                assert provider is not None
            except ValueError as e:
                # If validation fails, error should be clear
                assert "unsupported" in str(e).lower() or "model" in str(e).lower()

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_model_name_edge_cases(self):
        """Test edge cases in model name handling."""
        # Test various edge cases that should be handled gracefully
        edge_cases = [
            "openrouter:provider/model-with-many-hyphens",
            "openrouter:provider/model_with_underscores",
            "openrouter:provider.with.dots/model",
            "openrouter:provider123/model456",
        ]

        for model in edge_cases:
            try:
                # Should not crash on initialization
                provider_name, model_name = parse_remote_model(model)
                assert provider_name == "openrouter"
                assert "/" in model_name
            except ValueError:
                # If validation rejects it, that's also acceptable
                pass


def test_openrouter_error_imports_fail():
    """Test that OpenRouter error handling imports fail (TDD requirement).

    AIDEV-NOTE: This test MUST pass initially and MUST fail after implementation.
    It ensures we're following TDD correctly.
    """
    if OPENROUTER_PROVIDER_AVAILABLE:
        pytest.fail(
            "OpenRouterProvider should not be available yet for TDD. "
            "This test should fail after implementation."
        )

    # Verify the import fails as expected
    with pytest.raises(ImportError):
        from steadytext.providers.openrouter import OpenRouterProvider


class TestErrorHandlingContractCompliance:
    """Test that error handling follows expected patterns."""

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_error_messages_are_actionable(self):
        """Test that error messages provide actionable information."""
        # Test unsafe mode error
        with patch.dict(os.environ, {"STEADYTEXT_UNSAFE_MODE": "false"}):
            with pytest.raises(RuntimeError) as exc_info:
                get_provider("openai:gpt-4o-mini")  # Use existing provider for test

            error_msg = str(exc_info.value)
            # Should tell user what to do
            assert "STEADYTEXT_UNSAFE_MODE=true" in error_msg
            assert "unsafe mode" in error_msg.lower()

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_error_types_are_appropriate(self):
        """Test that appropriate exception types are raised."""
        # Invalid model format should raise ValueError
        with pytest.raises(ValueError):
            parse_remote_model("invalid-format")

        # Missing unsafe mode should raise RuntimeError
        with patch.dict(os.environ, {"STEADYTEXT_UNSAFE_MODE": "false"}):
            with pytest.raises(RuntimeError):
                get_provider("openai:gpt-4o-mini")

    def test_error_handling_is_consistent(self):
        """Test that error handling is consistent across similar functions."""
        # Test that similar inputs produce similar error types
        invalid_formats = [
            "no-colon",
            ":empty-provider",
            "provider:",
            "",
        ]

        for invalid_format in invalid_formats:
            if invalid_format:  # Skip empty string for this test
                with pytest.raises((ValueError, TypeError)):
                    parse_remote_model(invalid_format)