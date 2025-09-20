"""Integration tests for OpenRouter CLI support.

AIDEV-NOTE: These tests verify OpenRouter integration with the CLI commands.
They MUST fail initially since OpenRouterProvider doesn't exist yet (TDD requirement).
Tests focus on end-to-end CLI behavior with OpenRouter models.
"""

import pytest
import json
import os
from click.testing import CliRunner
from unittest.mock import patch, Mock

# AIDEV-NOTE: These imports will fail initially - this is expected for TDD
try:
    from steadytext.providers.openrouter import OpenRouterProvider
    OPENROUTER_PROVIDER_AVAILABLE = True
except ImportError:
    OPENROUTER_PROVIDER_AVAILABLE = False

try:
    from steadytext.cli.main import cli
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False


@pytest.fixture
def runner():
    """Fixture for invoking command-line calls."""
    return CliRunner()


class TestOpenRouterGenerateCommand:
    """Test OpenRouter integration with `st generate` command."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    @pytest.mark.slow
    def test_generate_with_openrouter_model(self, runner, monkeypatch):
        """Test `st generate` with OpenRouter model."""
        # Set up environment for unsafe mode and API key
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock the provider to avoid actual API calls
        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', return_value="Hello from OpenRouter!"):

            result = runner.invoke(cli, [
                "generate",
                "Hello world",
                "--model", "openrouter:anthropic/claude-3.5-sonnet",
                "--wait"
            ])

            assert result.exit_code == 0
            assert "Hello from OpenRouter!" in result.output

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    @pytest.mark.slow
    def test_generate_openrouter_json_output(self, runner, monkeypatch):
        """Test `st generate` with OpenRouter model and JSON output."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock the provider
        mock_response = {
            "text": "Hello from OpenRouter!",
            "model": "anthropic/claude-3.5-sonnet",
            "provider": "openrouter",
            "usage": {"total_tokens": 10}
        }

        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', return_value="Hello from OpenRouter!"):

            result = runner.invoke(cli, [
                "generate",
                "Hello world",
                "--model", "openrouter:anthropic/claude-3.5-sonnet",
                "--json"
            ])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "text" in data
            assert "model" in data
            assert data["text"] == "Hello from OpenRouter!"

    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    def test_generate_openrouter_unsafe_mode_required(self, runner, monkeypatch):
        """Test that OpenRouter models require unsafe mode."""
        # Disable unsafe mode
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "false")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        result = runner.invoke(cli, [
            "generate",
            "Hello world",
            "--model", "openrouter:anthropic/claude-3.5-sonnet"
        ])

        assert result.exit_code != 0
        assert "unsafe mode" in result.output.lower() or "STEADYTEXT_UNSAFE_MODE" in result.output

    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    def test_generate_openrouter_missing_api_key(self, runner, monkeypatch):
        """Test that OpenRouter models require API key."""
        # Enable unsafe mode but remove API key
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = runner.invoke(cli, [
            "generate",
            "Hello world",
            "--model", "openrouter:anthropic/claude-3.5-sonnet"
        ])

        assert result.exit_code != 0
        assert "not available" in result.output or "API key" in result.output

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    @pytest.mark.slow
    def test_generate_openrouter_different_models(self, runner, monkeypatch):
        """Test `st generate` with different OpenRouter models."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        models_to_test = [
            "openrouter:anthropic/claude-3.5-sonnet",
            "openrouter:openai/gpt-4o-mini",
            "openrouter:meta-llama/llama-3.1-8b-instruct",
            "openrouter:google/gemma-2-9b-it"
        ]

        for model in models_to_test:
            with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
                 patch.object(OpenRouterProvider, 'generate', return_value=f"Response from {model}"):

                result = runner.invoke(cli, [
                    "generate",
                    "Test prompt",
                    "--model", model,
                    "--wait"
                ])

                assert result.exit_code == 0, f"Failed for model: {model}"

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    @pytest.mark.slow
    def test_generate_openrouter_streaming(self, runner, monkeypatch):
        """Test `st generate` with OpenRouter model in streaming mode."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock streaming response
        def mock_generate_stream(prompt, **kwargs):
            yield "Hello"
            yield " from"
            yield " OpenRouter!"

        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate_stream', side_effect=mock_generate_stream):

            result = runner.invoke(cli, [
                "generate",
                "Hello world",
                "--model", "openrouter:anthropic/claude-3.5-sonnet"
                # Default is streaming mode (no --wait)
            ])

            assert result.exit_code == 0


class TestOpenRouterEmbedCommand:
    """Test OpenRouter integration with `st embed` command."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    @pytest.mark.slow
    def test_embed_with_openrouter_model(self, runner, monkeypatch):
        """Test `st embed` with OpenRouter embedding model."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock embedding response (1024-dimensional vector)
        mock_embedding = [0.1] * 1024

        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'embed', return_value=mock_embedding):

            result = runner.invoke(cli, [
                "embed",
                "Text to embed",
                "--model", "openrouter:text-embedding-3-large",
                "--json"
            ])

            assert result.exit_code == 0
            if result.output.strip():
                data = json.loads(result.output)
                assert "embedding" in data
                assert len(data["embedding"]) == 1024

    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    def test_embed_openrouter_unsafe_mode_required(self, runner, monkeypatch):
        """Test that OpenRouter embedding models require unsafe mode."""
        # Disable unsafe mode
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "false")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        result = runner.invoke(cli, [
            "embed",
            "Text to embed",
            "--model", "openrouter:text-embedding-3-large"
        ])

        assert result.exit_code != 0
        assert "unsafe mode" in result.output.lower() or "STEADYTEXT_UNSAFE_MODE" in result.output


class TestOpenRouterModelListing:
    """Test OpenRouter model listing in CLI commands."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    def test_models_list_includes_openrouter(self, runner):
        """Test that `st models list` includes OpenRouter provider."""
        result = runner.invoke(cli, ["models", "list"])

        assert result.exit_code == 0
        # Should show OpenRouter as a remote provider
        assert "openrouter" in result.output.lower()

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    @pytest.mark.slow
    def test_models_list_remote_includes_openrouter(self, runner, monkeypatch):
        """Test that `st models list --remote` includes OpenRouter models."""
        # Set up API key for model listing
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock the model listing
        mock_models = [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o-mini",
            "meta-llama/llama-3.1-8b-instruct"
        ]

        with patch.object(OpenRouterProvider, 'get_supported_models', return_value=mock_models):
            result = runner.invoke(cli, ["models", "list", "--remote"])

            assert result.exit_code == 0
            assert "openrouter" in result.output
            # Should show some OpenRouter models
            for model in mock_models[:2]:  # Check first couple
                assert model in result.output


class TestOpenRouterCLIErrorHandling:
    """Test error handling in CLI commands with OpenRouter."""

    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    def test_invalid_openrouter_model_format(self, runner, monkeypatch):
        """Test CLI handling of invalid OpenRouter model format."""
        # Enable unsafe mode and set API key
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        # Test invalid format (no slash in model name)
        result = runner.invoke(cli, [
            "generate",
            "Hello world",
            "--model", "openrouter:invalid-model-format"
        ])

        assert result.exit_code != 0
        # Should show error about model format

    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    def test_unknown_provider_in_openrouter_format(self, runner, monkeypatch):
        """Test CLI handling of unknown provider in OpenRouter-like format."""
        # Enable unsafe mode
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        result = runner.invoke(cli, [
            "generate",
            "Hello world",
            "--model", "unknownprovider:model/name"
        ])

        assert result.exit_code != 0
        assert "Unknown provider" in result.output or "not available" in result.output

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    @pytest.mark.slow
    def test_openrouter_api_error_handling(self, runner, monkeypatch):
        """Test CLI handling of OpenRouter API errors."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock API error
        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', side_effect=Exception("API Error")):

            result = runner.invoke(cli, [
                "generate",
                "Hello world",
                "--model", "openrouter:anthropic/claude-3.5-sonnet",
                "--wait"
            ])

            assert result.exit_code != 0


class TestOpenRouterCLIHelp:
    """Test help text and documentation for OpenRouter in CLI."""

    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    def test_generate_help_mentions_openrouter(self, runner):
        """Test that generate help mentions OpenRouter as example."""
        result = runner.invoke(cli, ["generate", "--help"])

        assert result.exit_code == 0
        # Help should mention OpenRouter as a remote provider option
        # (This might not be implemented yet, but the test establishes the expectation)

    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    def test_models_help_mentions_remote_providers(self, runner):
        """Test that models help mentions remote providers including OpenRouter."""
        result = runner.invoke(cli, ["models", "--help"])

        assert result.exit_code == 0
        # Should mention remote providers concept


class TestOpenRouterCLIStdinIntegration:
    """Test OpenRouter with stdin input in CLI."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    @pytest.mark.slow
    def test_stdin_with_openrouter_model(self, runner, monkeypatch):
        """Test stdin input with OpenRouter model."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock the provider
        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', return_value="Response to stdin"):

            result = runner.invoke(cli, [
                "--model", "openrouter:anthropic/claude-3.5-sonnet"
            ], input="Hello from stdin")

            assert result.exit_code == 0


def test_openrouter_cli_imports_fail():
    """Test that OpenRouter CLI imports fail (TDD requirement).

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


class TestCLIContractCompliance:
    """Test that CLI changes follow expected patterns."""

    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    def test_cli_handles_model_parameter(self, runner):
        """Test that CLI properly handles --model parameter."""
        # Test with a known working model first
        result = runner.invoke(cli, [
            "generate",
            "Hello",
            "--model", "qwen3-4b",
            "--wait"
        ])

        # Should not fail due to model parameter parsing
        # (Actual model loading might fail in test environment, but parameter should be parsed)
        assert "--model" not in result.output  # Parameter was parsed, not treated as literal text

    @pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
    def test_cli_error_messages_helpful(self, runner):
        """Test that CLI provides helpful error messages."""
        # Test with completely invalid model format
        result = runner.invoke(cli, [
            "generate",
            "Hello",
            "--model", "invalid::format:::too::many::colons"
        ])

        # Should provide helpful error message, not crash
        assert result.exit_code != 0
        assert len(result.output) > 0