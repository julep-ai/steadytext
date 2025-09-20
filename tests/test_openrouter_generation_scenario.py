"""Test OpenRouter provider basic text generation user scenarios.

AIDEV-NOTE: These tests represent complete user workflows for text generation
with OpenRouter models as described in the quickstart guide. Tests MUST fail
initially since OpenRouterProvider doesn't exist yet (TDD approach).
"""

import os
import pytest
import warnings
from unittest.mock import Mock, patch

# AIDEV-NOTE: These imports will fail initially - this is expected for TDD
from steadytext.providers.openrouter import OpenRouterProvider, OpenRouterError
from steadytext.providers.base import UnsafeModeWarning
from steadytext.providers.registry import get_provider


class TestOpenRouterBasicGeneration:
    """Test basic text generation scenarios with OpenRouter."""

    def test_provider_initialization_with_api_key(self):
        """Test OpenRouter provider initialization with API key."""
        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            model="anthropic/claude-3.5-sonnet"
        )
        assert provider.api_key == "sk-or-test-key"
        assert provider.model == "anthropic/claude-3.5-sonnet"

    def test_provider_initialization_from_environment(self, monkeypatch):
        """Test OpenRouter provider initialization from OPENROUTER_API_KEY."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key")
        provider = OpenRouterProvider(model="anthropic/claude-3.5-sonnet")
        assert provider.api_key == "sk-or-env-key"

    def test_provider_availability_without_api_key(self, monkeypatch):
        """Test provider availability check when API key is missing."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        provider = OpenRouterProvider(api_key=None)
        assert not provider.is_available()

    def test_supported_models_list(self):
        """Test that provider returns list of supported models."""
        provider = OpenRouterProvider()
        models = provider.get_supported_models()
        assert isinstance(models, list)
        # Should include popular models from quickstart
        expected_models = [
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-haiku",
            "openai/gpt-4",
            "openai/gpt-4-turbo",
            "meta-llama/llama-3.1-70b-instruct"
        ]
        for model in expected_models:
            assert model in models

    @pytest.mark.slow
    @patch("steadytext.providers.openrouter._get_openai")
    def test_basic_text_generation_scenario(self, mock_get_openai, monkeypatch):
        """Test the basic text generation scenario from quickstart.md.

        Scenario: User wants to generate text using Claude 3.5 Sonnet via OpenRouter.
        """
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        # Mock OpenAI-compatible client (OpenRouter uses OpenAI client format)
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Quantum computing is a revolutionary technology that uses quantum mechanical phenomena..."))]

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            model="anthropic/claude-3.5-sonnet"
        )

        # Should issue unsafe mode warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.generate(
                "Explain quantum computing in simple terms",
                temperature=0.0,
                seed=42
            )
            assert len(w) == 1
            assert issubclass(w[0].category, UnsafeModeWarning)
            assert "UNSAFE MODE WARNING" in str(w[0].message)

        assert "quantum computing" in result.lower()
        assert len(result) > 50  # Should be a substantial response

        # Verify API call to OpenRouter
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "anthropic/claude-3.5-sonnet"
        assert "Explain quantum computing" in call_args.kwargs["messages"][0]["content"]

    @pytest.mark.slow
    @patch("steadytext.providers.openrouter._get_openai")
    def test_advanced_generation_parameters(self, mock_get_openai, monkeypatch):
        """Test advanced generation parameters scenario from quickstart.md."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock OpenAI-compatible client
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="In circuits of light and logic dance,\nAlgorithms weave their coded trance..."))]

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            model="anthropic/claude-3.5-sonnet"
        )

        result = provider.generate(
            "Write a creative poem",
            temperature=0.9,
            max_tokens=150,
            top_p=0.9,
            seed=123
        )

        assert isinstance(result, str)
        assert len(result) > 20

        # Verify advanced parameters were passed correctly
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.9
        assert call_args.kwargs["max_tokens"] == 150
        assert call_args.kwargs["top_p"] == 0.9
        assert call_args.kwargs["seed"] == 123

    @pytest.mark.slow
    def test_integration_with_steadytext_generate_function(self, monkeypatch):
        """Test integration with main SteadyText generate function.

        This simulates the drop-in replacement scenario from quickstart.md.
        """
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        # This import should work after OpenRouter provider is implemented
        from steadytext import generate

        with patch("steadytext.providers.openrouter._get_openai") as mock_get_openai:
            # Mock OpenAI-compatible client
            mock_openai = Mock()
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Hello from OpenRouter!"))]

            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client
            mock_get_openai.return_value = mock_openai

            # This should work exactly like the quickstart example
            result = generate(
                "Hello world",
                model="openrouter:anthropic/claude-3.5-sonnet",
                unsafe_mode=True
            )

            assert result == "Hello from OpenRouter!"

    def test_provider_registry_integration(self, monkeypatch):
        """Test that OpenRouter provider is properly registered."""
        from steadytext.providers.registry import list_providers, is_remote_model, parse_remote_model

        # Check provider is in registry
        providers = list_providers()
        assert "openrouter" in providers

        # Check remote model detection
        assert is_remote_model("openrouter:anthropic/claude-3.5-sonnet")
        assert is_remote_model("openrouter:openai/gpt-4")

        # Check model parsing
        provider, model = parse_remote_model("openrouter:anthropic/claude-3.5-sonnet")
        assert provider == "openrouter"
        assert model == "anthropic/claude-3.5-sonnet"

    @pytest.mark.slow
    def test_unsafe_mode_requirement(self, monkeypatch):
        """Test that OpenRouter requires unsafe mode."""
        monkeypatch.delenv("STEADYTEXT_UNSAFE_MODE", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        with pytest.raises(RuntimeError, match="Remote models require unsafe mode"):
            get_provider("openrouter:anthropic/claude-3.5-sonnet")

    @pytest.mark.slow
    def test_multiple_model_support(self, monkeypatch):
        """Test support for different OpenRouter models."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        test_cases = [
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3-haiku",
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "meta-llama/llama-3.1-70b-instruct",
            "google/gemini-pro"
        ]

        for model_name in test_cases:
            provider = OpenRouterProvider(
                api_key="sk-or-test-key",
                model=model_name
            )
            assert provider.model == model_name
            assert provider.is_available()


class TestOpenRouterConfiguration:
    """Test OpenRouter provider configuration scenarios."""

    def test_custom_base_url_configuration(self):
        """Test custom base URL configuration."""
        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            model="anthropic/claude-3.5-sonnet",
            base_url="https://openrouter.ai/api/v1"
        )
        assert provider.base_url == "https://openrouter.ai/api/v1"

    def test_timeout_configuration(self):
        """Test timeout configuration."""
        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            model="anthropic/claude-3.5-sonnet",
            timeout=60
        )
        assert provider.timeout == 60

    def test_max_retries_configuration(self):
        """Test max retries configuration."""
        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            model="anthropic/claude-3.5-sonnet",
            max_retries=5
        )
        assert provider.max_retries == 5

    def test_environment_variable_configuration(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key")
        monkeypatch.setenv("OPENROUTER_TIMEOUT", "60")
        monkeypatch.setenv("OPENROUTER_MAX_RETRIES", "3")
        monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        provider = OpenRouterProvider(model="anthropic/claude-3.5-sonnet")

        assert provider.api_key == "sk-or-env-key"
        # These would be read from environment if implemented
        # assert provider.timeout == 60
        # assert provider.max_retries == 3
        # assert provider.base_url == "https://openrouter.ai/api/v1"


# AIDEV-NOTE: These tests will fail initially because:
# 1. OpenRouterProvider class doesn't exist yet
# 2. OpenRouterError exception doesn't exist yet
# 3. Provider registration for "openrouter" isn't implemented
# 4. This is expected and correct for TDD approach
# 5. Tests represent complete user scenarios from quickstart.md