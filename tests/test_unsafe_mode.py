"""Tests for unsafe mode remote model providers.

AIDEV-NOTE: These tests verify the unsafe mode functionality without
actually calling remote APIs (which would be non-deterministic and costly).
"""

import pytest
import warnings
from unittest.mock import Mock, patch

from steadytext.providers.base import RemoteModelProvider, UnsafeModeWarning
from steadytext.providers.openai import OpenAIProvider
from steadytext.providers.cerebras import CerebrasProvider
from steadytext.providers.registry import (
    is_remote_model,
    parse_remote_model,
    get_provider,
    is_unsafe_mode_enabled,
    list_providers,
)


class TestRemoteModelDetection:
    """Test remote model detection and parsing."""

    def test_is_remote_model(self):
        """Test detection of remote model strings."""
        assert is_remote_model("openai:gpt-4")
        assert is_remote_model("cerebras:llama3.1-8b")
        assert is_remote_model("voyageai:voyage-3-large")
        assert is_remote_model("jina:jina-embeddings-v3")
        assert not is_remote_model("gemma-3n-2b")
        assert not is_remote_model("qwen3-4b")
        assert not is_remote_model(None)
        assert not is_remote_model("")
        assert not is_remote_model("unknown:model")

    def test_parse_remote_model(self):
        """Test parsing of remote model strings."""
        provider, model = parse_remote_model("openai:gpt-4o-mini")
        assert provider == "openai"
        assert model == "gpt-4o-mini"

        provider, model = parse_remote_model("cerebras:llama3.1-8b")
        assert provider == "cerebras"
        assert model == "llama3.1-8b"

        provider, model = parse_remote_model("voyageai:voyage-3-large")
        assert provider == "voyageai"
        assert model == "voyage-3-large"

        provider, model = parse_remote_model("jina:jina-embeddings-v3")
        assert provider == "jina"
        assert model == "jina-embeddings-v3"

        # Test invalid formats
        with pytest.raises(ValueError, match="Invalid remote model format"):
            parse_remote_model("no-colon")

        with pytest.raises(ValueError, match="Unknown provider"):
            parse_remote_model("unknown:model")


class TestUnsafeModeEnvironment:
    """Test unsafe mode environment variable handling."""

    def test_unsafe_mode_disabled_by_default(self, monkeypatch):
        """Test that unsafe mode is disabled by default."""
        monkeypatch.delenv("STEADYTEXT_UNSAFE_MODE", raising=False)
        assert not is_unsafe_mode_enabled()

    def test_unsafe_mode_enabled(self, monkeypatch):
        """Test enabling unsafe mode."""
        for value in ["true", "True", "TRUE", "1", "yes"]:
            monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", value)
            assert is_unsafe_mode_enabled()

    def test_unsafe_mode_disabled(self, monkeypatch):
        """Test disabling unsafe mode."""
        for value in ["false", "False", "FALSE", "0", "no", "anything"]:
            monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", value)
            assert not is_unsafe_mode_enabled()


class TestRemoteModelProvider:
    """Test base RemoteModelProvider functionality."""

    def test_warning_issued_once(self):
        """Test that warning is issued only once per provider."""

        class TestProvider(RemoteModelProvider):
            @property
            def provider_name(self):
                return "TestProvider"

            def is_available(self):
                return True

            def generate(self, prompt, **kwargs):
                self._issue_warning()
                return "test"

            def generate_iter(self, prompt, **kwargs):
                self._issue_warning()
                yield "test"

        provider = TestProvider()

        # First call should issue warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            provider.generate("test")
            assert len(w) == 1
            assert issubclass(w[0].category, UnsafeModeWarning)
            assert "UNSAFE MODE WARNING" in str(w[0].message)

        # Second call should not issue warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            provider.generate("test again")
            assert len(w) == 0


class TestOpenAIProvider:
    """Test OpenAI provider (mocked)."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini")
        assert provider.api_key == "test-key"
        assert provider.model == "gpt-4o-mini"

    def test_init_from_env(self, monkeypatch):
        """Test initialization from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        provider = OpenAIProvider(model="gpt-4o-mini")
        assert provider.api_key == "env-key"

    def test_is_available_no_key(self, monkeypatch):
        """Test availability check without API key."""
        # Ensure OPENAI_API_KEY is not set in environment
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIProvider(api_key=None)
        assert not provider.is_available()

    @patch("steadytext.providers.openai._get_openai")
    def test_is_available_with_openai(self, mock_get_openai):
        """Test availability with OpenAI library available."""
        mock_get_openai.return_value = Mock()
        provider = OpenAIProvider(api_key="test", model="gpt-4o-mini")
        assert provider.is_available()

    def test_supported_models(self):
        """Test getting supported models."""
        provider = OpenAIProvider()
        models = provider.get_supported_models()
        # No static model list anymore - let provider handle it
        assert isinstance(models, list)
        assert len(models) == 0  # Empty list returned

    @patch("steadytext.providers.openai._get_openai")
    def test_generate_mock(self, mock_get_openai, monkeypatch):
        """Test generation with mocked OpenAI."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock OpenAI module and client
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated text"))]

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = OpenAIProvider(api_key="test", model="gpt-4o-mini")

        # Should issue warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.generate("Test prompt", seed=42)
            assert len(w) == 1
            assert issubclass(w[0].category, UnsafeModeWarning)

        assert result == "Generated text"

        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o-mini"
        assert call_args.kwargs["seed"] == 42
        assert call_args.kwargs["temperature"] == 0.0


class TestCerebrasProvider:
    """Test Cerebras provider (mocked)."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = CerebrasProvider(api_key="test-key", model="llama3.1-8b")
        assert provider.api_key == "test-key"
        assert provider.model == "llama3.1-8b"

    def test_init_from_env(self, monkeypatch):
        """Test initialization from environment."""
        monkeypatch.setenv("CEREBRAS_API_KEY", "env-key")
        provider = CerebrasProvider(model="llama3.1-8b")
        assert provider.api_key == "env-key"

    def test_supported_models(self):
        """Test getting supported models."""
        provider = CerebrasProvider()
        models = provider.get_supported_models()
        # No static model list anymore - let provider handle it
        assert isinstance(models, list)
        assert len(models) == 0  # Empty list returned

    @patch("steadytext.providers.cerebras._get_openai")
    def test_uses_openai_client(self, mock_get_openai, monkeypatch):
        """Test that Cerebras uses OpenAI client with custom base URL."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock OpenAI module and client
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Cerebras generated text"))]

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = CerebrasProvider(api_key="test", model="llama3.1-8b")

        result = provider.generate("Test prompt", seed=42)
        assert result == "Cerebras generated text"

        # Verify OpenAI client was created with Cerebras base URL
        mock_openai.OpenAI.assert_called_once_with(
            base_url="https://api.cerebras.ai/v1", api_key="test"
        )


class TestProviderRegistry:
    """Test provider registry functionality."""

    def test_list_providers(self):
        """Test listing available providers."""
        providers = list_providers()
        assert "openai" in providers
        assert "cerebras" in providers
        assert "voyageai" in providers
        assert "jina" in providers
        assert len(providers) >= 4

    def test_get_provider_unsafe_mode_required(self, monkeypatch):
        """Test that get_provider requires unsafe mode."""
        monkeypatch.delenv("STEADYTEXT_UNSAFE_MODE", raising=False)

        with pytest.raises(RuntimeError, match="Remote models require unsafe mode"):
            get_provider("openai:gpt-4")

    @patch("steadytext.providers.openai.OpenAIProvider.is_available")
    def test_get_provider_not_available(self, mock_is_available, monkeypatch):
        """Test error when provider not available."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        mock_is_available.return_value = False

        with pytest.raises(RuntimeError, match="Provider openai is not available"):
            get_provider("openai:gpt-4")


class TestIntegration:
    """Test integration with main generate function."""

    @patch("steadytext.providers.openai._get_openai")
    def test_generate_with_remote_model(self, mock_get_openai, monkeypatch):
        """Test generate function with remote model."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock OpenAI
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Remote generated text"))]

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        # Import here to ensure environment is set
        from steadytext import generate

        result = generate("Test prompt", model="openai:gpt-4o-mini")
        assert result == "Remote generated text"

    def test_generate_without_unsafe_mode(self, monkeypatch):
        """Test that remote models fail without unsafe mode."""
        monkeypatch.delenv("STEADYTEXT_UNSAFE_MODE", raising=False)

        from steadytext import generate

        result = generate("Test prompt", model="openai:gpt-4")
        assert result is None  # Should fail and return None

    @patch("steadytext.providers.openai._get_openai")
    def test_generate_with_structured_output(self, mock_get_openai, monkeypatch):
        """Test structured generation with remote model."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock OpenAI
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content='{"name": "Alice", "age": 30}'))
        ]

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        # Import here to ensure environment is set
        from steadytext import generate

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        result = generate("Create a person", model="openai:gpt-4o-mini", schema=schema)
        assert result == '{"name": "Alice", "age": 30}'

        # Verify API call included response_format
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["response_format"] == {"type": "json_object"}


# AIDEV-NOTE: Additional tests could be added for:
# - Streaming generation with remote models
# - Error handling for API failures
# - Regex and choices with remote models
# - CLI command testing
# However, these would require more complex mocking
