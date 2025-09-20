"""Contract tests for OpenRouterProvider initialization.

AIDEV-NOTE: These tests verify the initialization interface defined in the contract.
They MUST fail initially since OpenRouterProvider doesn't exist yet (TDD requirement).
"""

import pytest
import os
from unittest.mock import patch, Mock

# AIDEV-NOTE: This import will fail initially - this is expected for TDD
try:
    from steadytext.providers.openrouter import OpenRouterProvider
    OPENROUTER_PROVIDER_AVAILABLE = True
except ImportError:
    OPENROUTER_PROVIDER_AVAILABLE = False


class TestOpenRouterProviderInitialization:
    """Test OpenRouterProvider initialization according to contract."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = OpenRouterProvider(api_key="test-key", model="anthropic/claude-3.5-sonnet")
        assert provider.api_key == "test-key"
        assert provider.model == "anthropic/claude-3.5-sonnet"

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_init_default_model(self):
        """Test initialization with default model."""
        provider = OpenRouterProvider(api_key="test-key")
        assert provider.model == "anthropic/claude-3.5-sonnet"

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_init_from_environment(self, monkeypatch):
        """Test initialization reading API key from environment."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        provider = OpenRouterProvider()
        assert provider.api_key == "env-key"

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_init_missing_api_key_raises_error(self, monkeypatch):
        """Test that missing API key raises RuntimeError."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="API key is missing"):
            OpenRouterProvider(api_key=None)

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_init_invalid_model_format_raises_error(self):
        """Test that invalid model format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model format"):
            OpenRouterProvider(api_key="test-key", model="invalid-model-without-slash")

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_init_valid_model_formats(self):
        """Test that valid model formats are accepted."""
        valid_models = [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o-mini",
            "meta-llama/llama-3.1-8b-instruct",
            "google/gemma-2-9b-it"
        ]

        for model in valid_models:
            provider = OpenRouterProvider(api_key="test-key", model=model)
            assert provider.model == model

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_init_empty_api_key_raises_error(self):
        """Test that empty API key raises RuntimeError."""
        with pytest.raises(RuntimeError, match="API key is missing"):
            OpenRouterProvider(api_key="")

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_init_whitespace_api_key_raises_error(self):
        """Test that whitespace-only API key raises RuntimeError."""
        with pytest.raises(RuntimeError, match="API key is missing"):
            OpenRouterProvider(api_key="   ")

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_provider_name_property(self):
        """Test that provider_name property returns correct value."""
        provider = OpenRouterProvider(api_key="test-key")
        assert provider.provider_name == "OpenRouter"

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_inherits_from_remote_model_provider(self):
        """Test that OpenRouterProvider inherits from RemoteModelProvider."""
        from steadytext.providers.base import RemoteModelProvider
        provider = OpenRouterProvider(api_key="test-key")
        assert isinstance(provider, RemoteModelProvider)


def test_openrouter_provider_import_fails():
    """Test that OpenRouterProvider import fails (TDD requirement).

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


class TestOpenRouterProviderContractCompliance:
    """Test OpenRouterProvider implements the required contract interface."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_has_required_methods(self):
        """Test that OpenRouterProvider has all required contract methods."""
        provider = OpenRouterProvider(api_key="test-key")

        # Required methods from contract
        assert hasattr(provider, 'is_available')
        assert hasattr(provider, 'get_supported_models')
        assert hasattr(provider, 'generate')
        assert hasattr(provider, 'embed')
        assert hasattr(provider, 'get_model_info')

        # Check method signatures are callable
        assert callable(getattr(provider, 'is_available'))
        assert callable(getattr(provider, 'get_supported_models'))
        assert callable(getattr(provider, 'generate'))
        assert callable(getattr(provider, 'embed'))
        assert callable(getattr(provider, 'get_model_info'))

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_has_required_properties(self):
        """Test that OpenRouterProvider has all required properties."""
        provider = OpenRouterProvider(api_key="test-key")

        # Required properties
        assert hasattr(provider, 'api_key')
        assert hasattr(provider, 'model')
        assert hasattr(provider, 'provider_name')