"""Integration tests for OpenRouter registry integration.

AIDEV-NOTE: These tests verify OpenRouter integration with the provider registry.
They MUST fail initially since OpenRouterProvider doesn't exist yet (TDD requirement).
Tests focus on end-to-end registry behavior rather than unit testing individual methods.
"""

import pytest
import os
from unittest.mock import patch, Mock
from typing import Dict, Type, List

# AIDEV-NOTE: These imports will fail initially - this is expected for TDD
try:
    from steadytext.providers.openrouter import OpenRouterProvider
    OPENROUTER_PROVIDER_AVAILABLE = True
except ImportError:
    OPENROUTER_PROVIDER_AVAILABLE = False

try:
    from steadytext.providers.registry import (
        PROVIDER_REGISTRY,
        get_provider,
        parse_remote_model,
        is_remote_model,
        list_providers,
        list_remote_models,
        is_unsafe_mode_enabled
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False


class TestOpenRouterRegistryIntegration:
    """Test OpenRouter integration with provider registry."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_openrouter_in_provider_registry(self):
        """Test that OpenRouter is registered in PROVIDER_REGISTRY."""
        assert "openrouter" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["openrouter"] == OpenRouterProvider

        # Verify existing providers are still there
        assert "openai" in PROVIDER_REGISTRY
        assert "cerebras" in PROVIDER_REGISTRY
        assert "voyageai" in PROVIDER_REGISTRY
        assert "jina" in PROVIDER_REGISTRY

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_is_remote_model_detects_openrouter(self):
        """Test that is_remote_model correctly identifies OpenRouter models."""
        # Valid OpenRouter models
        assert is_remote_model("openrouter:anthropic/claude-3.5-sonnet") is True
        assert is_remote_model("openrouter:openai/gpt-4o-mini") is True
        assert is_remote_model("openrouter:meta-llama/llama-3.1-8b-instruct") is True

        # Non-OpenRouter models
        assert is_remote_model("openai:gpt-4o-mini") is True
        assert is_remote_model("qwen3-4b") is False
        assert is_remote_model(None) is False
        assert is_remote_model("") is False

        # Invalid formats
        assert is_remote_model("openrouter") is False
        assert is_remote_model("unknown:model") is False

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_parse_remote_model_handles_openrouter(self):
        """Test that parse_remote_model correctly parses OpenRouter models."""
        # Valid OpenRouter models
        provider, model = parse_remote_model("openrouter:anthropic/claude-3.5-sonnet")
        assert provider == "openrouter"
        assert model == "anthropic/claude-3.5-sonnet"

        provider, model = parse_remote_model("openrouter:openai/gpt-4o-mini")
        assert provider == "openrouter"
        assert model == "openai/gpt-4o-mini"

        # Complex model names
        provider, model = parse_remote_model("openrouter:meta-llama/llama-3.1-70b-instruct:free")
        assert provider == "openrouter"
        assert model == "meta-llama/llama-3.1-70b-instruct:free"

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_parse_remote_model_invalid_openrouter_format(self):
        """Test that parse_remote_model raises error for invalid formats."""
        with pytest.raises(ValueError, match="Invalid remote model format"):
            parse_remote_model("openrouter")

        with pytest.raises(ValueError, match="Unknown provider"):
            parse_remote_model("unknownprovider:model")

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_list_providers_includes_openrouter(self):
        """Test that list_providers includes OpenRouter."""
        providers = list_providers()
        assert "openrouter" in providers

        # Verify all expected providers are present
        expected_providers = {"openai", "cerebras", "voyageai", "jina", "openrouter"}
        assert set(providers) >= expected_providers

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    @pytest.mark.slow
    def test_list_remote_models_includes_openrouter(self):
        """Test that list_remote_models includes OpenRouter models."""
        models = list_remote_models()
        assert "openrouter" in models
        assert isinstance(models["openrouter"], list)

        # Should handle errors gracefully and return empty list if API fails
        # In test environment, this will likely be empty

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_get_provider_unsafe_mode_required(self, monkeypatch):
        """Test that get_provider requires unsafe mode for OpenRouter."""
        # Disable unsafe mode
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "false")

        with pytest.raises(RuntimeError, match="Remote models require unsafe mode"):
            get_provider("openrouter:anthropic/claude-3.5-sonnet")

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_get_provider_missing_api_key(self, monkeypatch):
        """Test that get_provider raises error when OpenRouter API key is missing."""
        # Enable unsafe mode but remove API key
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="Provider openrouter is not available"):
            get_provider("openrouter:anthropic/claude-3.5-sonnet")

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    @pytest.mark.slow
    def test_get_provider_with_valid_api_key(self, monkeypatch):
        """Test that get_provider successfully creates OpenRouter provider with valid API key."""
        # Enable unsafe mode and set API key
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-12345678901234567890")

        # Mock the provider to avoid actual API calls
        with patch.object(OpenRouterProvider, 'is_available', return_value=True):
            provider = get_provider("openrouter:anthropic/claude-3.5-sonnet")
            assert isinstance(provider, OpenRouterProvider)
            assert provider.api_key == "sk-or-test-key-12345678901234567890"
            assert provider.model == "anthropic/claude-3.5-sonnet"

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    @pytest.mark.slow
    def test_get_provider_with_explicit_api_key(self, monkeypatch):
        """Test that get_provider uses explicit API key over environment."""
        # Enable unsafe mode
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key")

        # Mock the provider to avoid actual API calls
        with patch.object(OpenRouterProvider, 'is_available', return_value=True):
            provider = get_provider(
                "openrouter:anthropic/claude-3.5-sonnet",
                api_key="sk-or-explicit-key"
            )
            assert isinstance(provider, OpenRouterProvider)
            assert provider.api_key == "sk-or-explicit-key"
            assert provider.model == "anthropic/claude-3.5-sonnet"

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_get_provider_unavailable_provider(self, monkeypatch):
        """Test that get_provider raises error when provider.is_available() returns False."""
        # Enable unsafe mode and set API key
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        # Mock the provider to return False for is_available
        with patch.object(OpenRouterProvider, 'is_available', return_value=False):
            with pytest.raises(RuntimeError, match="Provider openrouter is not available"):
                get_provider("openrouter:anthropic/claude-3.5-sonnet")


class TestOpenRouterRegistryEdgeCases:
    """Test edge cases in OpenRouter registry integration."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_openrouter_model_with_multiple_colons(self):
        """Test OpenRouter models with multiple colons in the name."""
        # Some OpenRouter models have colons in their actual names
        provider, model = parse_remote_model("openrouter:meta-llama/llama-3.1-70b-instruct:free")
        assert provider == "openrouter"
        assert model == "meta-llama/llama-3.1-70b-instruct:free"

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_openrouter_case_sensitivity(self):
        """Test that OpenRouter provider name is case sensitive."""
        assert is_remote_model("openrouter:anthropic/claude-3.5-sonnet") is True
        assert is_remote_model("OpenRouter:anthropic/claude-3.5-sonnet") is False
        assert is_remote_model("OPENROUTER:anthropic/claude-3.5-sonnet") is False

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_registry_maintains_order(self):
        """Test that adding OpenRouter maintains expected registry structure."""
        # Verify the registry is a dict and maintains expected providers
        assert isinstance(PROVIDER_REGISTRY, dict)

        # Get all provider names
        provider_names = list(PROVIDER_REGISTRY.keys())

        # Verify OpenRouter is present
        assert "openrouter" in provider_names

        # Verify all expected providers are present (order doesn't matter for functionality)
        expected = {"openai", "cerebras", "voyageai", "jina", "openrouter"}
        assert set(provider_names) >= expected


def test_openrouter_registry_imports_fail():
    """Test that OpenRouter registry imports fail (TDD requirement).

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


class TestRegistryContractCompliance:
    """Test that registry changes follow the contract specifications."""

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_provider_registry_structure(self):
        """Test that PROVIDER_REGISTRY has expected structure."""
        # Registry should be a dict mapping strings to classes
        assert isinstance(PROVIDER_REGISTRY, dict)

        for provider_name, provider_class in PROVIDER_REGISTRY.items():
            assert isinstance(provider_name, str)
            assert isinstance(provider_class, type)

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_unsafe_mode_detection(self):
        """Test that unsafe mode detection works correctly."""
        # Test various environment variable values
        with patch.dict(os.environ, {"STEADYTEXT_UNSAFE_MODE": "true"}):
            assert is_unsafe_mode_enabled() is True

        with patch.dict(os.environ, {"STEADYTEXT_UNSAFE_MODE": "1"}):
            assert is_unsafe_mode_enabled() is True

        with patch.dict(os.environ, {"STEADYTEXT_UNSAFE_MODE": "yes"}):
            assert is_unsafe_mode_enabled() is True

        with patch.dict(os.environ, {"STEADYTEXT_UNSAFE_MODE": "false"}):
            assert is_unsafe_mode_enabled() is False

        with patch.dict(os.environ, {"STEADYTEXT_UNSAFE_MODE": "0"}):
            assert is_unsafe_mode_enabled() is False

        with patch.dict(os.environ, {}, clear=True):
            assert is_unsafe_mode_enabled() is False

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_list_providers_returns_list(self):
        """Test that list_providers returns a list of strings."""
        providers = list_providers()
        assert isinstance(providers, list)

        for provider in providers:
            assert isinstance(provider, str)
            assert len(provider) > 0

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_list_remote_models_returns_dict(self):
        """Test that list_remote_models returns correctly structured dict."""
        models = list_remote_models()
        assert isinstance(models, dict)

        for provider_name, model_list in models.items():
            assert isinstance(provider_name, str)
            assert isinstance(model_list, list)

            for model in model_list:
                assert isinstance(model, str)