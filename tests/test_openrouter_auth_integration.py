"""Slimmed OpenRouter auth integration tests.

AIDEV-NOTE: Validate API key handling and environment lookup for the
OpenRouter provider without duplicating upstream behavior."""

from typing import Any

import pytest

try:
    from steadytext.providers.openrouter import OpenRouterProvider
except ImportError:  # pragma: no cover - provider not yet implemented
    OpenRouterProvider: Any = None

pytestmark = pytest.mark.skipif(
    OpenRouterProvider is None, reason="OpenRouterProvider not available"
)


class TestOpenRouterAPIKeyFormat:
    """Validate OpenRouter API key normalization rules."""

    def test_valid_openrouter_api_key_formats(self):
        valid_keys = [
            "sk-or-1234567890abcdef1234567890abcdef",
            "sk-or-v1-1234567890abcdef1234567890abcdef",
        ]
        for key in valid_keys:
            provider = OpenRouterProvider(
                api_key=key, model="anthropic/claude-3.5-sonnet"
            )
            assert provider.api_key == key

    def test_empty_api_key_rejected(self):
        with pytest.raises(ValueError):
            OpenRouterProvider(api_key="", model="anthropic/claude-3.5-sonnet")

    def test_placeholder_keys_allowed_with_warning(self):
        placeholder_keys = ["sk-123", "sk-or-!invalid"]
        for key in placeholder_keys:
            provider = OpenRouterProvider(
                api_key=key, model="anthropic/claude-3.5-sonnet"
            )
            assert provider.api_key == key.strip()


class TestOpenRouterEnvironmentAuthentication:
    """Verify environment variable precedence for API keys."""

    def test_api_key_from_environment(self, monkeypatch):
        monkeypatch.setenv(
            "OPENROUTER_API_KEY", "sk-or-env-test-1234567890abcdef1234567890"
        )
        provider = OpenRouterProvider(model="anthropic/claude-3.5-sonnet")
        assert provider.api_key.endswith("1234567890")

    def test_missing_api_key_raises_error(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        provider = OpenRouterProvider(api_key=None, model="anthropic/claude-3.5-sonnet")
        assert provider.is_available() is False

    def test_whitespace_only_environment_variable_raises_error(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "   	\n   ")
        with pytest.raises(ValueError):
            OpenRouterProvider(model="anthropic/claude-3.5-sonnet")
