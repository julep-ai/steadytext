"""Tests for unsafe mode embedding providers.

AIDEV-NOTE: These tests verify the unsafe mode embedding functionality without
actually calling remote APIs (which would be non-deterministic and costly).
"""

import numpy as np
import pytest
import warnings
from unittest.mock import Mock, patch

from steadytext.providers.base import UnsafeModeWarning
from steadytext.providers.openai import OpenAIProvider
from steadytext.providers.voyageai import VoyageAIProvider
from steadytext.providers.registry import get_provider


class TestVoyageAIProvider:
    """Test VoyageAI embedding provider (mocked)."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = VoyageAIProvider(api_key="test-key", model="voyage-3-large")
        assert provider.api_key == "test-key"
        assert provider.model == "voyage-3-large"

    def test_init_from_env(self, monkeypatch):
        """Test initialization from environment."""
        monkeypatch.setenv("VOYAGEAI_API_KEY", "env-key")
        provider = VoyageAIProvider(model="voyage-3-large")
        assert provider.api_key == "env-key"

    def test_is_available_no_key(self, monkeypatch):
        """Test availability check without API key."""
        monkeypatch.delenv("VOYAGEAI_API_KEY", raising=False)
        provider = VoyageAIProvider(api_key=None)
        assert not provider.is_available()

    @patch("steadytext.providers.voyageai._get_voyageai")
    def test_is_available_with_voyageai(self, mock_get_voyageai):
        """Test availability with VoyageAI library available."""
        mock_get_voyageai.return_value = Mock()
        provider = VoyageAIProvider(api_key="test", model="voyage-3-large")
        assert provider.is_available()

    def test_supported_models(self):
        """Test getting supported models."""
        provider = VoyageAIProvider()
        models = provider.get_supported_models()
        assert isinstance(models, list)
        assert len(models) == 0  # Empty list returned

    @patch("steadytext.providers.voyageai._get_voyageai")
    def test_embed_single_text(self, mock_get_voyageai, monkeypatch):
        """Test embedding single text with mocked VoyageAI."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock VoyageAI module and client
        mock_voyageai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.embeddings = [Mock(embedding=[0.1, 0.2, 0.3])]
        
        mock_client.embed.return_value = mock_response
        mock_voyageai.Client.return_value = mock_client
        mock_get_voyageai.return_value = mock_voyageai

        provider = VoyageAIProvider(api_key="test", model="voyage-3-large")

        # Should issue warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.embed("Test text", seed=42)
            assert len(w) == 1
            assert issubclass(w[0].category, UnsafeModeWarning)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])

        # Verify API call
        mock_client.embed.assert_called_once()
        call_args = mock_client.embed.call_args
        assert call_args.kwargs["model"] == "voyage-3-large"
        assert call_args.kwargs["texts"] == ["Test text"]
        assert call_args.kwargs["input_type"] == "document"

    @patch("steadytext.providers.voyageai._get_voyageai")
    def test_embed_batch_texts(self, mock_get_voyageai, monkeypatch):
        """Test embedding batch of texts with mocked VoyageAI."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock VoyageAI module and client
        mock_voyageai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.embeddings = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        
        mock_client.embed.return_value = mock_response
        mock_voyageai.Client.return_value = mock_client
        mock_get_voyageai.return_value = mock_voyageai

        provider = VoyageAIProvider(api_key="test", model="voyage-3-large")

        result = provider.embed(["Text 1", "Text 2"], seed=42)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(result[1], [0.4, 0.5, 0.6])

        # Verify API call
        mock_client.embed.assert_called_once()
        call_args = mock_client.embed.call_args
        assert call_args.kwargs["texts"] == ["Text 1", "Text 2"]

    @patch("steadytext.providers.voyageai._get_voyageai")
    def test_embed_with_custom_input_type(self, mock_get_voyageai, monkeypatch):
        """Test embedding with custom input type."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock VoyageAI module and client
        mock_voyageai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.embeddings = [Mock(embedding=[0.1, 0.2, 0.3])]
        
        mock_client.embed.return_value = mock_response
        mock_voyageai.Client.return_value = mock_client
        mock_get_voyageai.return_value = mock_voyageai

        provider = VoyageAIProvider(api_key="test", model="voyage-3-large")

        result = provider.embed("Query text", input_type="query")

        # Verify API call used custom input type
        call_args = mock_client.embed.call_args
        assert call_args.kwargs["input_type"] == "query"

    @patch("steadytext.providers.voyageai._get_voyageai")
    def test_embed_error_handling(self, mock_get_voyageai, monkeypatch):
        """Test error handling in embed method."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock VoyageAI module to raise an error
        mock_voyageai = Mock()
        mock_client = Mock()
        mock_client.embed.side_effect = Exception("API Error")
        mock_voyageai.Client.return_value = mock_client
        mock_get_voyageai.return_value = mock_voyageai

        provider = VoyageAIProvider(api_key="test", model="voyage-3-large")

        result = provider.embed("Test text")
        assert result is None  # Should return None on error


class TestOpenAIProviderEmbeddings:
    """Test OpenAI embedding functionality (mocked)."""

    @patch("steadytext.providers.openai._get_openai")
    def test_embed_single_text(self, mock_get_openai, monkeypatch):
        """Test embedding single text with OpenAI."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock OpenAI module and client
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = OpenAIProvider(api_key="test", model="text-embedding-3-large")

        # Should issue warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.embed("Test text", seed=42)
            assert len(w) == 1
            assert issubclass(w[0].category, UnsafeModeWarning)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])

        # Verify API call
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["model"] == "text-embedding-3-large"
        assert call_args.kwargs["input"] == ["Test text"]

    @patch("steadytext.providers.openai._get_openai")
    def test_embed_batch_texts(self, mock_get_openai, monkeypatch):
        """Test embedding batch of texts with OpenAI."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock OpenAI module and client
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = OpenAIProvider(api_key="test", model="text-embedding-3-large")

        result = provider.embed(["Text 1", "Text 2"], seed=42)

        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(result[1], [0.4, 0.5, 0.6])

        # Verify API call
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["input"] == ["Text 1", "Text 2"]

    @patch("steadytext.providers.openai._get_openai")
    def test_embed_with_custom_model(self, mock_get_openai, monkeypatch):
        """Test embedding with custom model specification."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock OpenAI module and client
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = OpenAIProvider(api_key="test", model="text-embedding-ada-002")

        result = provider.embed("Test text", model="text-embedding-3-small")

        # Verify API call used the override model
        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["model"] == "text-embedding-3-small"

    @patch("steadytext.providers.openai._get_openai")
    def test_embed_error_handling(self, mock_get_openai, monkeypatch):
        """Test error handling in embed method."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock OpenAI module to raise an error
        mock_openai = Mock()
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = OpenAIProvider(api_key="test", model="text-embedding-3-large")

        result = provider.embed("Test text")
        assert result is None  # Should return None on error


class TestCoreEmbedderIntegration:
    """Test integration with core embedder and unsafe mode."""

    @patch("steadytext.providers.voyageai._get_voyageai")
    def test_embed_with_voyageai_model(self, mock_get_voyageai, monkeypatch):
        """Test core_embed function with VoyageAI model."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("VOYAGEAI_API_KEY", "test-key")

        # Mock VoyageAI
        mock_voyageai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.embeddings = [Mock(embedding=[0.1, 0.2, 0.3])]
        
        mock_client.embed.return_value = mock_response
        mock_voyageai.Client.return_value = mock_client
        mock_get_voyageai.return_value = mock_voyageai

        # Import here to ensure environment is set
        from steadytext import embed

        result = embed("Test text", model="voyageai:voyage-3-large", unsafe_mode=True)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    @patch("steadytext.providers.openai._get_openai")
    def test_embed_with_openai_model(self, mock_get_openai, monkeypatch):
        """Test core_embed function with OpenAI model."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock OpenAI
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        # Import here to ensure environment is set
        from steadytext import embed

        result = embed("Test text", model="openai:text-embedding-3-large", unsafe_mode=True)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_embed_without_unsafe_mode(self, monkeypatch):
        """Test that remote embedding models fail without unsafe mode."""
        monkeypatch.delenv("STEADYTEXT_UNSAFE_MODE", raising=False)

        from steadytext import embed

        result = embed("Test text", model="voyageai:voyage-3-large", unsafe_mode=False)
        assert result is None  # Should fail and return None

    def test_embed_with_invalid_provider(self, monkeypatch):
        """Test embedding with invalid provider."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        from steadytext import embed

        result = embed("Test text", model="invalid:model", unsafe_mode=True)
        assert result is None  # Should fail and return None

    @patch("steadytext.providers.voyageai._get_voyageai")
    def test_embed_batch_with_remote_model(self, mock_get_voyageai, monkeypatch):
        """Test batch embedding with remote model."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("VOYAGEAI_API_KEY", "test-key")

        # Mock VoyageAI
        mock_voyageai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.embeddings = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        
        mock_client.embed.return_value = mock_response
        mock_voyageai.Client.return_value = mock_client
        mock_get_voyageai.return_value = mock_voyageai

        # Import here to ensure environment is set
        from steadytext import embed

        result = embed(["Text 1", "Text 2"], model="voyageai:voyage-3-large", unsafe_mode=True)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)


# AIDEV-NOTE: These tests verify the new unsafe_mode embedding functionality
# for both VoyageAI and OpenAI providers without making actual API calls.
# The tests use mocking to simulate API responses and verify correct behavior.