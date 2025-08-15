"""Tests for unsafe mode embedding providers.

AIDEV-NOTE: These tests verify the unsafe mode embedding functionality without
actually calling remote APIs (which would be non-deterministic and costly).
"""

import os
import numpy as np
import warnings
from unittest.mock import Mock, patch

from steadytext.providers.base import UnsafeModeWarning
from steadytext.providers.openai import OpenAIProvider
from steadytext.providers.voyageai import VoyageAIProvider
from steadytext.providers.jina import JinaProvider


class TestVoyageAIProvider:
    """Test VoyageAI embedding provider (mocked)."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = VoyageAIProvider(api_key="test-key", model="voyage-3-large")
        assert provider.api_key == "test-key"
        assert provider.model == "voyage-3-large"

    def test_init_from_env(self, monkeypatch):
        """Test initialization from environment."""
        monkeypatch.setenv("VOYAGE_API_KEY", "env-key")
        provider = VoyageAIProvider(model="voyage-3-large")
        assert provider.api_key == "env-key"

    def test_is_available_no_key(self, monkeypatch):
        """Test availability check without API key."""
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
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
        assert len(models) == 10  # Should return 10 supported models
        assert "voyage-3" in models
        assert "voyage-3-lite" in models

    @patch("steadytext.providers.voyageai._get_voyageai")
    def test_embed_single_text(self, mock_get_voyageai, monkeypatch):
        """Test embedding single text with mocked VoyageAI."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock VoyageAI module and client
        mock_voyageai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        # Create a 1024-dimensional embedding for testing
        mock_embedding = np.random.randn(1024).tolist()
        mock_response.embeddings = [mock_embedding]

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
        assert result.shape == (1024,)
        # Result should be normalized version of mock_embedding
        assert np.allclose(np.linalg.norm(result), 1.0, rtol=1e-5)

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
        # Create 1024-dimensional embeddings for testing
        mock_embedding1 = np.random.randn(1024).tolist()
        mock_embedding2 = np.random.randn(1024).tolist()
        mock_response.embeddings = [mock_embedding1, mock_embedding2]

        mock_client.embed.return_value = mock_response
        mock_voyageai.Client.return_value = mock_client
        mock_get_voyageai.return_value = mock_voyageai

        provider = VoyageAIProvider(api_key="test", model="voyage-3-large")

        result = provider.embed(["Text 1", "Text 2"], seed=42)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)  # Should be averaged to single embedding
        # Result should be normalized
        assert np.allclose(np.linalg.norm(result), 1.0, rtol=1e-5)

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
        # Create a 1024-dimensional embedding for testing
        mock_embedding = np.random.randn(1024).tolist()
        mock_response.embeddings = [mock_embedding]

        mock_client.embed.return_value = mock_response
        mock_voyageai.Client.return_value = mock_client
        mock_get_voyageai.return_value = mock_voyageai

        provider = VoyageAIProvider(api_key="test", model="voyage-3-large")

        provider.embed("Query text", input_type="query")

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
        # Create a 1024-dimensional embedding for testing
        mock_embedding = np.random.randn(1024).tolist()
        mock_response.data = [Mock(embedding=mock_embedding)]

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
        assert result.shape == (1024,)
        # Result should be normalized version of mock_embedding
        assert np.allclose(np.linalg.norm(result), 1.0, rtol=1e-5)

        # Verify API call
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["model"] == "text-embedding-3-small"  # Default model
        assert call_args.kwargs["input"] == ["Test text"]

    @patch("steadytext.providers.openai._get_openai")
    def test_embed_batch_texts(self, mock_get_openai, monkeypatch):
        """Test embedding batch of texts with OpenAI."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock OpenAI module and client
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        # Create 1024-dimensional embeddings for testing
        mock_embedding1 = np.random.randn(1024).tolist()
        mock_embedding2 = np.random.randn(1024).tolist()
        mock_response.data = [
            Mock(embedding=mock_embedding1),
            Mock(embedding=mock_embedding2),
        ]

        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = OpenAIProvider(api_key="test", model="text-embedding-3-large")

        result = provider.embed(["Text 1", "Text 2"], seed=42)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)  # Should be averaged to single embedding
        # Result should be normalized
        assert np.allclose(np.linalg.norm(result), 1.0, rtol=1e-5)

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
        # Create a 1024-dimensional embedding for testing
        mock_embedding = np.random.randn(1024).tolist()
        mock_response.data = [Mock(embedding=mock_embedding)]

        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = OpenAIProvider(api_key="test", model="text-embedding-ada-002")

        provider.embed("Test text", model="text-embedding-3-small")

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
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

        # Mock VoyageAI
        mock_voyageai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        # Create a 1024-dimensional embedding for testing
        mock_embedding = np.random.randn(1024).tolist()
        mock_response.embeddings = [Mock(embedding=mock_embedding)]

        mock_client.embed.return_value = mock_response
        mock_voyageai.Client.return_value = mock_client
        mock_get_voyageai.return_value = mock_voyageai

        # Import here to ensure environment is set
        from steadytext import embed

        result = embed("Test text", model="voyageai:voyage-3-large", unsafe_mode=True)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)

    @patch("steadytext.providers.openai._get_openai")
    def test_embed_with_openai_model(self, mock_get_openai, monkeypatch):
        """Test core_embed function with OpenAI model."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock OpenAI
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        # Create a 1024-dimensional embedding for testing
        mock_embedding = np.random.randn(1024).tolist()
        mock_response.data = [Mock(embedding=mock_embedding)]

        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        # Import here to ensure environment is set
        from steadytext import embed

        result = embed(
            "Test text", model="openai:text-embedding-3-large", unsafe_mode=True
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)

    def test_embed_without_unsafe_mode(self, monkeypatch):
        """Test that remote embedding models fail without unsafe mode."""
        monkeypatch.delenv("STEADYTEXT_UNSAFE_MODE", raising=False)

        from steadytext import embed

        result = embed("Test text", model="voyageai:voyage-3-large", unsafe_mode=False)
        # Should fallback to zero vector when unsafe_mode is False for remote models
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)
        assert np.allclose(result, 0)  # Should be zero vector

    def test_embed_with_invalid_provider(self, monkeypatch):
        """Test embedding with invalid provider."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        from steadytext import embed

        result = embed("Test text", model="invalid:model", unsafe_mode=True)
        # Should fallback to local model when provider is invalid
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)
        # When models are loaded, falls back to local model (non-zero)
        # When models aren't loaded, returns zero vector
        if os.environ.get("STEADYTEXT_ALLOW_MODEL_DOWNLOADS") == "true":
            assert not np.allclose(result, 0)  # Should be non-zero (local model)
        else:
            assert np.allclose(result, 0)  # Should be zero vector

    @patch("steadytext.providers.voyageai._get_voyageai")
    def test_embed_batch_with_remote_model(self, mock_get_voyageai, monkeypatch):
        """Test batch embedding with remote model."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

        # Mock VoyageAI
        mock_voyageai = Mock()
        mock_client = Mock()
        mock_response = Mock()
        # Create 1024-dimensional embeddings for testing
        mock_embedding1 = np.random.randn(1024).tolist()
        mock_embedding2 = np.random.randn(1024).tolist()
        # For batch, the embed function would average them, so return a single embedding
        averaged_embedding = np.mean(
            [mock_embedding1, mock_embedding2], axis=0
        ).tolist()
        mock_response.embeddings = [Mock(embedding=averaged_embedding)]

        mock_client.embed.return_value = mock_response
        mock_voyageai.Client.return_value = mock_client
        mock_get_voyageai.return_value = mock_voyageai

        # Import here to ensure environment is set
        from steadytext import embed

        result = embed(
            ["Text 1", "Text 2"], model="voyageai:voyage-3-large", unsafe_mode=True
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)  # Should be a single averaged embedding


class TestJinaProvider:
    """Test Jina AI embedding provider (mocked)."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = JinaProvider(api_key="test-key", model="jina-embeddings-v3")
        assert provider.api_key == "test-key"
        assert provider.model == "jina-embeddings-v3"

    def test_init_from_env(self, monkeypatch):
        """Test initialization from environment."""
        monkeypatch.setenv("JINA_API_KEY", "env-key")
        provider = JinaProvider(model="jina-embeddings-v3")
        assert provider.api_key == "env-key"

    def test_is_available_no_key(self, monkeypatch):
        """Test availability check without API key."""
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        provider = JinaProvider(api_key=None)
        assert not provider.is_available()

    @patch("steadytext.providers.jina._get_requests")
    def test_is_available_with_requests(self, mock_get_requests):
        """Test availability with requests library available."""
        mock_get_requests.return_value = Mock()
        provider = JinaProvider(api_key="test", model="jina-embeddings-v3")
        assert provider.is_available()

    def test_supported_models(self):
        """Test getting supported models."""
        provider = JinaProvider()
        models = provider.get_supported_models()
        assert isinstance(models, list)
        assert "jina-embeddings-v3" in models
        assert "jina-embeddings-v2-base-en" in models

    @patch("steadytext.providers.jina._get_requests")
    def test_embed_single_text(self, mock_get_requests, monkeypatch):
        """Test embedding single text with mocked Jina API."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock requests module and response
        mock_requests = Mock()
        mock_response = Mock()
        # Create a 1024-dimensional embedding for testing
        mock_embedding = np.random.randn(1024).tolist()
        mock_response.json.return_value = {"data": [{"embedding": mock_embedding}]}
        mock_response.raise_for_status = Mock()

        mock_requests.post.return_value = mock_response
        mock_get_requests.return_value = mock_requests

        provider = JinaProvider(api_key="test", model="jina-embeddings-v3")

        # Should issue warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.embed("Test text", seed=42)
            assert len(w) == 1
            assert issubclass(w[0].category, UnsafeModeWarning)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)
        # Result should be normalized version of mock_embedding
        assert np.allclose(np.linalg.norm(result), 1.0, rtol=1e-5)

        # Verify API call
        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        assert call_args[0][0] == "https://api.jina.ai/v1/embeddings"
        assert call_args.kwargs["json"]["model"] == "jina-embeddings-v3"
        assert call_args.kwargs["json"]["input"] == ["Test text"]

    @patch("steadytext.providers.jina._get_requests")
    def test_embed_batch_texts(self, mock_get_requests, monkeypatch):
        """Test embedding batch of texts with mocked Jina API."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock requests module and response
        mock_requests = Mock()
        mock_response = Mock()
        # Create 1024-dimensional embeddings for testing
        mock_embedding1 = np.random.randn(1024).tolist()
        mock_embedding2 = np.random.randn(1024).tolist()
        mock_response.json.return_value = {
            "data": [{"embedding": mock_embedding1}, {"embedding": mock_embedding2}]
        }
        mock_response.raise_for_status = Mock()

        mock_requests.post.return_value = mock_response
        mock_get_requests.return_value = mock_requests

        provider = JinaProvider(api_key="test", model="jina-embeddings-v3")

        result = provider.embed(["Text 1", "Text 2"], seed=42)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)  # Should be averaged
        # Result should be normalized
        assert np.allclose(np.linalg.norm(result), 1.0, rtol=1e-5)

        # Verify API call
        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        assert call_args.kwargs["json"]["input"] == ["Text 1", "Text 2"]

    @patch("steadytext.providers.jina._get_requests")
    def test_embed_with_task_parameter(self, mock_get_requests, monkeypatch):
        """Test embedding with task parameter."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock requests module and response
        mock_requests = Mock()
        mock_response = Mock()
        # Create a 1024-dimensional embedding for testing
        mock_embedding = np.random.randn(1024).tolist()
        mock_response.json.return_value = {"data": [{"embedding": mock_embedding}]}
        mock_response.raise_for_status = Mock()

        mock_requests.post.return_value = mock_response
        mock_get_requests.return_value = mock_requests

        provider = JinaProvider(api_key="test", model="jina-embeddings-v3")

        provider.embed("Query text", task="retrieval.query")

        # Verify API call used task parameter
        call_args = mock_requests.post.call_args
        assert call_args.kwargs["json"]["task"] == "retrieval.query"

    @patch("steadytext.providers.jina._get_requests")
    def test_embed_error_handling(self, mock_get_requests, monkeypatch):
        """Test error handling in embed method."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock requests module to raise an error
        mock_requests = Mock()
        mock_requests.post.side_effect = Exception("API Error")
        mock_get_requests.return_value = mock_requests

        provider = JinaProvider(api_key="test", model="jina-embeddings-v3")

        result = provider.embed("Test text")
        assert result is None  # Should return None on error


# AIDEV-NOTE: These tests verify the new unsafe_mode embedding functionality
# for VoyageAI, OpenAI, and Jina providers without making actual API calls.
# The tests use mocking to simulate API responses and verify correct behavior.
