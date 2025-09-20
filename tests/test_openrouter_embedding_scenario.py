"""Test OpenRouter provider embedding generation user scenarios.

AIDEV-NOTE: These tests represent complete embedding workflows with OpenRouter
models as described in the quickstart guide. Tests MUST fail initially since
OpenRouterProvider doesn't exist yet (TDD approach).
"""

import os
import pytest
import warnings
import numpy as np
from unittest.mock import Mock, patch

# AIDEV-NOTE: These imports will fail initially - this is expected for TDD
from steadytext.providers.openrouter import OpenRouterProvider, OpenRouterError
from steadytext.providers.base import UnsafeModeWarning


class TestOpenRouterEmbeddingGeneration:
    """Test embedding generation scenarios with OpenRouter."""

    @pytest.mark.slow
    @patch("steadytext.providers.openrouter._get_openai")
    def test_basic_embedding_scenario(self, mock_get_openai, monkeypatch):
        """Test basic text embedding scenario from quickstart.md.

        Scenario: User wants to generate embeddings using OpenRouter.
        """
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        # Mock OpenAI-compatible embeddings client
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()

        # Create mock 1536-dimensional embeddings (OpenAI text-embedding-3-small default)
        mock_embedding1 = np.random.randn(1536).tolist()
        mock_embedding2 = np.random.randn(1536).tolist()
        mock_response.data = [
            Mock(embedding=mock_embedding1),
            Mock(embedding=mock_embedding2),
        ]

        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            model="openai/text-embedding-3-small"
        )

        # Should issue unsafe mode warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = provider.embed(
                ["Hello world", "Goodbye world"],
                seed=42
            )
            assert len(w) == 1
            assert issubclass(w[0].category, UnsafeModeWarning)

        # Should return averaged and normalized embedding
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)  # SteadyText normalizes to 1024 dimensions
        assert np.allclose(np.linalg.norm(result), 1.0, rtol=1e-5)  # Should be normalized

        # Verify API call to OpenRouter
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["model"] == "text-embedding-3-small"
        assert call_args.kwargs["input"] == ["Hello world", "Goodbye world"]

    @pytest.mark.slow
    @patch("steadytext.providers.openrouter._get_openai")
    def test_single_text_embedding(self, mock_get_openai, monkeypatch):
        """Test embedding single text with OpenRouter."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock OpenAI-compatible embeddings client
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()

        # Create mock 1536-dimensional embedding
        mock_embedding = np.random.randn(1536).tolist()
        mock_response.data = [Mock(embedding=mock_embedding)]

        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            model="openai/text-embedding-3-large"
        )

        result = provider.embed("Hello world", seed=42)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)  # Normalized to SteadyText standard
        assert np.allclose(np.linalg.norm(result), 1.0, rtol=1e-5)

        # Verify API call
        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["model"] == "text-embedding-3-large"
        assert call_args.kwargs["input"] == ["Hello world"]

    @pytest.mark.slow
    def test_embedding_integration_with_steadytext(self, monkeypatch):
        """Test embedding integration with main SteadyText embed function.

        This simulates the quickstart.md example of similarity calculation.
        """
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        # This import should work after OpenRouter provider is implemented
        from steadytext import embed

        with patch("steadytext.providers.openrouter._get_openai") as mock_get_openai:
            # Mock OpenAI-compatible embeddings client
            mock_openai = Mock()
            mock_client = Mock()
            mock_response = Mock()

            # Create mock embeddings that are similar to each other
            base_embedding = np.random.randn(1536)
            mock_embedding1 = base_embedding + np.random.randn(1536) * 0.1
            mock_embedding2 = base_embedding + np.random.randn(1536) * 0.1

            mock_response.data = [
                Mock(embedding=mock_embedding1.tolist()),
                Mock(embedding=mock_embedding2.tolist()),
            ]

            mock_client.embeddings.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client
            mock_get_openai.return_value = mock_openai

            # Test the exact quickstart.md example
            embeddings = embed(
                ["Hello world", "Goodbye world"],
                model="openrouter:openai/text-embedding-3-small",
                unsafe_mode=True
            )

            # Should return single averaged embedding
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape == (1024,)

            # For batch embedding scenario, we would need separate calls
            # Let's also test individual embeddings for similarity calculation
            mock_client.embeddings.create.return_value = Mock(data=[Mock(embedding=mock_embedding1.tolist())])
            emb1 = embed("Hello world", model="openrouter:openai/text-embedding-3-small", unsafe_mode=True)

            mock_client.embeddings.create.return_value = Mock(data=[Mock(embedding=mock_embedding2.tolist())])
            emb2 = embed("Goodbye world", model="openrouter:openai/text-embedding-3-small", unsafe_mode=True)

            # Calculate similarity as in quickstart example
            similarity = np.dot(emb1, emb2)
            assert isinstance(similarity, (float, np.floating))
            assert -1.0 <= similarity <= 1.0  # Cosine similarity range

    @pytest.mark.slow
    @patch("steadytext.providers.openrouter._get_openai")
    def test_embedding_different_models(self, mock_get_openai, monkeypatch):
        """Test embedding with different OpenRouter embedding models."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        embedding_models = [
            "openai/text-embedding-3-small",
            "openai/text-embedding-3-large",
            "openai/text-embedding-ada-002",
            "voyage/voyage-2",
            "thenlper/gte-large"
        ]

        for model_name in embedding_models:
            # Mock OpenAI-compatible embeddings client
            mock_openai = Mock()
            mock_client = Mock()
            mock_response = Mock()

            # Different models might have different dimensions
            if "text-embedding-3-large" in model_name:
                embedding_dim = 3072
            elif "voyage" in model_name:
                embedding_dim = 1024
            else:
                embedding_dim = 1536

            mock_embedding = np.random.randn(embedding_dim).tolist()
            mock_response.data = [Mock(embedding=mock_embedding)]

            mock_client.embeddings.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client
            mock_get_openai.return_value = mock_openai

            provider = OpenRouterProvider(
                api_key="sk-or-test-key",
                model=model_name
            )

            result = provider.embed("Test text")

            assert isinstance(result, np.ndarray)
            assert result.shape == (1024,)  # All normalized to SteadyText standard

            # Verify correct model was used
            call_args = mock_client.embeddings.create.call_args
            # OpenRouter would strip the "openai/" prefix for OpenAI models
            expected_model = model_name.replace("openai/", "")
            assert call_args.kwargs["model"] == expected_model

    @pytest.mark.slow
    @patch("steadytext.providers.openrouter._get_openai")
    def test_embedding_error_handling(self, mock_get_openai, monkeypatch):
        """Test error handling in embedding generation."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock OpenAI-compatible client that raises an error
        mock_openai = Mock()
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("OpenRouter API Error")
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            model="openai/text-embedding-3-small"
        )

        # Should handle error gracefully and return None
        result = provider.embed("Test text")
        assert result is None

    def test_embedding_supported_models_list(self):
        """Test that provider returns list of supported embedding models."""
        provider = OpenRouterProvider()
        models = provider.get_supported_embedding_models()
        assert isinstance(models, list)

        # Should include models from quickstart
        expected_models = [
            "openai/text-embedding-3-small",
            "openai/text-embedding-3-large",
            "voyage/voyage-2",
            "thenlper/gte-large"
        ]
        for model in expected_models:
            assert model in models

    @pytest.mark.slow
    @patch("steadytext.providers.openrouter._get_openai")
    def test_batch_embedding_efficiency(self, mock_get_openai, monkeypatch):
        """Test efficient batch embedding processing."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock OpenAI-compatible embeddings client
        mock_openai = Mock()
        mock_client = Mock()
        mock_response = Mock()

        # Create multiple mock embeddings for batch processing
        texts = ["text1", "text2", "text3", "text4"]
        mock_embeddings = [np.random.randn(1536).tolist() for _ in texts]
        mock_response.data = [Mock(embedding=emb) for emb in mock_embeddings]

        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            model="openai/text-embedding-3-small"
        )

        result = provider.embed(texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)  # Averaged embedding

        # Should make only one API call for batch processing
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["input"] == texts

    @pytest.mark.slow
    @patch("steadytext.providers.openrouter._get_openai")
    def test_embedding_deterministic_behavior(self, mock_get_openai, monkeypatch):
        """Test that embedding generation is deterministic."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")

        # Mock OpenAI-compatible embeddings client
        mock_openai = Mock()
        mock_client = Mock()

        # Create deterministic mock embedding
        deterministic_embedding = [0.1] * 1536  # Simple deterministic values
        mock_response = Mock(data=[Mock(embedding=deterministic_embedding)])

        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        mock_get_openai.return_value = mock_openai

        provider = OpenRouterProvider(
            api_key="sk-or-test-key",
            model="openai/text-embedding-3-small"
        )

        # Generate embeddings twice with same input
        result1 = provider.embed("Deterministic test", seed=12345)
        result2 = provider.embed("Deterministic test", seed=12345)

        # Should produce identical results
        assert np.allclose(result1, result2)


class TestOpenRouterCLIEmbeddingScenarios:
    """Test CLI embedding scenarios from quickstart.md."""

    @pytest.mark.slow
    def test_cli_embedding_basic_command(self, monkeypatch):
        """Test basic CLI embedding command scenario.

        This represents: echo "Hello world" | st embed --model "openrouter:openai/text-embedding-3-small" --format json
        """
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        # This would be tested with CLI integration once implemented
        from steadytext.cli.commands.embed import embed_command

        # Placeholder for CLI embedding test
        pass

    @pytest.mark.slow
    def test_cli_embedding_multiple_files(self, monkeypatch):
        """Test CLI embedding with multiple input files.

        This represents: st embed --model "openrouter:openai/text-embedding-3-small" --input "text1.txt,text2.txt" --format numpy
        """
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        # Placeholder for CLI multiple file embedding test
        pass

    @pytest.mark.slow
    def test_cli_embedding_output_formats(self, monkeypatch):
        """Test different CLI embedding output formats."""
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        # Test different output formats: json, numpy, hex
        formats = ["json", "numpy", "hex"]
        for fmt in formats:
            # Would test: st embed --model "openrouter:openai/text-embedding-3-small" --format {fmt}
            pass


# AIDEV-NOTE: These embedding tests will fail initially because:
# 1. OpenRouterProvider class doesn't exist yet
# 2. embed method isn't implemented
# 3. get_supported_embedding_models method doesn't exist
# 4. OpenRouter embedding integration isn't complete
# 5. CLI embedding support needs to be added
# 6. This is expected and correct for TDD approach
# 7. Tests represent complete embedding scenarios from quickstart.md