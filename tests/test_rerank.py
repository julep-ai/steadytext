# AIDEV-NOTE: Tests for reranking functionality
import os
from unittest.mock import Mock, patch
from steadytext import rerank
from steadytext.core.reranker import (
    DeterministicReranker,
    _fallback_rerank_score,
)


class TestReranking:
    """Test suite for reranking functionality."""

    def test_basic_rerank_with_scores(self):
        """Test basic reranking with scores returned."""
        query = "What is Python?"
        documents = [
            "Python is a programming language",
            "Snakes are reptiles",
            "Java is also a programming language",
        ]

        # Since model might not be available in tests, we test the API
        with patch("steadytext.core.reranker.get_reranker") as mock_get_reranker:
            mock_instance = Mock()
            mock_instance.rerank.return_value = [
                ("Python is a programming language", 0.95),
                ("Java is also a programming language", 0.3),
                ("Snakes are reptiles", 0.1),
            ]
            mock_get_reranker.return_value = mock_instance

            results = rerank(query, documents, return_scores=True)

            assert len(results) == 3
            assert results[0][0] == "Python is a programming language"
            assert results[0][1] == 0.95

    def test_rerank_without_scores(self):
        """Test reranking returning only documents."""
        query = "climate change"
        documents = [
            "Global warming is real",
            "Pizza is delicious",
            "Climate change affects weather patterns",
        ]

        with patch("steadytext.core.reranker.get_reranker") as mock_get_reranker:
            mock_instance = Mock()

            # Mock the rerank method to handle return_scores parameter correctly
            def mock_rerank(*args, **kwargs):
                results = [
                    ("Climate change affects weather patterns", 0.8),
                    ("Global warming is real", 0.6),
                    ("Pizza is delicious", 0.1),
                ]
                if kwargs.get("return_scores", True):
                    return results
                else:
                    return [doc for doc, _ in results]

            mock_instance.rerank = mock_rerank
            mock_get_reranker.return_value = mock_instance

            results = rerank(query, documents, return_scores=False)

            assert len(results) == 3
            assert results[0] == "Climate change affects weather patterns"
            assert isinstance(results, list)

    def test_fallback_rerank_score(self):
        """Test the fallback scoring function."""
        # Test with overlapping words
        score = _fallback_rerank_score(
            "python programming", "Python is a programming language"
        )
        assert score > 0.5  # Should have high overlap

        # Test with no overlap
        score = _fallback_rerank_score("climate change", "Pizza is delicious")
        assert score == 0.0

        # Test with empty query
        score = _fallback_rerank_score("", "Some document")
        assert score == 0.0

    def test_rerank_empty_documents(self):
        """Test reranking with empty document list."""
        with patch("steadytext.core.reranker.get_reranker") as mock_get_reranker:
            mock_instance = Mock()
            # Should return empty list for empty input
            mock_instance.rerank.return_value = []
            mock_get_reranker.return_value = mock_instance

            results = rerank("test query", [])
            assert results == []

    def test_rerank_single_document(self):
        """Test reranking with single document."""
        query = "test"
        document = "This is a test document"

        with patch("steadytext.core.reranker.get_reranker") as mock_get_reranker:
            mock_instance = Mock()
            mock_instance.rerank.return_value = [(document, 1.0)]
            mock_get_reranker.return_value = mock_instance

            results = rerank(query, document, return_scores=True)

            assert len(results) == 1
            assert results[0][0] == document

    def test_rerank_custom_task(self):
        """Test reranking with custom task description."""
        query = "fever symptoms"
        documents = ["Patient has high temperature", "Weather is hot today"]
        task = "Find medical information relevant to the symptom query"

        with patch("steadytext.core.reranker.get_reranker") as mock_get_reranker:
            mock_instance = Mock()
            mock_instance.rerank.return_value = [
                ("Patient has high temperature", 0.9),
                ("Weather is hot today", 0.2),
            ]
            mock_get_reranker.return_value = mock_instance

            rerank(query, documents, task=task, return_scores=True)

            # Check that task was passed to rerank method
            mock_instance.rerank.assert_called_with(
                query=query, documents=documents, task=task, return_scores=True, seed=42
            )

    def test_rerank_determinism(self):
        """Test that reranking is deterministic with same seed."""
        query = "test query"
        documents = ["doc1", "doc2", "doc3"]

        # Mock to return consistent results
        with patch("steadytext.core.reranker.get_reranker") as mock_get_reranker:
            mock_instance = Mock()
            mock_instance.rerank.return_value = [
                ("doc1", 0.8),
                ("doc2", 0.6),
                ("doc3", 0.4),
            ]
            mock_get_reranker.return_value = mock_instance

            # Call multiple times with same seed
            results1 = rerank(query, documents, seed=42)
            results2 = rerank(query, documents, seed=42)

            assert results1 == results2


class TestDeterministicReranker:
    """Test the DeterministicReranker class directly."""

    def test_reranker_initialization(self):
        """Test reranker initialization."""
        # Ensure STEADYTEXT_SKIP_MODEL_LOAD is not set
        original_skip = os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD")
        if original_skip:
            del os.environ["STEADYTEXT_SKIP_MODEL_LOAD"]

        try:
            with patch(
                "steadytext.core.reranker.get_generator_model_instance"
            ) as mock_get_model:
                mock_model = Mock()

                # Mock tokenize method to return different token IDs for yes and no
                def tokenize_side_effect(text, add_bos=False):
                    if b"yes" in text:
                        return [123]
                    elif b"no" in text:
                        return [456]
                    return []

                mock_model.tokenize.side_effect = tokenize_side_effect
                mock_get_model.return_value = mock_model

                reranker = DeterministicReranker()

                assert reranker.model is not None
                assert reranker._yes_token_id == 123
                assert reranker._no_token_id == 456
        finally:
            # Restore original environment variable
            if original_skip:
                os.environ["STEADYTEXT_SKIP_MODEL_LOAD"] = original_skip

    def test_reranker_fallback(self):
        """Test reranker fallback when model unavailable."""
        with patch(
            "steadytext.core.reranker.get_generator_model_instance", return_value=None
        ):
            reranker = DeterministicReranker()

            results = reranker.rerank(
                "python programming",
                ["Python is great", "Java is okay", "Pizza recipe"],
                return_scores=True,
            )

            # Should use fallback scoring
            assert len(results) == 3
            # Python document should score highest due to word overlap
            assert results[0][0] == "Python is great"
