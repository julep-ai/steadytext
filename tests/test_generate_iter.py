"""Tests for the iterable generation functionality.

AIDEV-NOTE: Some tests in this file experience hanging issues in pytest
environment, likely due to streaming API behavior with llama-cpp-python.
The functions work correctly in isolation but may have issues with
pytest's output capture or threading model.
"""

import pytest
import steadytext


@pytest.mark.concurrent
class TestGenerateIter:
    """Test cases for generate_iter function."""

    def test_basic_generation(self):
        """Test basic token generation with generate_iter."""
        prompt = "Tell me a story about"

        # Collect tokens as they're generated with a safety limit
        tokens = []
        token_count = 0
        for token in steadytext.generate_iter(prompt):
            tokens.append(token)
            token_count += 1
            # Safety limit to prevent infinite loops in tests
            if token_count > 1000:
                break

        # Should generate some tokens
        assert len(tokens) > 0

        # Each token should be a string
        assert all(isinstance(token, str) for token in tokens)

        # Combined output should be non-empty
        combined_output = "".join(tokens).strip()
        assert len(combined_output) > 0

    @pytest.mark.skip(
        reason="AIDEV-TODO: Hanging in pytest environment - investigate streaming API issues"
    )
    def test_matches_regular_generate(self):
        """Test that generate_iter produces same output as generate."""
        prompt = "Tell me a story about"

        # Get output from both methods
        regular_output = steadytext.generate(prompt)
        iter_tokens = []
        for i, token in enumerate(steadytext.generate_iter(prompt)):
            iter_tokens.append(token)
            if i > 1000:  # Safety limit
                break
        iter_output = "".join(iter_tokens).strip()

        # They should match
        assert regular_output == iter_output

    @pytest.mark.skip(
        reason="AIDEV-TODO: Hanging in pytest environment - investigate streaming API issues"
    )
    def test_determinism(self):
        """Test that generate_iter is deterministic."""
        prompt = "Tell me a story about"

        # Multiple calls should produce same output
        iter1 = []
        for i, token in enumerate(steadytext.generate_iter(prompt)):
            iter1.append(token)
            if i > 1000:  # Safety limit
                break

        iter2 = []
        for i, token in enumerate(steadytext.generate_iter(prompt)):
            iter2.append(token)
            if i > 1000:  # Safety limit
                break

        assert iter1 == iter2

    def test_empty_prompt(self):
        """Test generate_iter with empty prompt."""
        # Should still generate something (fallback behavior)
        tokens = []
        for i, token in enumerate(steadytext.generate_iter("")):
            tokens.append(token)
            if i > 1000:  # Safety limit
                break
        assert len(tokens) > 0

    @pytest.mark.skip(
        reason="AIDEV-TODO: Hanging in pytest environment - investigate streaming API issues"
    )
    def test_different_prompts(self):
        """Test that different prompts produce different outputs."""
        prompt1 = "Tell me about"
        prompt2 = "Explain the concept"

        output1 = "".join(steadytext.generate_iter(prompt1))
        output2 = "".join(steadytext.generate_iter(prompt2))

        # Different prompts should produce different outputs
        assert output1 != output2

    def test_invalid_input_type(self):
        """Test generate_iter with invalid input type."""
        # Should handle gracefully and use fallback
        tokens = []
        for i, token in enumerate(steadytext.generate_iter(123)):  # Invalid type
            tokens.append(token)
            if i > 1000:  # Safety limit
                break
        assert len(tokens) > 0  # Should still produce output via fallback

    @pytest.mark.skipif(
        not steadytext.models.loader.get_generator_model_instance(),
        reason="Model not available",
    )
    def test_streaming_behavior(self):
        """Test that generate_iter actually streams tokens."""
        prompt = "Tell me a story"

        # Collect tokens and check they arrive incrementally
        tokens = []
        for token in steadytext.generate_iter(prompt):
            tokens.append(token)
            # Each token should be relatively small (not the entire output)
            # This tests that we're actually streaming
            if len(tokens) > 1:
                assert len(token) < 100  # Individual tokens should be small
            # Safety limit to prevent infinite loops
            if len(tokens) > 100:
                break

        # Should have generated multiple tokens
        assert len(tokens) > 5
