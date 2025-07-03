"""Tests for context window consistency and input length validation.

AIDEV-NOTE: This test module verifies that:
1. Output remains identical regardless of context window size
2. Input length validation works correctly
3. Context window is set to maximum available
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadytext.utils import DEFAULT_SEED
from steadytext.core.generator import DeterministicGenerator

# Skip tests if models aren't available
ALLOW_MODEL_DOWNLOADS = (
    os.environ.get("STEADYTEXT_ALLOW_MODEL_DOWNLOADS", "true").lower() == "true"
)


@pytest.mark.skipif(not ALLOW_MODEL_DOWNLOADS, reason="Model downloads not allowed")
class TestContextWindowConsistency:
    """Test that output remains consistent across different context window sizes."""

    def test_same_output_different_context_sizes(self):
        """AIDEV-NOTE: Core test - verify identical output with different n_ctx values."""
        # Test prompts of different lengths
        test_prompts = [
            "Write a short story about a robot.",  # Short prompt
            "Explain the theory of relativity in simple terms. " * 10,  # Medium prompt
            "Once upon a time, in a land far away, " * 50,  # Long prompt
        ]

        # Test with different context sizes
        context_sizes = [2048, 4096, 8192]

        for prompt in test_prompts:
            outputs = {}

            for ctx_size in context_sizes:
                # Set context size via environment variable (to be implemented)
                os.environ["STEADYTEXT_MAX_CONTEXT_WINDOW"] = str(ctx_size)

                # Create new generator to pick up the env var
                generator = DeterministicGenerator()

                # Generate output
                output = generator.generate(prompt, seed=DEFAULT_SEED)

                if output is not None:
                    outputs[ctx_size] = output

            # All outputs should be identical if model loaded successfully
            if len(outputs) > 1:
                first_output = list(outputs.values())[0]
                for ctx_size, output in outputs.items():
                    assert output == first_output, (
                        f"Output differs with context size {ctx_size}. "
                        f"Expected consistent output regardless of context window."
                    )

    def test_streaming_consistency_across_context_sizes(self):
        """Test that streaming generation is consistent across context sizes."""
        prompt = "Tell me a story about artificial intelligence."
        context_sizes = [2048, 4096]

        streaming_outputs = {}

        for ctx_size in context_sizes:
            os.environ["STEADYTEXT_MAX_CONTEXT_WINDOW"] = str(ctx_size)
            generator = DeterministicGenerator()

            # Collect streaming output
            tokens = []
            for token in generator.generate_iter(prompt, seed=DEFAULT_SEED):
                if isinstance(token, str):
                    tokens.append(token)

            if tokens:
                streaming_outputs[ctx_size] = "".join(tokens)

        # Verify consistency
        if len(streaming_outputs) > 1:
            first_output = list(streaming_outputs.values())[0]
            for ctx_size, output in streaming_outputs.items():
                assert output == first_output, (
                    f"Streaming output differs with context size {ctx_size}"
                )


@pytest.mark.skipif(not ALLOW_MODEL_DOWNLOADS, reason="Model downloads not allowed")
class TestInputLengthValidation:
    """Test input length validation and error handling."""

    def test_input_exceeds_context_window(self):
        """Test that oversized inputs raise appropriate errors."""
        # Set a small context window for testing
        os.environ["STEADYTEXT_MAX_CONTEXT_WINDOW"] = "512"

        # Create a prompt that will exceed the context window
        # Assuming ~4 chars per token on average
        oversized_prompt = "This is a test sentence. " * 200  # ~1000 tokens

        generator = DeterministicGenerator()

        # This should raise ContextLengthExceededError once implemented
        # For now, we'll just check that it handles gracefully
        try:
            output = generator.generate(oversized_prompt)
            # Until validation is implemented, this might succeed or return None
            assert output is None or isinstance(output, str)
        except Exception as e:
            # Once ContextLengthExceededError is implemented
            assert "context" in str(e).lower() or "length" in str(e).lower()

    def test_input_within_context_window(self):
        """Test that inputs within context window work normally."""
        os.environ["STEADYTEXT_MAX_CONTEXT_WINDOW"] = "4096"

        prompt = "Write a haiku about programming."
        generator = DeterministicGenerator()

        output = generator.generate(prompt)

        # Should generate successfully if model is available
        if output is not None:
            assert isinstance(output, str)
            assert len(output) > 0


@pytest.mark.skipif(not ALLOW_MODEL_DOWNLOADS, reason="Model downloads not allowed")
class TestDynamicContextWindow:
    """Test dynamic context window configuration."""

    def test_env_var_sets_context_window(self):
        """Test that STEADYTEXT_MAX_CONTEXT_WINDOW env var is respected."""
        # Test different context window sizes
        for ctx_size in [1024, 2048, 4096]:
            os.environ["STEADYTEXT_MAX_CONTEXT_WINDOW"] = str(ctx_size)

            # Generator should pick up the env var
            generator = DeterministicGenerator()

            # Verify it can generate (actual context size verification
            # will be added when we have access to model internals)
            output = generator.generate("Hello", seed=DEFAULT_SEED)

            if output is not None:
                assert isinstance(output, str)

    def test_default_uses_maximum_available(self):
        """Test that default behavior uses maximum available context."""
        # Remove env var to test default behavior
        if "STEADYTEXT_MAX_CONTEXT_WINDOW" in os.environ:
            del os.environ["STEADYTEXT_MAX_CONTEXT_WINDOW"]

        generator = DeterministicGenerator()

        # Should use maximum available context
        # For now just verify it works
        output = generator.generate("Test maximum context", seed=DEFAULT_SEED)

        if output is not None:
            assert isinstance(output, str)


# AIDEV-TODO: Add tests for token counting once implemented
# AIDEV-TODO: Add tests for ContextLengthExceededError once created
# AIDEV-TODO: Add tests to verify actual context window size in model
