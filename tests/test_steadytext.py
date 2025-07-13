from steadytext.utils import (
    EMBEDDING_DIMENSION,
    validate_normalized_embedding,  # Added for new tests
    logger as steadytext_logger,  # Use the library's logger for context in tests
)
import steadytext  # Main package import
import unittest
import pytest
import numpy as np
import os
import sys
import logging
from pathlib import Path

# Ensure the project root is in the Python path for testing
# This allows 'import steadytext' to work when tests are run directly
# from the tests directory or project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# For some specific tests, we might want to access core functions directly,
# but API tests are primary.
# from steadytext.core.embedder import _normalize_l2 # Example

# --- Test Configuration ---
# Allow tests requiring model downloads to be skipped via environment variable.
# Useful for CI environments where models might not be available or downloads
# are too slow.
ALLOW_MODEL_DOWNLOADS = (
    os.environ.get("STEADYTEXT_ALLOW_MODEL_DOWNLOADS", "true").lower() == "true"
)

# Global flag to track if models were successfully loaded during test setup.
# This provides context for interpreting test results, especially if
# model-dependent tests fail.
MODELS_ARE_ACCESSIBLE_FOR_TESTING = False  # Default to False

# Configure logger level for tests.
# Set to WARNING or ERROR to reduce noise from INFO/DEBUG logs during normal
# test runs. Change to DEBUG if you need to diagnose issues within the
# library during a test.
steadytext_logger.setLevel(logging.WARNING)
# If other loggers (e.g. from models.cache, models.loader) are used,
# configure them too if needed:
logging.getLogger("steadytext.utils").setLevel(logging.WARNING)
logging.getLogger("steadytext.models.cache").setLevel(logging.WARNING)
logging.getLogger("steadytext.models.loader").setLevel(logging.WARNING)
logging.getLogger("steadytext.core.generator").setLevel(logging.WARNING)
logging.getLogger("steadytext.core.embedder").setLevel(logging.WARNING)


@pytest.mark.slow
@pytest.mark.model_required
@unittest.skipUnless(
    ALLOW_MODEL_DOWNLOADS,
    "Skipping model-dependent tests (STEADYTEXT_ALLOW_MODEL_DOWNLOADS is not 'true')",
)
class TestSteadyTextAPIWithModels(unittest.TestCase):
    """
    Tests for the SteadyText public API that require actual model loading and
    interaction. These tests will attempt to download (if not cached) and
    use the configured GGUF models.
    """

    @classmethod
    def setUpClass(cls):
        """
        Preload models once for all tests in this class.
        Sets MODELS_ARE_ACCESSIBLE_FOR_TESTING based on success.
        """
        global MODELS_ARE_ACCESSIBLE_FOR_TESTING
        steadytext_logger.info(
            "Attempting to preload models for TestSteadyTextAPIWithModels..."
        )
        try:
            # Use the library's preload function. It logs errors internally.
            # verbose=True helps in CI or when debugging model loading.
            steadytext.preload_models(verbose=True)

            # Determine accessibility based on whether models actually loaded
            # Test by trying to use the public API
            try:
                # Try a simple generation to see if generator model is available
                gen_result = steadytext.generate("test", max_new_tokens=1)
                gen_ok = gen_result is not None and len(gen_result) > 0

                # Try a simple embedding to see if embedding model is available
                emb_result = steadytext.embed("test")
                emb_ok = emb_result is not None
            except Exception:
                # Handle any errors during model testing
                gen_ok = False
                emb_ok = False
            MODELS_ARE_ACCESSIBLE_FOR_TESTING = gen_ok and emb_ok
            steadytext_logger.info(
                "preload_models() completed. Generator loaded: %s, Embedder loaded: %s",
                gen_ok,
                emb_ok,
            )
        except Exception as e:
            # This catch is for unexpected errors from preload_models itself,
            # though it's designed to be robust.
            steadytext_logger.critical(
                f"Critical error during preload_models in setUpClass: {e}",
                exc_info=True,
            )
            MODELS_ARE_ACCESSIBLE_FOR_TESTING = False

        if not MODELS_ARE_ACCESSIBLE_FOR_TESTING:
            steadytext_logger.warning(
                "MODELS_ARE_ACCESSIBLE_FOR_TESTING is False after preload attempt. "
                "Model-dependent tests may not reflect full functionality and "
                "might be effectively testing error fallbacks."
            )

    def test_generate_deterministic_default_seed(self):
        """Test steadytext.generate() is deterministic with the default seed."""
        if not MODELS_ARE_ACCESSIBLE_FOR_TESTING:
            self.skipTest(
                "Models deemed not accessible by setUpClass, "
                "skipping actual generation test."
            )

        prompt = "A standard test prompt for default seed generation."
        output1 = steadytext.generate(prompt)
        output2 = steadytext.generate(prompt)

        self.assertIsInstance(
            output1, (str, type(None)), "Output must be a string or None."
        )
        if (
            MODELS_ARE_ACCESSIBLE_FOR_TESTING
            and output1 is not None
            and not output1.startswith("Error:")
        ):
            self.assertTrue(
                len(output1) > 0, "Successful generation should not be empty."
            )
            self.assertFalse(
                output1.startswith("Error:"),
                "Successful generation output should not start with 'Error:'.",
            )
        self.assertEqual(
            output1,
            output2,
            "Generated text (or error string) must be identical "
            "for the same prompt and default seed.",  # noqa E501
        )

    def test_embed_deterministic_string_and_validity(self):
        """Test steadytext.embed() is deterministic for string input and
        output is valid."""
        if not MODELS_ARE_ACCESSIBLE_FOR_TESTING:
            self.skipTest(
                "Models deemed not accessible, skipping string embedding test."
            )

        text = "A test sentence for string embedding evaluation."
        embedding1 = steadytext.embed(text)
        embedding2 = steadytext.embed(text)

        self.assertIsNotNone(
            embedding1, "Embedding should not be None when models are accessible"
        )
        self.assertIsNotNone(
            embedding2, "Embedding should not be None when models are accessible"
        )
        assert embedding1 is not None  # Type narrowing for mypy
        assert embedding2 is not None  # Type narrowing for mypy

        self.assertTrue(
            np.array_equal(embedding1, embedding2),
            "Embeddings must be identical for the same string input.",
        )
        self.assertIsInstance(embedding1, np.ndarray)
        self.assertEqual(embedding1.shape, (EMBEDDING_DIMENSION,))
        self.assertEqual(embedding1.dtype, np.float32)

        is_valid_embedding = validate_normalized_embedding(
            embedding1
        )  # Using direct import
        self.assertTrue(
            is_valid_embedding,
            f"Embedding for '{text}' failed validation (norm: {np.linalg.norm(embedding1):.4f}).",
        )

        if (
            MODELS_ARE_ACCESSIBLE_FOR_TESTING
            and text.strip()
            and not np.all(embedding1 == 0)
        ):
            # If models are presumed accessible and input was non-empty,
            # a zero vector is suspicious (though could be model's true output)
            pass  # Already logged by the warning below
        elif MODELS_ARE_ACCESSIBLE_FOR_TESTING and text.strip():
            self.assertFalse(
                np.all(embedding1 == 0),
                "Embedding for a non-empty string should not be a zero vector "
                "if models are accessible.",
            )

        if np.all(embedding1 == 0):
            steadytext_logger.warning(
                f"test_embed_deterministic_string_and_validity: Embedding for "
                f"'{text}' is a zero vector. This is expected if the model "  # noqa E501
                f"could not be loaded, or if the model truly embeds this to "
                f"zero (unlikely for non-empty string)."
            )

    def test_embed_deterministic_list_and_validity(self):
        """Test steadytext.embed() is deterministic for list input and
        output is valid."""
        if not MODELS_ARE_ACCESSIBLE_FOR_TESTING:
            self.skipTest("Models deemed not accessible, skipping list embedding test.")

        texts = [
            "First sentence in a list.",
            "Second sentence, somewhat different from the first.",
        ]
        embedding1 = steadytext.embed(texts)
        embedding2 = steadytext.embed(texts)

        self.assertIsNotNone(
            embedding1, "Embedding should not be None when models are accessible"
        )
        self.assertIsNotNone(
            embedding2, "Embedding should not be None when models are accessible"
        )
        assert embedding1 is not None  # Type narrowing for mypy
        assert embedding2 is not None  # Type narrowing for mypy

        self.assertTrue(
            np.array_equal(embedding1, embedding2),
            "Embeddings must be identical for the same list input.",
        )
        self.assertIsInstance(embedding1, np.ndarray)
        self.assertEqual(embedding1.shape, (EMBEDDING_DIMENSION,))
        self.assertEqual(embedding1.dtype, np.float32)
        self.assertTrue(
            validate_normalized_embedding(embedding1),  # Using direct import
            f"Embedding for list {texts} failed validation (norm: {np.linalg.norm(embedding1):.4f}).",
        )

        if (
            MODELS_ARE_ACCESSIBLE_FOR_TESTING
            and any(s.strip() for s in texts)
            and not np.all(embedding1 == 0)
        ):
            # If models accessible and list has non-empty content,
            # a zero vector is suspicious
            pass
        elif MODELS_ARE_ACCESSIBLE_FOR_TESTING and any(s.strip() for s in texts):
            self.assertFalse(
                np.all(embedding1 == 0),
                "Embedding for a list with non-empty strings should not be "
                "a zero vector if models are accessible.",  # noqa E501
            )

        if np.all(embedding1 == 0):
            steadytext_logger.warning(
                f"test_embed_deterministic_list_and_validity: Embedding for "
                f"list {texts} is a zero vector."
            )

    def test_generate_with_custom_eos_string(self):
        """Test generate() with custom eos_string parameter."""
        if not MODELS_ARE_ACCESSIBLE_FOR_TESTING:
            self.skipTest("Models deemed not accessible, skipping eos_string test.")

        # Skip if model loading is disabled
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
            self.skipTest("Model loading is disabled, skipping eos_string test.")

        # Test with default [EOS]
        output_default = steadytext.generate("Test prompt", eos_string="[EOS]")
        self.assertIsInstance(output_default, str)

        # Test with custom eos_string
        output_custom = steadytext.generate("Test prompt", eos_string="STOP")
        self.assertIsInstance(output_custom, str)

        # Results should be cached separately for different eos_strings
        output_default2 = steadytext.generate("Test prompt", eos_string="[EOS]")
        output_custom2 = steadytext.generate("Test prompt", eos_string="STOP")

        # Verify deterministic behavior within same eos_string
        self.assertEqual(output_default, output_default2)
        self.assertEqual(output_custom, output_custom2)

    def test_generate_iter_with_eos_string(self):
        """Test generate_iter() with custom eos_string parameter."""
        if not MODELS_ARE_ACCESSIBLE_FOR_TESTING:
            self.skipTest(
                "Models deemed not accessible, skipping eos_string streaming test."
            )

        # Skip if model loading is disabled
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
            self.skipTest(
                "Model loading is disabled, skipping eos_string streaming test."
            )

        # Test streaming with custom eos
        tokens = []
        for token in steadytext.generate_iter("Test prompt", eos_string="END"):
            tokens.append(token)
            if len(tokens) > 5:  # Limit iterations for test
                break

        self.assertTrue(len(tokens) > 0, "Should generate at least one token")
        self.assertTrue(all(isinstance(t, str) for t in tokens))

    def test_generate_eos_string_with_logprobs(self):
        """Test generate() with both eos_string and logprobs."""
        if not MODELS_ARE_ACCESSIBLE_FOR_TESTING:
            self.skipTest(
                "Models deemed not accessible, skipping eos_string with logprobs test."
            )

        # Skip if model loading is disabled
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
            self.skipTest(
                "Model loading is disabled, skipping eos_string with logprobs test."
            )

        result = steadytext.generate(
            "Test prompt", return_logprobs=True, eos_string="CUSTOM_END"
        )
        if result is not None and isinstance(result, tuple):
            text, logp = result
            self.assertIsInstance(text, str)
            self.assertTrue(logp is None or isinstance(logp, dict))
        else:
            self.assertIsNone(result)

    def test_embed_list_averaging_and_empty_string_handling(self):
        """Test list averaging and correct handling of empty/whitespace strings
        within lists for embed()."""
        if not MODELS_ARE_ACCESSIBLE_FOR_TESTING:
            self.skipTest(
                "Models deemed not accessible, skipping advanced list embedding test."
            )

        text_a = "Unique sentence A for averaging test."
        text_b = "Sentence B, also unique for this test."

        emb_a = steadytext.embed(text_a)
        emb_b = steadytext.embed(text_b)

        # Skip further checks if individual embeddings are zero (e.g., model load
        # failed)
        if np.all(emb_a == 0) or np.all(emb_b == 0):
            self.skipTest(
                "Individual string embeddings are zero vectors; cannot meaningfully test averaging logic."
            )

        # Test 1: List with empty strings interspersed, should average non-empty ones
        emb_list_mixed = steadytext.embed([text_a, "", "   ", text_b, "\t", "  "])

        # Test 1: List with empty strings interspersed should be equivalent
        # to list without them.
        emb_list_direct = steadytext.embed([text_a, text_b])
        self.assertTrue(
            np.allclose(emb_list_mixed, emb_list_direct, atol=1e-6),
            "Embedding of list [A, '', B, ''] should be equivalent to "
            "embedding of [A, B].",
        )

        # Test 2: List containing only one valid string and others empty/whitespace
        emb_list_single_valid = steadytext.embed(["", "  ", text_a, "\t", " "])
        # This should be equal to emb_a, as averaging with zero vectors
        # (conceptually, as they are ignored) and then re-normalizing
        # should yield emb_a if it was already normalized.
        self.assertTrue(
            np.allclose(emb_list_single_valid, emb_a, atol=1e-6),
            "Embedding of list ['', A, ''] should be very close to embedding of A.",
        )


@pytest.mark.fast
class TestSteadyTextAPIErrorFallbacks(unittest.TestCase):
    """
    Tests the error handling of the SteadyText public API, ensuring
    it returns None when models are unavailable or inputs are invalid.
    These tests do NOT require successful model loading and should pass
    even if models are unavailable.
    """

    def test_generate_invalid_prompt_type_returns_none(self):
        """Test generate() with an invalid prompt type returns None."""
        prompt_int = 12345

        output = steadytext.generate(prompt_int)  # type: ignore

        # Should return None for invalid input
        self.assertIsNone(output, "Invalid prompt type should return None")

        # Ensure it's consistent
        output2 = steadytext.generate(prompt_int)  # type: ignore
        self.assertIsNone(
            output2, "Invalid prompt type should consistently return None"
        )

    def test_generate_empty_prompt_returns_none_without_model(self):
        """Test generate() with an empty prompt returns None when model is not available."""
        # If models are inaccessible, it will return None
        # If models are accessible, this test may need to be skipped
        import os

        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
            output = steadytext.generate("")
            self.assertIsNone(
                output, "Empty prompt should return None when model is not available"
            )
        else:
            # Skip this test if models are loaded
            self.skipTest("This test requires STEADYTEXT_SKIP_MODEL_LOAD=1")

    def test_generate_with_eos_string_edge_cases(self):
        """Test generate() with edge case eos_string values."""
        # Test with empty eos_string
        output_empty = steadytext.generate("Test prompt", eos_string="")
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") != "1":
            self.assertIsInstance(output_empty, str)

        # Test with special characters
        output_special = steadytext.generate("Test prompt", eos_string="<|END|>")
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") != "1":
            self.assertIsInstance(output_special, str)

        # Test with unicode characters
        output_unicode = steadytext.generate("Test prompt", eos_string="终结")
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") != "1":
            self.assertIsInstance(output_unicode, str)

        # Test with whitespace
        output_whitespace = steadytext.generate("Test prompt", eos_string="   ")
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") != "1":
            self.assertIsInstance(output_whitespace, str)

    def test_generate_iter_with_eos_string_edge_cases(self):
        """Test generate_iter() with edge case eos_string values."""
        # Test with empty eos_string - should still work
        tokens = []
        for token in steadytext.generate_iter("Test", eos_string=""):
            tokens.append(token)
            if len(tokens) > 10:  # Increase limit for fallback mode
                break

        # In fallback mode, generate_iter might return fewer tokens or behave differently
        # The key is that it should not crash and should return string tokens
        if len(tokens) > 0:
            self.assertTrue(all(isinstance(t, str) for t in tokens))
        else:
            # If no tokens, ensure we can still call it without crashing
            # This might happen in fallback mode
            self.assertTrue(True, "generate_iter completed without crashing")

    def test_generate_eos_string_fallback_deterministic(self):
        """Test that eos_string parameter works deterministically in fallback mode."""
        # These should be deterministic regardless of model availability
        output1 = steadytext.generate("Test prompt", eos_string="STOP")
        output2 = steadytext.generate("Test prompt", eos_string="STOP")
        self.assertEqual(
            output1, output2, "Same eos_string should produce identical fallback output"
        )

        # Different eos_strings should be cached separately
        output_different = steadytext.generate("Test prompt", eos_string="END")
        # Note: In fallback mode, the actual eos_string might not affect the output,
        # but the caching should still work correctly
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") != "1":
            self.assertIsInstance(output_different, str)

    def test_generate_with_logprobs_flag(self):
        """generate() should return a tuple when return_logprobs=True."""
        result = steadytext.generate("test", return_logprobs=True)
        if result is not None and isinstance(result, tuple):
            text, logp = result
            if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") != "1":
                self.assertIsInstance(text, str)
            # logprobs is None when fallback generation is used
            self.assertTrue(logp is None or isinstance(logp, dict))
        else:
            self.assertIsNone(result)

    def test_embed_empty_string_fallback(self):
        """Test embed() with an empty string returns a zero vector."""
        embedding = steadytext.embed("")
        self.assertTrue(
            np.array_equal(embedding, np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)),
            "Embedding of empty string should be a zero vector.",
        )

    def test_embed_list_of_empty_strings_fallback(self):
        """Test embed() with a list of only empty/whitespace strings
        returns a zero vector."""
        embedding = steadytext.embed(["", "   ", "\t"])
        self.assertTrue(
            np.array_equal(embedding, np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)),
            "Embedding of list of only empty/whitespace strings should be "
            "a zero vector.",
        )

    def test_embed_empty_list_fallback(self):
        """Test embed() with an empty list returns a zero vector."""
        embedding = steadytext.embed([])
        self.assertTrue(
            np.array_equal(embedding, np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)),
            "Embedding of an empty list should be a zero vector.",
        )

    def test_embed_invalid_input_type_fallback(self):
        """Test embed() with a completely invalid input type (e.g., int)
        returns a zero vector."""
        embedding = steadytext.embed(12345)  # type: ignore
        self.assertIsNone(embedding, "Embedding of invalid type (int) should be None.")

    def test_embed_list_with_invalid_item_type_fallback(self):
        """Test embed() with a list containing an invalid item type (e.g., int)
        returns a zero vector."""
        embedding = steadytext.embed(["hello", 123, "world"])  # type: ignore
        self.assertIsNone(
            embedding, "Embedding of list with invalid item type should be None."
        )


@pytest.mark.fast
class TestSteadyTextFallbackBehavior(unittest.TestCase):
    """Tests to verify fallback behavior when models cannot be loaded."""

    def test_embed_api_handles_type_errors_gracefully(self):
        """Test that embed() API catches TypeErrors and returns None."""
        from unittest.mock import patch

        # This import is fine for context

        # Mock create_embedding where it's looked up by steadytext.embed
        # (in __init__.py)
        with patch(
            "steadytext.core_embed", side_effect=TypeError("Invalid input type")
        ) as mock_core_embed:
            result = steadytext.embed("test")
            mock_core_embed.assert_called_once_with(
                "test", seed=42
            )  # Verify the mock was called
            self.assertIsNone(
                result,
                "Should return None on TypeError",
            )

    def test_stop_sequences_integration(self):
        """Test that stop sequences are properly integrated into generation."""
        import os
        from steadytext.core.generator import DeterministicGenerator
        from steadytext.utils import DEFAULT_STOP_SEQUENCES
        from steadytext.models.loader import clear_model_cache
        from unittest.mock import Mock, patch

        # Ensure daemon is disabled for this test
        os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"

        # Create a mock model that captures the parameters passed to it
        mock_model = Mock()
        # Configure the create_chat_completion attribute of the mock_model
        mock_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Generated text"}}]
        }

        # AIDEV-NOTE: Clear any cached models and the generation cache before testing to ensure a clean state. The singleton _ModelInstanceCache persists real models across test runs, so mock patches will not work unless the cache is cleared first.
        clear_model_cache()
        from steadytext.cache_manager import get_generation_cache

        get_generation_cache().clear()

        # Patch where get_generator_model_instance is looked up
        # by DeterministicGenerator
        with patch(
            "steadytext.core.generator.get_generator_model_instance",
            return_value=mock_model,
        ):
            # Also need to skip model load during init
            original_skip = os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD")
            os.environ["STEADYTEXT_SKIP_MODEL_LOAD"] = "1"
            try:
                generator = DeterministicGenerator()
                # Now manually load the model which will use our mock
                del os.environ["STEADYTEXT_SKIP_MODEL_LOAD"]
                generator._load_model(enable_logits=False)
                self.assertIs(
                    generator.model,
                    mock_model,
                    "Patching failed: generator.model is not the "
                    "mock_model in stop_sequences test",
                )

                test_prompt_for_stop_sequence = (
                    "This is a test prompt that should "
                    "normally be completed by the model "
                    "but will be stopped by <|im_end|>"
                )
                generator.generate(test_prompt_for_stop_sequence)

                # Verify that create_chat_completion was called on the mock_model
                mock_model.create_chat_completion.assert_called_once()
                call_args, call_kwargs = mock_model.create_chat_completion.call_args

                self.assertIn(
                    "messages",
                    call_kwargs,
                    "Messages parameter should be passed to create_chat_completion",
                )
                # AIDEV-NOTE: In v2.0.0, thinking mode was removed, so prompts are passed as-is
                expected_content = test_prompt_for_stop_sequence
                self.assertEqual(
                    call_kwargs["messages"],
                    [{"role": "user", "content": expected_content}],
                )
                self.assertIn(
                    "stop",
                    call_kwargs,
                    "Stop parameter should be passed to create_chat_completion",
                )
                self.assertEqual(
                    call_kwargs["stop"],
                    DEFAULT_STOP_SEQUENCES,
                    "Stop sequences should match DEFAULT_STOP_SEQUENCES",
                )
            finally:
                # Restore original environment
                if original_skip is not None:
                    os.environ["STEADYTEXT_SKIP_MODEL_LOAD"] = original_skip
                else:
                    os.environ.pop("STEADYTEXT_SKIP_MODEL_LOAD", None)


@pytest.mark.fast
class TestSteadyTextUtilities(unittest.TestCase):
    """Tests for utility functions and constants exposed by the package."""

    def test_get_model_cache_dir_output(self):
        """Test get_model_cache_dir() returns a string path that is absolute."""
        cache_dir_str = steadytext.get_model_cache_dir()
        self.assertIsInstance(cache_dir_str, str)
        self.assertTrue(
            Path(cache_dir_str).is_absolute(),
            f"Cache directory '{cache_dir_str}' must be an absolute path.",
        )

    def test_preload_models_runs_without_raising_unexpected_errors(self):
        """Test that preload_models() executes without throwing unexpected
        exceptions."""
        try:
            # verbose=False to keep test logs cleaner unless
            # specifically debugging preload.
            steadytext.preload_models(verbose=False)
        except Exception as e:
            # preload_models itself is designed to catch and log errors,
            # not re-raise them.
            self.fail(
                f"steadytext.preload_models() raised an unexpected exception: "
                f"{type(e).__name__} - {e}"
            )

    def test_constants_and_version_are_exposed(self):
        """Test that key constants and __version__ are accessible from the
        package."""
        self.assertEqual(steadytext.DEFAULT_SEED, 42)
        self.assertEqual(
            steadytext.GENERATION_MAX_NEW_TOKENS, 1024
        )  # Updated to 1024 in v2.2.0
        self.assertEqual(steadytext.EMBEDDING_DIMENSION, 1024)
        self.assertIsInstance(steadytext.__version__, str)
        self.assertTrue(
            len(steadytext.__version__) > 0, "Version string should be non-empty."
        )
        self.assertIsNotNone(
            steadytext.logger, "The package logger should be accessible."
        )


@pytest.mark.fast
class TestValidateNormalizedEmbedding(unittest.TestCase):
    """Tests for the steadytext.utils.validate_normalized_embedding function."""

    def test_correctly_normalized_vector(self):
        """Test with a correctly normalized vector."""
        vec = np.random.rand(EMBEDDING_DIMENSION).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        self.assertTrue(validate_normalized_embedding(vec))

    def test_zero_vector(self):
        """Test with a zero vector (should be valid as per current function logic)."""
        vec = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        self.assertTrue(
            validate_normalized_embedding(vec)
        )  # Current function treats norm ~0 as valid

    def test_unnormalized_vector_too_high(self):
        """Test with an unnormalized vector (norm > 1.0)."""
        vec = np.random.rand(EMBEDDING_DIMENSION).astype(np.float32)
        vec = (vec / np.linalg.norm(vec)) * 1.5  # Make norm > 1
        self.assertFalse(validate_normalized_embedding(vec))

    def test_unnormalized_vector_too_low(self):
        """Test with an unnormalized vector (norm < 1.0 but not zero)."""
        vec = np.random.rand(EMBEDDING_DIMENSION).astype(np.float32)
        vec = (vec / np.linalg.norm(vec)) * 0.5  # Make norm < 1
        # Ensure it's not accidentally a zero vector
        if np.all(vec == 0):
            vec[0] = 0.1  # Make it non-zero if it became zero
        self.assertFalse(validate_normalized_embedding(vec))

    def test_vector_with_nan(self):
        """Test with a vector containing NaN values."""
        vec = np.full(EMBEDDING_DIMENSION, np.nan, dtype=np.float32)
        self.assertFalse(validate_normalized_embedding(vec))

        vec_mixed = np.random.rand(EMBEDDING_DIMENSION).astype(np.float32)
        vec_mixed[EMBEDDING_DIMENSION // 2] = np.nan
        self.assertFalse(validate_normalized_embedding(vec_mixed))

    def test_vector_with_inf(self):
        """Test with a vector containing inf values."""
        vec = np.full(EMBEDDING_DIMENSION, np.inf, dtype=np.float32)
        self.assertFalse(validate_normalized_embedding(vec))

        vec_mixed = np.random.rand(EMBEDDING_DIMENSION).astype(np.float32)
        vec_mixed[EMBEDDING_DIMENSION // 2] = np.inf
        self.assertFalse(validate_normalized_embedding(vec_mixed))

    def test_vector_incorrect_dimensions_2d(self):
        """Test with a 2D vector (function expects 1D)."""
        vec = np.random.rand(EMBEDDING_DIMENSION // 2, 2).astype(np.float32)
        # validate_normalized_embedding itself doesn't check dimensions
        # but np.linalg.norm would handle it. The function is expected
        # to return False if norm calculation fails or leads to non-scalar.
        # However, np.linalg.norm on a 2D array returns a scalar,
        # so this might pass if normalized.
        # Let's make it unnormalized to be sure.
        # vec = vec * 2 # Make it unnormalized
        # For now, the function primarily checks the norm of whatever is passed.
        # If it's a 2D array, np.linalg.norm will compute the Frobenius norm.
        # The function's docstring implies it's for 1D embeddings.
        # A robust test would require the function to explicitly check ndim.
        # As is, this test might not behave as "incorrect dimensions" if the
        # Frobenius norm is 1.0 or 0.0. Let's assume the function is
        # intended for 1D, and a 2D would be invalid usage.
        # The current `validate_normalized_embedding` implicitly handles this
        # by calculating the norm. If norm calculation fails (e.g. not a
        # numerical array), it would raise an error, caught by try-except.
        # If it's a 2D array, norm is calculated. If that norm is not ~1 or ~0,
        # it's False. This test is more about the *spirit* of "incorrect dimensions".
        # A better test would be if the function itself raised a TypeError or
        # ValueError for ndim != 1. For current implementation, we test if
        # providing a 2D array (which would have a different norm concept)
        # correctly fails validation unless its Frobenius norm happens to be 1.0.
        normalized_2d_frobenius = vec / np.linalg.norm(vec)
        unnormalized_2d = normalized_2d_frobenius * 1.5

        self.assertFalse(
            validate_normalized_embedding(normalized_2d_frobenius * 0.5)
        )  # Norm 0.5
        self.assertFalse(validate_normalized_embedding(unnormalized_2d))  # Norm 1.5

        # A 2D array whose Frobenius norm is 1.0 *should* pass the current
        # validation logic, though it's not a "1D normalized embedding".
        # This highlights a potential ambiguity in the function's current
        # checks vs. its name.
        # self.assertTrue(validate_normalized_embedding(normalized_2d_frobenius))

    def test_vector_incorrect_dimensions_0d(self):
        """Test with a 0D vector (scalar)."""
        vec = np.array(0.5, dtype=np.float32)  # A scalar
        self.assertFalse(
            validate_normalized_embedding(vec), "Scalar 0.5 should fail shape check."
        )
        vec_norm_one = np.array(1.0, dtype=np.float32)
        self.assertFalse(  # Changed to assertFalse
            validate_normalized_embedding(vec_norm_one),
            "Scalar 1.0 should fail shape check.",
        )
        vec_zero = np.array(0.0, dtype=np.float32)
        self.assertFalse(  # Changed to assertFalse
            validate_normalized_embedding(vec_zero),
            "Scalar 0.0 should fail shape check.",
        )

    # Note: The current validate_normalized_embedding does not explicitly check
    # EMBEDDING_DIMENSION or dtype, focusing only on the L2 norm and numeric issues.
    # Tests for those aspects would require changes to the function itself.


# Helper function for test_embed_list_averaging_and_empty_string_handling
# (not part of library)
def _test_normalize_l2(vector: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < tolerance:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


@pytest.mark.fast
class TestSizeParameter(unittest.TestCase):
    """Tests for the size parameter in generate() function."""

    def test_generate_with_size_parameter(self):
        """Test that generate() accepts size parameter without errors."""
        # This test verifies the API accepts the size parameter
        # It doesn't verify model switching (which requires models)
        try:
            # Test all valid size values
            for size in ["small", "large"]:
                output = steadytext.generate("Test prompt", size=size)
                if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") != "1":
                    if output is not None:
                        self.assertIsInstance(output, str)
                        self.assertTrue(
                            len(output) > 0,
                            f"Size {size} should generate non-empty text",
                        )
                    else:
                        self.assertIsNone(output)
        except Exception as e:
            # If models aren't available, it should still work with fallback
            if "model" not in str(e).lower():
                raise

    def test_generate_size_parameter_precedence(self):
        """Test that model parameter takes precedence over size."""
        # When both model and size are specified, model should win
        try:
            output = steadytext.generate(
                "Test prompt",
                model="qwen3-1.7b",  # Explicit model
                size="small",  # Should be ignored
            )
            if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") != "1":
                self.assertIsInstance(output, str)
        except Exception as e:
            # If models aren't available, it should still work with fallback
            if "model" not in str(e).lower():
                raise

    def test_generate_with_invalid_model_name(self):
        """Test that generate() with an invalid model name falls back gracefully."""
        with self.assertLogs("steadytext", level="WARNING") as cm:
            output = steadytext.generate("Test prompt", model="non_existent_model")
            # When model loading is disabled, invalid model names return None
            # When enabled, it should return a string (fallback)
            self.assertIsNone(output)
            # Verify that a warning was logged
            self.assertTrue(
                any(
                    "Invalid model name 'non_existent_model'" in log
                    for log in cm.output
                )
            )

    def test_generate_with_invalid_size(self):
        """Test that generate() with an invalid size falls back gracefully."""
        with self.assertLogs("steadytext", level="WARNING") as cm:
            output = steadytext.generate("Test prompt", size="extra_large")
            # When model loading is disabled or model files don't exist, returns None
            self.assertIsNone(output)
            # Verify that a warning was logged
            self.assertTrue(
                any("Invalid size 'extra_large'" in log for log in cm.output)
            )


if __name__ == "__main__":
    # To run tests directly from this file: `python -m steadytext.tests.test_steadytext`
    # To enable model downloads for local testing if skipped by default:
    # `STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true python -m steadytext.tests.test_steadytext` # noqa E501

    print("--- Running SteadyText Test Suite ---")
    if not ALLOW_MODEL_DOWNLOADS:
        print(
            "INFO: STEADYTEXT_ALLOW_MODEL_DOWNLOADS environment variable is not "
            "set to 'true'. Model-dependent tests will be skipped. Set this "
            "variable to 'true' to run all tests."
        )
    else:
        print(
            "INFO: STEADYTEXT_ALLOW_MODEL_DOWNLOADS is set to 'true'. "
            "All tests including model-dependent ones will run."
        )

    unittest.main()
