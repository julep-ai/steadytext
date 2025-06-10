import unittest
import numpy as np
import os
import sys
import logging
from pathlib import Path

# Ensure the src directory is in the Python path for testing
# This allows 'import steadytext' to work when tests are run directly from the tests directory or project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import steadytext # Main package import
from steadytext.utils import (
    DEFAULT_SEED,
    GENERATION_MAX_NEW_TOKENS,
    EMBEDDING_DIMENSION,
    logger as steadytext_logger # Use the library's logger for context in tests
)
# For some specific tests, we might want to access core functions directly, but API tests are primary.
# from steadytext.core.embedder import _normalize_l2 # Example if needed for a specific test setup

# --- Test Configuration ---
# Allow tests requiring model downloads to be skipped via environment variable.
# Useful for CI environments where models might not be available or downloads are too slow.
ALLOW_MODEL_DOWNLOADS = os.environ.get("STEADYTEXT_ALLOW_MODEL_DOWNLOADS", "true").lower() == "true"

# Global flag to track if models were successfully loaded during test setup.
# This provides context for interpreting test results, especially if model-dependent tests fail.
MODELS_ARE_ACCESSIBLE_FOR_TESTING = False # Default to False

# Configure logger level for tests.
# Set to WARNING or ERROR to reduce noise from INFO/DEBUG logs during normal test runs.
# Change to DEBUG if you need to diagnose issues within the library during a test.
steadytext_logger.setLevel(logging.WARNING)
# If other loggers (e.g. from models.cache, models.loader) are used, configure them too if needed:
logging.getLogger("steadytext.utils").setLevel(logging.WARNING)
logging.getLogger("steadytext.models.cache").setLevel(logging.WARNING)
logging.getLogger("steadytext.models.loader").setLevel(logging.WARNING)
logging.getLogger("steadytext.core.generator").setLevel(logging.WARNING)
logging.getLogger("steadytext.core.embedder").setLevel(logging.WARNING)


@unittest.skipUnless(ALLOW_MODEL_DOWNLOADS, "Skipping model-dependent tests (STEADYTEXT_ALLOW_MODEL_DOWNLOADS is not 'true')")
class TestSteadyTextAPIWithModels(unittest.TestCase):
    """
    Tests for the SteadyText public API that require actual model loading and interaction.
    These tests will attempt to download (if not cached) and use the configured GGUF models.
    """

    @classmethod
    def setUpClass(cls):
        """
        Preload models once for all tests in this class.
        Sets MODELS_ARE_ACCESSIBLE_FOR_TESTING based on success.
        """
        global MODELS_ARE_ACCESSIBLE_FOR_TESTING
        steadytext_logger.info("Attempting to preload models for TestSteadyTextAPIWithModels...")
        try:
            # Use the library's preload function. It logs errors internally.
            # verbose=True helps in CI or when debugging model loading.
            steadytext.preload_models(verbose=True)

            # A simple check: if preload_models didn't raise an error, assume models might be usable.
            # More specific checks could involve trying to get model instances, but that repeats preload logic.
            # The individual tests will ultimately determine if the models function as expected.
            MODELS_ARE_ACCESSIBLE_FOR_TESTING = True
            steadytext_logger.info("preload_models() completed. Assuming models may be accessible.")
        except Exception as e:
            # This catch is for unexpected errors from preload_models itself, though it's designed to be robust.
            steadytext_logger.critical(f"Critical error during preload_models in setUpClass: {e}", exc_info=True)
            MODELS_ARE_ACCESSIBLE_FOR_TESTING = False

        if not MODELS_ARE_ACCESSIBLE_FOR_TESTING:
            steadytext_logger.warning("MODELS_ARE_ACCESSIBLE_FOR_TESTING is False after preload attempt. Model-dependent tests may not reflect full functionality and might be effectively testing error fallbacks.")


    def test_generate_deterministic_default_seed(self):
        """Test steadytext.generate() is deterministic with the default seed."""
        if not MODELS_ARE_ACCESSIBLE_FOR_TESTING:
             self.skipTest("Models deemed not accessible by setUpClass, skipping actual generation test.")

        prompt = "A standard test prompt for default seed generation."
        output1 = steadytext.generate(prompt)
        output2 = steadytext.generate(prompt)

        self.assertIsInstance(output1, str, "Output must be a string.")
        # If model loading failed, output1 will be an error string.
        if not output1.startswith("Error:"):
            self.assertTrue(len(output1) > 0, "Successful generation should not be empty.")
        self.assertEqual(output1, output2, "Generated text (or error string) must be identical for the same prompt and default seed.")

    def test_generate_deterministic_custom_seed(self):
        """Test steadytext.generate() is deterministic with a custom seed and output varies from default seed."""
        if not MODELS_ARE_ACCESSIBLE_FOR_TESTING:
            self.skipTest("Models deemed not accessible, skipping custom seed generation test.")

        prompt = "Another unique test prompt for custom seed evaluation."
        custom_seed = DEFAULT_SEED + 42 # A seed different from default

        output_default = steadytext.generate(prompt, seed=DEFAULT_SEED)
        output_custom1 = steadytext.generate(prompt, seed=custom_seed)
        output_custom2 = steadytext.generate(prompt, seed=custom_seed)

        self.assertIsInstance(output_custom1, str)
        self.assertEqual(output_custom1, output_custom2, "Generated text (or error string) must be identical for the same custom seed.")

        # Only compare content if both generations were successful (not error strings)
        if not output_default.startswith("Error:") and not output_custom1.startswith("Error:"):
            self.assertNotEqual(output_default, output_custom1, "Generated text should differ for different seeds when generation is successful.")
            self.assertTrue(len(output_custom1) > 0, "Successful custom seed generation should not be empty.")
        elif output_default.startswith("Error:") and output_custom1.startswith("Error:"):
            # Error messages include seed, so they should differ if seeds are different.
             self.assertNotEqual(output_default, output_custom1, "Error messages should differ if seeds (part of error message) are different.")
        else:
            steadytext_logger.warning("One generation attempt resulted in an error, skipping content comparison for seed variation.")


    def test_embed_deterministic_string_and_validity(self):
        """Test steadytext.embed() is deterministic for string input and output is valid."""
        if not MODELS_ARE_ACCESSIBLE_FOR_TESTING:
            self.skipTest("Models deemed not accessible, skipping string embedding test.")

        text = "A test sentence for string embedding evaluation."
        embedding1 = steadytext.embed(text)
        embedding2 = steadytext.embed(text)

        self.assertTrue(np.array_equal(embedding1, embedding2), "Embeddings must be identical for the same string input.")
        self.assertIsInstance(embedding1, np.ndarray)
        self.assertEqual(embedding1.shape, (EMBEDDING_DIMENSION,))
        self.assertEqual(embedding1.dtype, np.float32)

        # validate_normalized_embedding allows zero vectors (norm 0.0)
        is_valid_embedding = steadytext.validate_normalized_embedding(embedding1)
        self.assertTrue(is_valid_embedding, f"Embedding for '{text}' failed validation (norm: {np.linalg.norm(embedding1):.4f}).")

        if np.all(embedding1 == 0):
            steadytext_logger.warning(f"test_embed_deterministic_string_and_validity: Embedding for '{text}' is a zero vector. This is expected if the model could not be loaded, or if the model truly embeds this to zero (unlikely for non-empty string).")


    def test_embed_deterministic_list_and_validity(self):
        """Test steadytext.embed() is deterministic for list input and output is valid."""
        if not MODELS_ARE_ACCESSIBLE_FOR_TESTING:
            self.skipTest("Models deemed not accessible, skipping list embedding test.")

        texts = ["First sentence in a list.", "Second sentence, somewhat different from the first."]
        embedding1 = steadytext.embed(texts)
        embedding2 = steadytext.embed(texts)

        self.assertTrue(np.array_equal(embedding1, embedding2), "Embeddings must be identical for the same list input.")
        self.assertIsInstance(embedding1, np.ndarray)
        self.assertEqual(embedding1.shape, (EMBEDDING_DIMENSION,))
        self.assertEqual(embedding1.dtype, np.float32)
        self.assertTrue(steadytext.validate_normalized_embedding(embedding1),
                        f"Embedding for list {texts} failed validation (norm: {np.linalg.norm(embedding1):.4f}).")
        if np.all(embedding1 == 0):
             steadytext_logger.warning(f"test_embed_deterministic_list_and_validity: Embedding for list {texts} is a zero vector.")


    def test_embed_list_averaging_and_empty_string_handling(self):
        """Test list averaging and correct handling of empty/whitespace strings within lists for embed()."""
        if not MODELS_ARE_ACCESSIBLE_FOR_TESTING:
            self.skipTest("Models deemed not accessible, skipping advanced list embedding test.")

        text_a = "Unique sentence A for averaging test."
        text_b = "Sentence B, also unique for this test."

        emb_a = steadytext.embed(text_a)
        emb_b = steadytext.embed(text_b)

        # Skip further checks if individual embeddings are zero (e.g., model load failed)
        if np.all(emb_a == 0) or np.all(emb_b == 0):
            self.skipTest("Individual string embeddings are zero vectors; cannot meaningfully test averaging logic.")

        # Test 1: List with empty strings interspersed, should average non-empty ones
        emb_list_mixed = steadytext.embed([text_a, "", "   ", text_b, "\t", "  "])

        # Manual calculation of expected average of A and B (assuming _normalize_l2 from core.embedder)
        # This helper is defined at the end of this test file for test purposes only.
        expected_avg_emb = _test_normalize_l2((emb_a + emb_b) / 2.0)
        self.assertTrue(np.allclose(emb_list_mixed, expected_avg_emb, atol=1e-6),
                        "Embedding of list [A, '', B, ''] should be close to normalized average of A and B.")

        # Test 2: List containing only one valid string and others empty/whitespace
        emb_list_single_valid = steadytext.embed(["", "  ", text_a, "\t", " "])
        # This should be equal to emb_a, as averaging with zero vectors (conceptually, as they are ignored)
        # and then re-normalizing should yield emb_a if it was already normalized.
        self.assertTrue(np.allclose(emb_list_single_valid, emb_a, atol=1e-6),
                        "Embedding of list ['', A, ''] should be very close to embedding of A.")


class TestSteadyTextAPIErrorFallbacks(unittest.TestCase):
    """
    Tests the "Never Fails" aspect of the SteadyText public API, ensuring graceful error handling
    and deterministic fallback outputs (error strings for generate, zero vectors for embed).
    These tests do NOT require successful model loading and should pass even if models are unavailable.
    """

    def test_generate_invalid_prompt_type_fallback(self):
        """Test generate() with an invalid prompt type returns a deterministic error string."""
        prompt_int = 12345
        output = steadytext.generate(prompt_int) # type: ignore
        self.assertIsInstance(output, str)
        self.assertTrue(output.startswith("Error: Invalid prompt type."), f"Output '{output}' not an expected error string.")
        self.assertIn(f"(Input type: int, Seed: {DEFAULT_SEED})", output)

        # Ensure it's deterministic
        output2 = steadytext.generate(prompt_int, seed=DEFAULT_SEED) # type: ignore
        output3_diff_seed = steadytext.generate(prompt_int, seed=DEFAULT_SEED+1) # type: ignore
        self.assertEqual(output, output2, "Error string for same invalid input and seed should be deterministic.")
        self.assertNotEqual(output, output3_diff_seed, "Error string should reflect change in seed if seed is part of it.")


    def test_generate_empty_prompt_fallback(self):
        """Test generate() with an empty prompt; should not be 'Invalid prompt type' error."""
        # The core generator now uses a space " " for empty prompts.
        # If models are inaccessible, it will return a model loading error.
        # If models are accessible, it will generate text from " ".
        output = steadytext.generate("")
        self.assertIsInstance(output, str)
        self.assertTrue(len(output) > 0, "Output for empty prompt should not itself be empty.")
        self.assertFalse(output.startswith("Error: Invalid prompt type."), "Empty string is a valid type, should not be 'Invalid prompt type' error.")
        # If models can't load, it will be an error like "Error: Text generation model unavailable..."
        # If models load, it will be actual text. This test just ensures it's not an *invalid type* error.


    def test_embed_empty_string_fallback(self):
        """Test embed() with an empty string returns a zero vector."""
        embedding = steadytext.embed("")
        self.assertTrue(np.array_equal(embedding, np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)),
                        "Embedding of empty string should be a zero vector.")

    def test_embed_list_of_empty_strings_fallback(self):
        """Test embed() with a list of only empty/whitespace strings returns a zero vector."""
        embedding = steadytext.embed(["", "   ", "\t"])
        self.assertTrue(np.array_equal(embedding, np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)),
                        "Embedding of list of only empty/whitespace strings should be a zero vector.")

    def test_embed_empty_list_fallback(self):
        """Test embed() with an empty list returns a zero vector."""
        embedding = steadytext.embed([])
        self.assertTrue(np.array_equal(embedding, np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)),
                        "Embedding of an empty list should be a zero vector.")

    def test_embed_invalid_input_type_fallback(self):
        """Test embed() with a completely invalid input type (e.g., int) returns a zero vector."""
        embedding = steadytext.embed(12345) # type: ignore
        self.assertTrue(np.array_equal(embedding, np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)),
                        "Embedding of invalid type (int) should be a zero vector.")

    def test_embed_list_with_invalid_item_type_fallback(self):
        """Test embed() with a list containing an invalid item type (e.g., int) returns a zero vector."""
        embedding = steadytext.embed(["hello", 123, "world"]) # type: ignore
        self.assertTrue(np.array_equal(embedding, np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)),
                        "Embedding of list with invalid item type should be a zero vector.")


class TestSteadyTextUtilities(unittest.TestCase):
    """Tests for utility functions and constants exposed by the package."""

    def test_get_model_cache_dir_output(self):
        """Test get_model_cache_dir() returns a string path that is absolute."""
        cache_dir_str = steadytext.get_model_cache_dir()
        self.assertIsInstance(cache_dir_str, str)
        self.assertTrue(Path(cache_dir_str).is_absolute(), f"Cache directory '{cache_dir_str}' must be an absolute path.")

    def test_preload_models_runs_without_raising_unexpected_errors(self):
        """Test that preload_models() executes without throwing unexpected exceptions."""
        try:
            # verbose=False to keep test logs cleaner unless specifically debugging preload.
            steadytext.preload_models(verbose=False)
        except Exception as e:
            # preload_models itself is designed to catch and log errors, not re-raise them.
            self.fail(f"steadytext.preload_models() raised an unexpected exception: {type(e).__name__} - {e}")

    def test_constants_and_version_are_exposed(self):
        """Test that key constants and __version__ are accessible from the package."""
        self.assertEqual(steadytext.DEFAULT_SEED, 42)
        self.assertEqual(steadytext.GENERATION_MAX_NEW_TOKENS, 100)
        self.assertEqual(steadytext.EMBEDDING_DIMENSION, 1024)
        self.assertIsInstance(steadytext.__version__, str)
        self.assertTrue(len(steadytext.__version__) > 0, "Version string should be non-empty.")
        self.assertIsNotNone(steadytext.logger, "The package logger should be accessible.")

# Helper function for test_embed_list_averaging_and_empty_string_handling (not part of library)
def _test_normalize_l2(vector: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < tolerance: return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)

if __name__ == "__main__":
    # To run tests directly from this file: `python -m steadytext.tests.test_steadytext`
    # To enable model downloads for local testing if skipped by default:
    # `STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true python -m steadytext.tests.test_steadytext`

    print(f"--- Running SteadyText Test Suite ---")
    if not ALLOW_MODEL_DOWNLOADS:
        print("INFO: STEADYTEXT_ALLOW_MODEL_DOWNLOADS environment variable is not set to 'true'. "
              "Model-dependent tests will be skipped. Set this variable to 'true' to run all tests.")

    unittest.main()
