"""Tests for temperature parameter in text generation.

AIDEV-NOTE: These tests verify that temperature parameter works correctly
and that determinism is maintained with the same seed+temperature combination.
"""

import os
import unittest
from unittest.mock import Mock, patch

from steadytext import generate, generate_iter
from steadytext.utils import DEFAULT_SEED

# Allow tests requiring model downloads to be skipped
ALLOW_MODEL_DOWNLOADS = (
    os.environ.get("STEADYTEXT_ALLOW_MODEL_DOWNLOADS", "true").lower() == "true"
)


class TestTemperatureSupport(unittest.TestCase):
    """Test temperature parameter functionality."""

    def test_generate_accepts_temperature(self):
        """Test that generate function accepts temperature parameter."""
        # Mock the generator to avoid actual model loading
        with patch("steadytext.core.generator._get_generator_instance") as mock_gen:
            mock_instance = Mock()
            mock_instance.generate = Mock(return_value="Test output")
            mock_gen.return_value = mock_instance

            # Test with different temperature values
            result = generate("Test prompt", temperature=0.0)
            self.assertIsNotNone(result)

            # Verify temperature was passed to generator
            mock_instance.generate.assert_called_with(
                prompt="Test prompt",
                max_new_tokens=None,
                return_logprobs=False,
                eos_string="[EOS]",
                model=None,
                model_repo=None,
                model_filename=None,
                size=None,
                seed=DEFAULT_SEED,
                temperature=0.0,
                response_format=None,
                schema=None,
                regex=None,
                choices=None,
                return_pydantic=False,
                options=None,
            )

            # Test with higher temperature
            result = generate("Test prompt", temperature=0.7)
            mock_instance.generate.assert_called_with(
                prompt="Test prompt",
                max_new_tokens=None,
                return_logprobs=False,
                eos_string="[EOS]",
                model=None,
                model_repo=None,
                model_filename=None,
                size=None,
                seed=DEFAULT_SEED,
                temperature=0.7,
                response_format=None,
                schema=None,
                regex=None,
                choices=None,
                return_pydantic=False,
                options=None,
            )

    def test_generate_iter_accepts_temperature(self):
        """Test that generate_iter function accepts temperature parameter."""
        # Disable daemon to ensure direct generation path is used
        with patch.dict(os.environ, {"STEADYTEXT_DISABLE_DAEMON": "1"}):
            with patch("steadytext.core.generator._get_generator_instance") as mock_gen:
                mock_instance = Mock()
                # Use a Mock that tracks calls but returns a generator
                # This properly works with yield from in core_generate_iter
                called_with = {}

                def mock_gen_iter(*args, **kwargs):
                    # Store the kwargs for later assertion
                    called_with.update(kwargs)
                    yield "Test"
                    yield " "
                    yield "output"

                mock_instance.generate_iter = mock_gen_iter
                mock_gen.return_value = mock_instance

                # Test with temperature
                tokens = list(generate_iter("Test prompt", temperature=0.5))
                self.assertEqual(tokens, ["Test", " ", "output"])

                # Verify temperature was passed
                self.assertEqual(called_with.get("temperature"), 0.5)
                self.assertEqual(called_with.get("prompt"), "Test prompt")
                self.assertEqual(called_with.get("eos_string"), "[EOS]")
                self.assertEqual(called_with.get("seed"), DEFAULT_SEED)

    def test_cache_key_includes_temperature(self):
        """Test that cache key generation includes temperature."""
        from steadytext.utils import generate_cache_key

        # Same prompt, different temperatures should produce different cache keys
        key1 = generate_cache_key("Test prompt", "[EOS]", 0.0)
        key2 = generate_cache_key("Test prompt", "[EOS]", 0.5)
        key3 = generate_cache_key("Test prompt", "[EOS]", 1.0)

        # Default temperature (0.0) should not add temperature to key
        self.assertEqual(key1, "Test prompt")

        # Non-default temperatures should be included in key
        self.assertEqual(key2, "Test prompt::TEMP::0.5")
        self.assertEqual(key3, "Test prompt::TEMP::1.0")

        # Different temperatures should produce different keys
        self.assertNotEqual(key1, key2)
        self.assertNotEqual(key2, key3)

    def test_sampling_params_with_temperature(self):
        """Test that sampling parameters are adjusted based on temperature."""
        from steadytext.utils import get_sampling_params_with_temperature

        # Temperature 0.0 should use deterministic parameters
        params = get_sampling_params_with_temperature(0.0)
        self.assertEqual(params["temperature"], 0.0)
        self.assertEqual(params["top_k"], 1)
        self.assertEqual(params["top_p"], 1.0)
        self.assertEqual(params["min_p"], 0.0)

        # Higher temperature should adjust sampling parameters
        params = get_sampling_params_with_temperature(0.7)
        self.assertEqual(params["temperature"], 0.7)
        self.assertEqual(params["top_k"], 40)  # More tokens considered
        self.assertEqual(params["top_p"], 0.95)  # Nucleus sampling
        self.assertEqual(params["min_p"], 0.05)  # Minimum probability threshold

    @unittest.skipUnless(
        ALLOW_MODEL_DOWNLOADS,
        "Skipping model-dependent tests (STEADYTEXT_ALLOW_MODEL_DOWNLOADS not 'true')",
    )
    def test_determinism_with_same_seed_and_temperature(self):
        """Test that same seed + temperature produces identical results."""
        # Skip if model loading is disabled
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
            self.skipTest("Model loading disabled")

        prompt = "Hello world"
        seed = 123
        temperature = 0.5

        # Generate text twice with same parameters
        result1 = generate(
            prompt, seed=seed, temperature=temperature, max_new_tokens=20
        )
        result2 = generate(
            prompt, seed=seed, temperature=temperature, max_new_tokens=20
        )

        # Results should be identical
        if result1 is not None and result2 is not None:
            self.assertEqual(
                result1,
                result2,
                "Same seed + temperature should produce identical results",
            )

    @unittest.skipUnless(
        ALLOW_MODEL_DOWNLOADS,
        "Skipping model-dependent tests (STEADYTEXT_ALLOW_MODEL_DOWNLOADS not 'true')",
    )
    def test_different_temperatures_produce_different_outputs(self):
        """Test that different temperatures produce different outputs."""
        # Skip if model loading is disabled
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
            self.skipTest("Model loading disabled")

        prompt = "The weather today is"
        seed = 456

        # Generate with different temperatures
        result_0 = generate(prompt, seed=seed, temperature=0.0, max_new_tokens=20)
        result_5 = generate(prompt, seed=seed, temperature=0.5, max_new_tokens=20)
        result_10 = generate(prompt, seed=seed, temperature=1.0, max_new_tokens=20)

        # Different temperatures should generally produce different outputs
        # (though not guaranteed for all prompts/seeds)
        if result_0 and result_5 and result_10:
            # At least one should be different
            self.assertTrue(
                result_0 != result_5 or result_5 != result_10,
                "Different temperatures should generally produce different outputs",
            )

    def test_daemon_client_passes_temperature(self):
        """Test that daemon client correctly passes temperature parameter."""
        from steadytext.daemon.client import DaemonClient

        with patch("zmq.Context") as mock_context_class:
            mock_context = Mock()
            mock_socket = Mock()
            mock_context_class.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            # Mock successful connection
            mock_socket.connect = Mock()
            mock_socket.send = Mock()
            mock_socket.recv = Mock(
                return_value=b'{"id": "test-id", "result": "Test output"}'
            )

            client = DaemonClient()
            client._connected = True
            client.socket = mock_socket

            # Call generate with temperature
            client.generate("Test prompt", temperature=0.8)

            # Verify request included temperature
            sent_data = mock_socket.send.call_args[0][0]
            import json

            request = json.loads(sent_data)
            self.assertEqual(request["params"]["temperature"], 0.8)

    def test_cli_temperature_flag(self):
        """Test that CLI accepts --temperature flag."""
        from click.testing import CliRunner
        from steadytext.cli.main import cli

        runner = CliRunner()

        # Test with temperature flag
        with patch("steadytext.generate") as mock_generate:
            mock_generate.return_value = "Test output"

            runner.invoke(
                cli, ["generate", "--temperature", "0.7", "--wait"], input="Test prompt"
            )

            # Check that generate was called with temperature
            if mock_generate.called:
                call_kwargs = mock_generate.call_args[1]
                self.assertEqual(call_kwargs.get("temperature"), 0.7)


class TestTemperatureWithProviders(unittest.TestCase):
    """Test temperature with remote providers."""

    def test_openai_provider_temperature(self):
        """Test that OpenAI provider accepts temperature."""
        from steadytext.providers.openai import OpenAIProvider

        with patch("steadytext.providers.openai._get_openai") as mock_openai:
            # Mock OpenAI module
            mock_module = Mock()
            mock_client = Mock()
            mock_completion = Mock()
            mock_completion.choices = [Mock(message=Mock(content="Test response"))]

            mock_module.OpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_module

            provider = OpenAIProvider(api_key="test-key")
            provider.generate("Test prompt", temperature=0.6)

            # Verify temperature was passed to OpenAI
            create_call = mock_client.chat.completions.create
            create_call.assert_called()
            call_kwargs = create_call.call_args[1]
            self.assertEqual(call_kwargs.get("temperature"), 0.6)

    def test_cerebras_provider_temperature(self):
        """Test that Cerebras provider accepts temperature."""
        from steadytext.providers.cerebras import CerebrasProvider

        with patch("steadytext.providers.cerebras._get_openai") as mock_openai:
            # Mock OpenAI module (used by Cerebras)
            mock_module = Mock()
            mock_client = Mock()
            mock_completion = Mock()
            mock_completion.choices = [Mock(message=Mock(content="Test response"))]

            mock_module.OpenAI.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_completion
            mock_openai.return_value = mock_module

            provider = CerebrasProvider(api_key="test-key")
            provider.generate("Test prompt", temperature=0.9)

            # Verify temperature was passed
            create_call = mock_client.chat.completions.create
            create_call.assert_called()
            call_kwargs = create_call.call_args[1]
            self.assertEqual(call_kwargs.get("temperature"), 0.9)


if __name__ == "__main__":
    unittest.main()
