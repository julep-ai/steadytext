"""Contract tests for OpenRouterProvider.generate() method.

AIDEV-NOTE: These tests verify the text generation interface defined in the contract.
They MUST fail initially since OpenRouterProvider doesn't exist yet (TDD requirement).
"""

import pytest
import warnings
from unittest.mock import patch, Mock, MagicMock
from typing import Iterator

# AIDEV-NOTE: This import will fail initially - this is expected for TDD
try:
    from steadytext.providers.openrouter import OpenRouterProvider
    from steadytext.providers.base import UnsafeModeWarning
    OPENROUTER_PROVIDER_AVAILABLE = True
except ImportError:
    OPENROUTER_PROVIDER_AVAILABLE = False
    # Define UnsafeModeWarning for tests
    class UnsafeModeWarning(UserWarning):
        pass


class TestOpenRouterProviderGenerate:
    """Test OpenRouterProvider.generate() according to contract."""

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_basic_text(self):
        """Test basic text generation."""
        # Mock OpenRouter API response
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "Hello, this is a generated response."
                    }
                }]
            }
            mock_post.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = provider.generate("Hello, world!")

                # Should issue unsafe mode warning
                assert len(w) == 1
                assert issubclass(w[0].category, UnsafeModeWarning)

            assert result == "Hello, this is a generated response."

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_with_custom_model(self):
        """Test generation with custom model override."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Custom model response"}}]
            }
            mock_post.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key", model="anthropic/claude-3.5-sonnet")
            result = provider.generate("Test", model="openai/gpt-4o-mini")

            # Verify correct model was used in API call
            mock_post.assert_called_once()
            call_data = mock_post.call_args[1]['json']
            assert call_data['model'] == "openai/gpt-4o-mini"

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_with_temperature(self):
        """Test generation with temperature parameter."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Temperature test response"}}]
            }
            mock_post.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            result = provider.generate("Test", temperature=0.8)

            # Verify temperature was passed to API
            call_data = mock_post.call_args[1]['json']
            assert call_data['temperature'] == 0.8

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_with_max_tokens(self):
        """Test generation with max_tokens parameter."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Max tokens test"}}]
            }
            mock_post.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            result = provider.generate("Test", max_tokens=100)

            # Verify max_tokens was passed to API
            call_data = mock_post.call_args[1]['json']
            assert call_data['max_tokens'] == 100

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_with_top_p(self):
        """Test generation with top_p parameter."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Top-p test response"}}]
            }
            mock_post.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            result = provider.generate("Test", top_p=0.9)

            # Verify top_p was passed to API
            call_data = mock_post.call_args[1]['json']
            assert call_data['top_p'] == 0.9

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_streaming_mode(self):
        """Test generation with streaming enabled."""
        # Mock streaming response
        def mock_iter_lines():
            yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
            yield b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
            yield b'data: [DONE]\n\n'

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = mock_iter_lines()
            mock_post.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            result = provider.generate("Test", stream=True)

            # Should return an iterator
            assert isinstance(result, Iterator)

            # Collect streamed content
            content = list(result)
            assert "Hello" in content[0]
            assert " world" in content[1]

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_with_custom_kwargs(self):
        """Test generation with custom OpenRouter-specific parameters."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Custom kwargs response"}}]
            }
            mock_post.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            result = provider.generate(
                "Test",
                presence_penalty=0.5,
                frequency_penalty=0.3,
                transforms=["middle-out"]
            )

            # Verify custom parameters were passed
            call_data = mock_post.call_args[1]['json']
            assert call_data['presence_penalty'] == 0.5
            assert call_data['frequency_penalty'] == 0.3
            assert call_data['transforms'] == ["middle-out"]

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_auth_error_handling(self):
        """Test proper handling of authentication errors."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.json.return_value = {
                "error": {"message": "Invalid API key"}
            }
            mock_post.return_value = mock_response

            provider = OpenRouterProvider(api_key="invalid-key")

            from steadytext.providers.openrouter import OpenRouterAuthError
            with pytest.raises(OpenRouterAuthError, match="Invalid API key"):
                provider.generate("Test")

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_rate_limit_error_handling(self):
        """Test proper handling of rate limit errors."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.json.return_value = {
                "error": {"message": "Rate limit exceeded"}
            }
            mock_post.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")

            from steadytext.providers.openrouter import OpenRouterRateLimitError
            with pytest.raises(OpenRouterRateLimitError, match="Rate limit exceeded"):
                provider.generate("Test")

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_model_error_handling(self):
        """Test proper handling of model-related errors."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {
                "error": {"message": "Model not found: invalid/model"}
            }
            mock_post.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")

            from steadytext.providers.openrouter import OpenRouterModelError
            with pytest.raises(OpenRouterModelError, match="Model not found"):
                provider.generate("Test", model="invalid/model")

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_fallback_to_deterministic(self):
        """Test fallback to deterministic generation on API failure."""
        with patch('requests.post') as mock_post:
            # Simulate network error
            mock_post.side_effect = ConnectionError("Network error")

            provider = OpenRouterProvider(api_key="test-key")

            # Should fall back to deterministic generation
            result = provider.generate("Hello world")
            assert result is not None
            assert isinstance(result, str)
            # Deterministic fallback should produce consistent output
            result2 = provider.generate("Hello world")
            assert result == result2

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_includes_proper_headers(self):
        """Test that generate includes proper HTTP headers."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Test response"}}]
            }
            mock_post.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-auth-key")
            provider.generate("Test")

            # Verify headers
            call_kwargs = mock_post.call_args[1]
            headers = call_kwargs['headers']
            assert headers['Authorization'] == 'Bearer test-auth-key'
            assert headers['Content-Type'] == 'application/json'
            assert 'User-Agent' in headers

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_uses_correct_endpoint(self):
        """Test that generate makes request to correct OpenRouter endpoint."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Test response"}}]
            }
            mock_post.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            provider.generate("Test")

            # Verify correct endpoint
            mock_post.assert_called_once()
            call_url = mock_post.call_args[0][0]
            assert "openrouter.ai/api/v1/chat/completions" in call_url


def test_openrouter_generate_import_fails():
    """Test that OpenRouterProvider import fails for generate tests (TDD requirement).

    AIDEV-NOTE: This test MUST pass initially and MUST fail after implementation.
    """
    if OPENROUTER_PROVIDER_AVAILABLE:
        pytest.fail(
            "OpenRouterProvider should not be available yet for TDD. "
            "This test should fail after implementation."
        )

    # Verify the import fails as expected
    with pytest.raises(ImportError):
        from steadytext.providers.openrouter import OpenRouterProvider


class TestOpenRouterGenerateContractCompliance:
    """Test that generation complies with contract requirements."""

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_return_types(self):
        """Test that generate returns correct types based on stream parameter."""
        with patch('requests.post') as mock_post:
            # Non-streaming response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Non-streaming response"}}]
            }
            mock_post.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")

            # Non-streaming should return string
            result = provider.generate("Test", stream=False)
            assert isinstance(result, str)

        # Mock streaming response
        def mock_iter_lines():
            yield b'data: {"choices": [{"delta": {"content": "Stream"}}]}\n\n'
            yield b'data: [DONE]\n\n'

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = mock_iter_lines()
            mock_post.return_value = mock_response

            # Streaming should return iterator
            result = provider.generate("Test", stream=True)
            assert isinstance(result, Iterator)

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_generate_parameter_validation(self):
        """Test parameter validation according to contract."""
        provider = OpenRouterProvider(api_key="test-key")

        # Temperature should be within valid range
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            provider.generate("Test", temperature=3.0)

        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            provider.generate("Test", temperature=-1.0)

        # Top-p should be within valid range
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            provider.generate("Test", top_p=1.5)

        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            provider.generate("Test", top_p=-0.1)

        # Max tokens should be positive
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            provider.generate("Test", max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            provider.generate("Test", max_tokens=-10)