"""Contract tests for OpenRouterProvider.get_supported_models() method.

AIDEV-NOTE: These tests verify the model listing interface defined in the contract.
They MUST fail initially since OpenRouterProvider doesn't exist yet (TDD requirement).
"""

import pytest
from unittest.mock import patch, Mock

# AIDEV-NOTE: This import will fail initially - this is expected for TDD
try:
    from steadytext.providers.openrouter import OpenRouterProvider
    OPENROUTER_PROVIDER_AVAILABLE = True
except ImportError:
    OPENROUTER_PROVIDER_AVAILABLE = False


class TestOpenRouterProviderModels:
    """Test OpenRouterProvider.get_supported_models() according to contract."""

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_supported_models_basic(self):
        """Test basic model listing functionality."""
        # Mock OpenRouter API response
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {
                        "id": "anthropic/claude-3.5-sonnet",
                        "name": "Anthropic: Claude 3.5 Sonnet",
                        "pricing": {"prompt": "0.000003", "completion": "0.000015"}
                    },
                    {
                        "id": "openai/gpt-4o-mini",
                        "name": "OpenAI: GPT-4o mini",
                        "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}
                    },
                    {
                        "id": "meta-llama/llama-3.1-8b-instruct",
                        "name": "Meta: Llama 3.1 8B Instruct",
                        "pricing": {"prompt": "0.00000018", "completion": "0.00000018"}
                    }
                ]
            }
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            models = provider.get_supported_models()

            # Should return list of model IDs in OpenRouter format
            assert isinstance(models, list)
            assert len(models) == 3
            assert "anthropic/claude-3.5-sonnet" in models
            assert "openai/gpt-4o-mini" in models
            assert "meta-llama/llama-3.1-8b-instruct" in models

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_supported_models_filters_format(self):
        """Test that only properly formatted models are included."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "anthropic/claude-3.5-sonnet", "name": "Valid Model 1"},
                    {"id": "openai/gpt-4o-mini", "name": "Valid Model 2"},
                    {"id": "invalid-model-without-slash", "name": "Invalid Model"},
                    {"id": "another/valid/model", "name": "Valid Model 3"},
                    {"id": "", "name": "Empty Model"},
                    {"id": "provider/", "name": "No Model Name"}
                ]
            }
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            models = provider.get_supported_models()

            # Should only include properly formatted models (provider/model)
            valid_models = [m for m in models if "/" in m and len(m.split("/")) >= 2]
            assert len(valid_models) == len(models)  # All returned models should be valid
            assert "anthropic/claude-3.5-sonnet" in models
            assert "openai/gpt-4o-mini" in models
            assert "another/valid/model" in models
            assert "invalid-model-without-slash" not in models

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_supported_models_caches_result(self):
        """Test that model list is cached for performance."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"}
                ]
            }
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")

            # First call
            models1 = provider.get_supported_models()
            assert len(models1) == 1

            # Second call should use cached result
            models2 = provider.get_supported_models()
            assert models1 == models2

            # Should only have made one HTTP request
            assert mock_get.call_count == 1

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_supported_models_handles_auth_error(self):
        """Test handling of authentication errors."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.json.return_value = {
                "error": {"message": "Invalid API key"}
            }
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="invalid-key")

            # Should return empty list on auth error (graceful degradation)
            models = provider.get_supported_models()
            assert isinstance(models, list)
            assert len(models) == 0

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_supported_models_handles_network_error(self):
        """Test handling of network errors."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = ConnectionError("Network unreachable")

            provider = OpenRouterProvider(api_key="test-key")

            # Should return empty list on network error (graceful degradation)
            models = provider.get_supported_models()
            assert isinstance(models, list)
            assert len(models) == 0

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_supported_models_handles_server_error(self):
        """Test handling of server errors."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.json.return_value = {
                "error": {"message": "Internal server error"}
            }
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")

            # Should return empty list on server error (graceful degradation)
            models = provider.get_supported_models()
            assert isinstance(models, list)
            assert len(models) == 0

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_supported_models_includes_auth_header(self):
        """Test that model listing includes correct authentication header."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-auth-key")
            provider.get_supported_models()

            # Verify authentication header was included
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert 'headers' in call_kwargs
            assert 'Authorization' in call_kwargs['headers']
            assert call_kwargs['headers']['Authorization'] == 'Bearer test-auth-key'

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_supported_models_uses_correct_endpoint(self):
        """Test that model listing uses correct OpenRouter endpoint."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            provider.get_supported_models()

            # Verify correct endpoint
            mock_get.assert_called_once()
            call_url = mock_get.call_args[0][0]
            assert "openrouter.ai/api/v1/models" in call_url

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_supported_models_sorts_alphabetically(self):
        """Test that models are returned in alphabetical order."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "openai/gpt-4o-mini", "name": "GPT-4o mini"},
                    {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
                    {"id": "meta-llama/llama-3.1-8b-instruct", "name": "Llama 3.1 8B"},
                    {"id": "google/gemma-2-9b-it", "name": "Gemma 2 9B"}
                ]
            }
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            models = provider.get_supported_models()

            # Should be sorted alphabetically
            assert models == sorted(models)

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_supported_models_handles_empty_response(self):
        """Test handling of empty model list response."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            models = provider.get_supported_models()

            # Should handle empty list gracefully
            assert isinstance(models, list)
            assert len(models) == 0

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_supported_models_handles_malformed_response(self):
        """Test handling of malformed API response."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"invalid": "format"}
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            models = provider.get_supported_models()

            # Should handle malformed response gracefully
            assert isinstance(models, list)
            assert len(models) == 0

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_model_info_basic(self):
        """Test get_model_info method for specific models."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {
                        "id": "anthropic/claude-3.5-sonnet",
                        "name": "Anthropic: Claude 3.5 Sonnet",
                        "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                        "context_length": 200000,
                        "architecture": {"modality": "text", "tokenizer": "Claude"}
                    }
                ]
            }
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            info = provider.get_model_info("anthropic/claude-3.5-sonnet")

            # Should return dictionary with model information
            assert isinstance(info, dict)
            assert info["id"] == "anthropic/claude-3.5-sonnet"
            assert "pricing" in info
            assert "context_length" in info
            assert "architecture" in info

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_model_info_not_found(self):
        """Test get_model_info with model that doesn't exist."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}  # Empty results
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")

            from steadytext.providers.openrouter import OpenRouterModelError
            with pytest.raises(OpenRouterModelError, match="Model not found"):
                provider.get_model_info("nonexistent/model")


def test_openrouter_models_import_fails():
    """Test that OpenRouterProvider import fails for models tests (TDD requirement).

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


class TestOpenRouterModelsContractCompliance:
    """Test that model listing complies with contract requirements."""

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_supported_models_return_type(self):
        """Test that get_supported_models always returns a list."""
        with patch('requests.get') as mock_get:
            # Test various response scenarios
            scenarios = [
                (200, {"data": [{"id": "test/model", "name": "Test"}]}),  # Success
                (401, {"error": {"message": "Unauthorized"}}),  # Auth error
                (500, {"error": {"message": "Server error"}})  # Server error
            ]

            for status_code, response_data in scenarios:
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.json.return_value = response_data
                mock_get.return_value = mock_response

                provider = OpenRouterProvider(api_key="test-key")
                # Clear any cached results
                if hasattr(provider, '_cached_models'):
                    delattr(provider, '_cached_models')

                result = provider.get_supported_models()
                assert isinstance(result, list), f"Expected list, got {type(result)} for status {status_code}"

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_supported_models_string_elements(self):
        """Test that all returned models are strings in correct format."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "anthropic/claude-3.5-sonnet", "name": "Claude"},
                    {"id": "openai/gpt-4o-mini", "name": "GPT-4o mini"},
                    {"id": "meta-llama/llama-3.1-8b-instruct", "name": "Llama"}
                ]
            }
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            models = provider.get_supported_models()

            # All elements should be strings in provider/model format
            for model in models:
                assert isinstance(model, str)
                assert "/" in model
                parts = model.split("/")
                assert len(parts) >= 2
                assert all(part.strip() for part in parts)  # No empty parts

    @pytest.mark.slow
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_get_model_info_return_type(self):
        """Test that get_model_info returns correct type."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {
                        "id": "test/model",
                        "name": "Test Model",
                        "pricing": {"prompt": "0.001", "completion": "0.002"}
                    }
                ]
            }
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            result = provider.get_model_info("test/model")

            # Should return dictionary
            assert isinstance(result, dict)
            assert "id" in result
            assert "name" in result