"""Contract tests for OpenRouterProvider.is_available() method.

AIDEV-NOTE: These tests verify the availability checking interface defined in the contract.
They MUST fail initially since OpenRouterProvider doesn't exist yet (TDD requirement).
"""

import pytest
import os
from unittest.mock import patch, Mock, MagicMock

# AIDEV-NOTE: This import will fail initially - this is expected for TDD
try:
    from steadytext.providers.openrouter import OpenRouterProvider
    OPENROUTER_PROVIDER_AVAILABLE = True
except ImportError:
    OPENROUTER_PROVIDER_AVAILABLE = False


class TestOpenRouterProviderAvailability:
    """Test OpenRouterProvider.is_available() according to contract."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_is_available_with_valid_api_key(self, monkeypatch):
        """Test is_available returns True with valid API key."""
        # Mock successful HTTP response
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="valid-test-key")
            assert provider.is_available() is True

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_is_available_with_invalid_api_key(self, monkeypatch):
        """Test is_available returns False with invalid API key."""
        # Mock authentication error response
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="invalid-key")
            assert provider.is_available() is False

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_is_available_with_missing_api_key(self, monkeypatch):
        """Test is_available returns False when API key is missing."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        # Since initialization should raise RuntimeError with missing key,
        # we test that availability check handles this correctly
        try:
            provider = OpenRouterProvider(api_key=None)
            # If it somehow gets created, it should not be available
            assert provider.is_available() is False
        except RuntimeError:
            # Expected behavior - missing API key should prevent initialization
            pass

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_is_available_with_network_error(self):
        """Test is_available returns False on network errors."""
        # Mock network connection error
        with patch('requests.get') as mock_get:
            mock_get.side_effect = ConnectionError("Network unreachable")

            provider = OpenRouterProvider(api_key="test-key")
            assert provider.is_available() is False

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_is_available_with_timeout(self):
        """Test is_available returns False on timeout."""
        # Mock timeout error
        with patch('requests.get') as mock_get:
            mock_get.side_effect = TimeoutError("Request timed out")

            provider = OpenRouterProvider(api_key="test-key")
            assert provider.is_available() is False

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_is_available_with_server_error(self):
        """Test is_available returns False on server errors."""
        # Mock server error response
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.json.return_value = {"error": {"message": "Internal server error"}}
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            assert provider.is_available() is False

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_is_available_caches_result(self):
        """Test that is_available caches the result for performance."""
        # Mock successful response
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")

            # First call
            result1 = provider.is_available()
            assert result1 is True

            # Second call should use cached result
            result2 = provider.is_available()
            assert result2 is True

            # Should only have made one HTTP request
            assert mock_get.call_count == 1

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_is_available_checks_correct_endpoint(self):
        """Test that is_available makes request to correct OpenRouter endpoint."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            provider.is_available()

            # Verify correct endpoint was called
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "openrouter.ai/api/v1" in call_args[0][0]

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_is_available_includes_auth_header(self):
        """Test that is_available includes correct authentication header."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-auth-key")
            provider.is_available()

            # Verify authentication header was included
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert 'headers' in call_kwargs
            assert 'Authorization' in call_kwargs['headers']
            assert call_kwargs['headers']['Authorization'] == 'Bearer test-auth-key'

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_is_available_respects_timeout(self):
        """Test that is_available respects configured timeout."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")
            provider.is_available()

            # Verify timeout was configured
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert 'timeout' in call_kwargs
            # Should use reasonable timeout (documented in contract as 30s connect, 120s read)
            expected_timeout = (30, 120)
            assert call_kwargs['timeout'] == expected_timeout


def test_openrouter_availability_import_fails():
    """Test that OpenRouterProvider import fails for availability tests (TDD requirement).

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


class TestOpenRouterAvailabilityContractCompliance:
    """Test that availability checking complies with contract requirements."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_is_available_returns_boolean(self):
        """Test that is_available always returns a boolean."""
        with patch('requests.get') as mock_get:
            # Test various response scenarios
            scenarios = [
                (200, {"data": []}),  # Success
                (401, {"error": {"message": "Unauthorized"}}),  # Auth error
                (500, {"error": {"message": "Server error"}})  # Server error
            ]

            for status_code, response_data in scenarios:
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_response.json.return_value = response_data
                mock_get.return_value = mock_response

                provider = OpenRouterProvider(api_key="test-key")
                result = provider.is_available()
                assert isinstance(result, bool), f"Expected bool, got {type(result)} for status {status_code}"

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_is_available_no_side_effects(self):
        """Test that is_available doesn't modify provider state."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": []}
            mock_get.return_value = mock_response

            provider = OpenRouterProvider(api_key="test-key")

            # Capture initial state
            initial_api_key = provider.api_key
            initial_model = provider.model

            # Call is_available
            provider.is_available()

            # Verify state unchanged
            assert provider.api_key == initial_api_key
            assert provider.model == initial_model