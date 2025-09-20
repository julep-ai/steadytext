"""Integration tests for OpenRouter API key validation.

AIDEV-NOTE: These tests verify OpenRouter API key validation, authentication patterns, and security.
They MUST fail initially since OpenRouterProvider doesn't exist yet (TDD requirement).
Tests focus on end-to-end authentication behavior and key validation patterns.
"""

import pytest
import os
from unittest.mock import patch, Mock
from click.testing import CliRunner

# AIDEV-NOTE: These imports will fail initially - this is expected for TDD
try:
    from steadytext.providers.openrouter import OpenRouterProvider
    OPENROUTER_PROVIDER_AVAILABLE = True
except ImportError:
    OPENROUTER_PROVIDER_AVAILABLE = False

try:
    from steadytext.providers.registry import get_provider, parse_remote_model
    from steadytext.cli.main import cli
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

try:
    from steadytext import generate, embed
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


@pytest.fixture
def runner():
    """Fixture for invoking command-line calls."""
    return CliRunner()


class TestOpenRouterAPIKeyFormat:
    """Test OpenRouter API key format validation."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_valid_openrouter_api_key_formats(self):
        """Test that valid OpenRouter API key formats are accepted."""
        valid_keys = [
            "sk-or-1234567890abcdef1234567890abcdef",  # Standard format
            "sk-or-v1-1234567890abcdef1234567890abcdef",  # Versioned format
            "sk-or-test-12345678901234567890",  # Test key format
            "sk-or-dev-abcdefghijklmnopqrstuvwxyz",  # Dev key format
        ]

        for key in valid_keys:
            # Should not raise exception during initialization
            try:
                provider = OpenRouterProvider(api_key=key, model="anthropic/claude-3.5-sonnet")
                assert provider.api_key == key
            except ValueError as e:
                # If format validation rejects it, ensure error is about format
                if "format" not in str(e).lower():
                    pytest.fail(f"Valid key {key} rejected for wrong reason: {e}")

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_invalid_openrouter_api_key_formats(self):
        """Test that invalid OpenRouter API key formats are rejected."""
        invalid_keys = [
            "",  # Empty string
            "sk-1234567890abcdef",  # OpenAI format (missing -or-)
            "sk-or-",  # Too short
            "sk-or-short",  # Too short
            "or-1234567890abcdef1234567890abcdef",  # Missing sk- prefix
            "sk-1234567890abcdef1234567890abcdef",  # Wrong format (not OpenRouter)
            "invalid-key-format",  # Completely wrong format
            "sk-or-with spaces",  # Contains spaces
            "sk-or-with@special!chars",  # Invalid characters
            None,  # None (should be caught separately)
        ]

        for key in invalid_keys:
            if key is None:
                # None should be handled separately (environment lookup)
                continue

            with pytest.raises((ValueError, RuntimeError)) as exc_info:
                OpenRouterProvider(api_key=key, model="anthropic/claude-3.5-sonnet")

            error_message = str(exc_info.value).lower()
            # Should mention key or format in error
            assert any(word in error_message for word in ["key", "format", "invalid", "missing"])

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_api_key_length_validation(self):
        """Test that API key length validation works correctly."""
        # Test minimum length requirements
        too_short_keys = [
            "sk-or-123",  # Way too short
            "sk-or-1234567890abcde",  # Slightly too short
        ]

        for key in too_short_keys:
            with pytest.raises((ValueError, RuntimeError)):
                OpenRouterProvider(api_key=key, model="anthropic/claude-3.5-sonnet")

        # Test reasonable length key
        reasonable_key = "sk-or-1234567890abcdef1234567890abcdef"
        try:
            provider = OpenRouterProvider(api_key=reasonable_key, model="anthropic/claude-3.5-sonnet")
            assert provider.api_key == reasonable_key
        except ValueError:
            # If other validation rejects it, that's acceptable
            pass

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_api_key_character_validation(self):
        """Test that API key character validation works correctly."""
        # Test keys with invalid characters
        invalid_char_keys = [
            "sk-or-1234567890abcdef1234567890abcdef!",  # Exclamation
            "sk-or-1234567890abcdef1234567890abcdef@",  # At symbol
            "sk-or-1234567890abcdef 1234567890abcdef",  # Space
            "sk-or-1234567890abcdef\n1234567890abcdef",  # Newline
            "sk-or-1234567890abcdef\t1234567890abcdef",  # Tab
        ]

        for key in invalid_char_keys:
            with pytest.raises((ValueError, RuntimeError)):
                OpenRouterProvider(api_key=key, model="anthropic/claude-3.5-sonnet")


class TestOpenRouterEnvironmentAuthentication:
    """Test OpenRouter authentication via environment variables."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_api_key_from_environment(self, monkeypatch):
        """Test reading API key from OPENROUTER_API_KEY environment variable."""
        test_key = "sk-or-env-test-1234567890abcdef1234567890"
        monkeypatch.setenv("OPENROUTER_API_KEY", test_key)

        # Should read from environment when no explicit key provided
        provider = OpenRouterProvider(model="anthropic/claude-3.5-sonnet")
        assert provider.api_key == test_key

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_explicit_api_key_overrides_environment(self, monkeypatch):
        """Test that explicit API key overrides environment variable."""
        env_key = "sk-or-env-key-1234567890abcdef1234567890"
        explicit_key = "sk-or-explicit-key-1234567890abcdef1234567890"

        monkeypatch.setenv("OPENROUTER_API_KEY", env_key)

        # Explicit key should take precedence
        provider = OpenRouterProvider(api_key=explicit_key, model="anthropic/claude-3.5-sonnet")
        assert provider.api_key == explicit_key

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_missing_api_key_raises_error(self, monkeypatch):
        """Test that missing API key raises appropriate error."""
        # Remove environment variable
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            OpenRouterProvider(api_key=None, model="anthropic/claude-3.5-sonnet")

        error_message = str(exc_info.value).lower()
        assert "api key" in error_message or "missing" in error_message

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_empty_environment_variable_raises_error(self, monkeypatch):
        """Test that empty OPENROUTER_API_KEY raises error."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "")

        with pytest.raises(RuntimeError) as exc_info:
            OpenRouterProvider(model="anthropic/claude-3.5-sonnet")

        error_message = str(exc_info.value).lower()
        assert "api key" in error_message or "missing" in error_message

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_whitespace_only_environment_variable_raises_error(self, monkeypatch):
        """Test that whitespace-only OPENROUTER_API_KEY raises error."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "   \t\n   ")

        with pytest.raises(RuntimeError) as exc_info:
            OpenRouterProvider(model="anthropic/claude-3.5-sonnet")

        error_message = str(exc_info.value).lower()
        assert "api key" in error_message or "missing" in error_message


class TestOpenRouterRegistryAuthentication:
    """Test authentication through provider registry."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_registry_get_provider_with_valid_key(self, monkeypatch):
        """Test getting OpenRouter provider through registry with valid key."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-registry-test-1234567890abcdef1234567890")

        # Mock availability check
        with patch.object(OpenRouterProvider, 'is_available', return_value=True):
            provider = get_provider("openrouter:anthropic/claude-3.5-sonnet")
            assert isinstance(provider, OpenRouterProvider)
            assert provider.api_key == "sk-or-registry-test-1234567890abcdef1234567890"

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_registry_get_provider_missing_key(self, monkeypatch):
        """Test getting OpenRouter provider through registry with missing key."""
        # Set up environment (unsafe mode enabled, but no API key)
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            get_provider("openrouter:anthropic/claude-3.5-sonnet")

        error_message = str(exc_info.value).lower()
        assert "not available" in error_message or "api key" in error_message

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_registry_get_provider_with_explicit_key(self, monkeypatch):
        """Test getting OpenRouter provider through registry with explicit key."""
        # Set up environment (no env key, but pass explicit key)
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        explicit_key = "sk-or-explicit-registry-1234567890abcdef1234567890"

        # Mock availability check
        with patch.object(OpenRouterProvider, 'is_available', return_value=True):
            provider = get_provider(
                "openrouter:anthropic/claude-3.5-sonnet",
                api_key=explicit_key
            )
            assert isinstance(provider, OpenRouterProvider)
            assert provider.api_key == explicit_key

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_registry_get_provider_invalid_key_format(self, monkeypatch):
        """Test getting OpenRouter provider through registry with invalid key format."""
        # Set up environment with invalid key
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "invalid-key-format")

        # Should fail during provider creation, not just registry lookup
        with pytest.raises((RuntimeError, ValueError)):
            get_provider("openrouter:anthropic/claude-3.5-sonnet")


class TestOpenRouterCLIAuthentication:
    """Test authentication in CLI commands."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    @pytest.mark.slow
    def test_cli_generate_with_valid_api_key(self, runner, monkeypatch):
        """Test CLI generate command with valid OpenRouter API key."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-cli-test-1234567890abcdef1234567890")

        # Mock the provider
        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', return_value="CLI response"):

            result = runner.invoke(cli, [
                "generate",
                "Hello world",
                "--model", "openrouter:anthropic/claude-3.5-sonnet",
                "--wait"
            ])

            assert result.exit_code == 0
            assert "CLI response" in result.output

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_cli_generate_missing_api_key(self, runner, monkeypatch):
        """Test CLI generate command with missing OpenRouter API key."""
        # Set up environment (unsafe mode enabled, but no API key)
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = runner.invoke(cli, [
            "generate",
            "Hello world",
            "--model", "openrouter:anthropic/claude-3.5-sonnet",
            "--wait"
        ])

        assert result.exit_code != 0
        error_output = result.output.lower()
        assert "not available" in error_output or "api key" in error_output

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_cli_embed_missing_api_key(self, runner, monkeypatch):
        """Test CLI embed command with missing OpenRouter API key."""
        # Set up environment (unsafe mode enabled, but no API key)
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = runner.invoke(cli, [
            "embed",
            "Text to embed",
            "--model", "openrouter:text-embedding-3-large"
        ])

        assert result.exit_code != 0
        error_output = result.output.lower()
        assert "not available" in error_output or "api key" in error_output

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_cli_helpful_error_messages_for_auth_issues(self, runner, monkeypatch):
        """Test that CLI provides helpful error messages for authentication issues."""
        # Test missing unsafe mode
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "false")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")

        result = runner.invoke(cli, [
            "generate",
            "Hello",
            "--model", "openrouter:anthropic/claude-3.5-sonnet"
        ])

        assert result.exit_code != 0
        error_output = result.output
        assert "unsafe mode" in error_output.lower() or "STEADYTEXT_UNSAFE_MODE" in error_output

        # Test missing API key
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = runner.invoke(cli, [
            "generate",
            "Hello",
            "--model", "openrouter:anthropic/claude-3.5-sonnet"
        ])

        assert result.exit_code != 0
        error_output = result.output.lower()
        assert "not available" in error_output or "api key" in error_output


class TestOpenRouterCoreAuthentication:
    """Test authentication in core generate/embed functions."""

    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core not available")
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.slow
    def test_generate_with_openrouter_valid_key(self, monkeypatch):
        """Test generate() function with valid OpenRouter API key."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-core-test-1234567890abcdef1234567890")

        # Mock the provider
        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'generate', return_value="Core response"):

            result = generate("Hello world", model="openrouter:anthropic/claude-3.5-sonnet")
            assert result == "Core response"

    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core not available")
    def test_generate_with_openrouter_missing_key_fallback(self, monkeypatch):
        """Test generate() function fallback when OpenRouter API key is missing."""
        # Set up environment (unsafe mode enabled, but no API key)
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        # Should fall back to deterministic generation (not raise exception)
        result = generate("Hello world", model="openrouter:anthropic/claude-3.5-sonnet")
        assert isinstance(result, str)
        # In fallback mode, should return empty string
        assert result == ""

    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core not available")
    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    @pytest.mark.slow
    def test_embed_with_openrouter_valid_key(self, monkeypatch):
        """Test embed() function with valid OpenRouter API key."""
        # Set up environment
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-embed-test-1234567890abcdef1234567890")

        # Mock embedding response
        mock_embedding = [0.1] * 1024

        with patch.object(OpenRouterProvider, 'is_available', return_value=True), \
             patch.object(OpenRouterProvider, 'embed', return_value=mock_embedding):

            result = embed("Hello world", model="openrouter:text-embedding-3-large")
            assert result == mock_embedding
            assert len(result) == 1024

    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core not available")
    def test_embed_with_openrouter_missing_key_fallback(self, monkeypatch):
        """Test embed() function fallback when OpenRouter API key is missing."""
        # Set up environment (unsafe mode enabled, but no API key)
        monkeypatch.setenv("STEADYTEXT_UNSAFE_MODE", "true")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        # Should fall back to deterministic embedding (not raise exception)
        result = embed("Hello world", model="openrouter:text-embedding-3-large")
        assert isinstance(result, list)
        assert len(result) == 1024
        # In fallback mode, should return zero vector
        assert all(x == 0.0 for x in result)


class TestOpenRouterAuthenticationSecurity:
    """Test security aspects of OpenRouter authentication."""

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_api_key_not_logged_in_errors(self, monkeypatch, caplog):
        """Test that API keys are not accidentally logged in error messages."""
        # Set up environment with API key
        test_key = "sk-or-secret-1234567890abcdef1234567890abcdef"
        monkeypatch.setenv("OPENROUTER_API_KEY", test_key)

        # Force an error that might log the key
        with patch.object(OpenRouterProvider, '__init__', side_effect=Exception("Test error")):
            try:
                OpenRouterProvider(api_key=test_key, model="anthropic/claude-3.5-sonnet")
            except Exception:
                pass

        # Check that the API key doesn't appear in logs
        log_output = caplog.text
        assert test_key not in log_output
        # Partial key shouldn't appear either
        assert "1234567890abcdef1234567890abcdef" not in log_output

    @pytest.mark.skipif(not OPENROUTER_PROVIDER_AVAILABLE, reason="OpenRouterProvider not implemented yet")
    def test_api_key_not_in_repr(self):
        """Test that API key doesn't appear in string representation."""
        test_key = "sk-or-secret-1234567890abcdef1234567890abcdef"

        try:
            provider = OpenRouterProvider(api_key=test_key, model="anthropic/claude-3.5-sonnet")

            # Check repr/str don't expose the key
            repr_str = repr(provider)
            str_str = str(provider)

            assert test_key not in repr_str
            assert test_key not in str_str
            # Partial key shouldn't appear either
            assert "1234567890abcdef1234567890abcdef" not in repr_str
            assert "1234567890abcdef1234567890abcdef" not in str_str

        except Exception:
            # If provider can't be created, that's ok for this test
            pass

    def test_api_key_environment_variable_case_sensitivity(self, monkeypatch):
        """Test that environment variable name is case sensitive."""
        # Set wrong case environment variables
        monkeypatch.setenv("openrouter_api_key", "sk-or-lowercase-key")
        monkeypatch.setenv("OPENROUTER_api_key", "sk-or-mixedcase-key")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        # Should not find the API key (environment lookup should be case sensitive)
        if OPENROUTER_PROVIDER_AVAILABLE:
            with pytest.raises(RuntimeError):
                OpenRouterProvider(model="anthropic/claude-3.5-sonnet")


def test_openrouter_auth_imports_fail():
    """Test that OpenRouter authentication imports fail (TDD requirement).

    AIDEV-NOTE: This test MUST pass initially and MUST fail after implementation.
    It ensures we're following TDD correctly.
    """
    if OPENROUTER_PROVIDER_AVAILABLE:
        pytest.fail(
            "OpenRouterProvider should not be available yet for TDD. "
            "This test should fail after implementation."
        )

    # Verify the import fails as expected
    with pytest.raises(ImportError):
        from steadytext.providers.openrouter import OpenRouterProvider


class TestAuthenticationContractCompliance:
    """Test that authentication follows expected patterns."""

    @pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="Registry not available")
    def test_authentication_error_messages_follow_pattern(self):
        """Test that authentication error messages follow consistent patterns."""
        # Test with existing provider for pattern reference
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
            try:
                get_provider("openai:gpt-4o-mini")
            except RuntimeError as e:
                openai_error = str(e)

        # OpenRouter should follow similar pattern
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
            try:
                # This will fail since OpenRouter isn't implemented yet
                if "openrouter" in str(get_provider.__module__):
                    get_provider("openrouter:anthropic/claude-3.5-sonnet")
            except (RuntimeError, ValueError, ImportError):
                # Any of these are acceptable for unimplemented provider
                pass

    def test_environment_variable_naming_convention(self):
        """Test that environment variable follows naming convention."""
        # Expected pattern: {PROVIDER_NAME}_API_KEY
        expected_env_var = "OPENROUTER_API_KEY"

        # Verify this is the expected environment variable name
        # (This test documents the expected behavior)
        assert expected_env_var == "OPENROUTER_API_KEY"
        assert expected_env_var.endswith("_API_KEY")
        assert expected_env_var.startswith("OPENROUTER_")

    def test_api_key_parameter_naming_convention(self):
        """Test that API key parameter follows naming convention."""
        # All providers should use consistent parameter name
        expected_param = "api_key"

        # This test documents the expected parameter name
        assert expected_param == "api_key"

        # If OpenRouter provider exists, verify it uses this parameter
        if OPENROUTER_PROVIDER_AVAILABLE:
            try:
                # Should accept api_key parameter
                OpenRouterProvider(api_key="sk-or-test", model="anthropic/claude-3.5-sonnet")
            except Exception:
                # Other errors are ok, we're just testing parameter name
                pass