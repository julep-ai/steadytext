"""Contract definition for OpenRouter provider implementation.

This contract defines the expected interface and behavior for the OpenRouter
provider class, following the RemoteModelProvider protocol.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, List, Dict, Any, Union
import numpy as np


class OpenRouterProviderContract(ABC):
    """Contract for OpenRouter provider implementation."""

    @abstractmethod
    def __init__(
        self, api_key: Optional[str] = None, model: str = "anthropic/claude-3.5-sonnet"
    ):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key (reads OPENROUTER_API_KEY if None)
            model: Default model to use (OpenRouter format: provider/model)

        Raises:
            RuntimeError: If API key is missing or invalid
            ValueError: If model format is invalid
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if OpenRouter provider is available.

        Returns:
            True if API key is valid and service is reachable
        """
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Get list of supported OpenRouter models.

        Returns:
            List of model names in OpenRouter format (provider/model)
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Iterator[str]]:
        """Generate text using OpenRouter API.

        Args:
            prompt: Input text prompt
            model: Model to use (overrides instance default)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter (0-1)
            stream: Whether to stream response
            **kwargs: Additional OpenRouter-specific parameters

        Returns:
            Generated text (string) or streaming iterator

        Raises:
            OpenRouterAuthError: If authentication fails
            OpenRouterRateLimitError: If rate limited
            OpenRouterModelError: If model is invalid or unavailable
            RuntimeError: For other API errors (with fallback to deterministic)
        """
        pass

    @abstractmethod
    def embed(
        self, texts: Union[str, List[str]], *, model: Optional[str] = None, **kwargs
    ) -> np.ndarray:
        """Generate embeddings using OpenRouter API.

        Args:
            texts: Text(s) to embed
            model: Embedding model to use
            **kwargs: Additional OpenRouter-specific parameters

        Returns:
            NumPy array of embeddings (2D for multiple texts)

        Raises:
            OpenRouterAuthError: If authentication fails
            OpenRouterModelError: If embedding model is invalid
            RuntimeError: For other API errors (with fallback to deterministic)
        """
        pass

    @abstractmethod
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific OpenRouter model.

        Args:
            model: Model name in OpenRouter format

        Returns:
            Dictionary with model information (pricing, capabilities, etc.)

        Raises:
            OpenRouterModelError: If model not found
        """
        pass


class OpenRouterConfigContract:
    """Contract for OpenRouter configuration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: tuple = (30, 120),
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize OpenRouter configuration.

        Args:
            api_key: API key (reads OPENROUTER_API_KEY if None)
            base_url: OpenRouter API base URL
            timeout: (connect_timeout, read_timeout) in seconds
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Base delay for exponential backoff retries
        """
        pass


class OpenRouterResponseContract:
    """Contract for OpenRouter API response parsing."""

    @abstractmethod
    def parse_chat_completion(self, response_data: dict) -> str:
        """Parse chat completion response to extract generated text.

        Args:
            response_data: Raw OpenRouter API response

        Returns:
            Generated text content

        Raises:
            ValueError: If response format is invalid
        """
        pass

    @abstractmethod
    def parse_embedding_response(self, response_data: dict) -> np.ndarray:
        """Parse embedding response to extract vectors.

        Args:
            response_data: Raw OpenRouter API response

        Returns:
            NumPy array of embedding vectors

        Raises:
            ValueError: If response format is invalid
        """
        pass


class OpenRouterErrorContract:
    """Contract for OpenRouter error handling."""

    @abstractmethod
    def handle_http_error(self, status_code: int, response_data: dict) -> Exception:
        """Map HTTP errors to appropriate exception types.

        Args:
            status_code: HTTP status code
            response_data: Error response from OpenRouter

        Returns:
            Appropriate exception instance
        """
        pass

    @abstractmethod
    def should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry.

        Args:
            error: Exception that occurred

        Returns:
            True if retry should be attempted
        """
        pass


# Registry Integration Contract
def register_openrouter_provider() -> None:
    """Register OpenRouter provider with the provider registry.

    This function should be called during module initialization to add
    OpenRouter to the PROVIDER_REGISTRY dictionary.

    Expected integration:
    - Add "openrouter" key to PROVIDER_REGISTRY
    - Add OPENROUTER_API_KEY validation to get_provider()
    - Update list_providers() to include OpenRouter
    - Update list_remote_models() to include OpenRouter models
    """
    pass


# CLI Integration Contract
def add_openrouter_cli_support() -> None:
    """Ensure CLI commands support OpenRouter provider.

    Expected behavior:
    - `st generate --model openrouter:anthropic/claude-3.5-sonnet "Hello"`
    - `st embed --model openrouter:openai/text-embedding-3-small "Text"`
    - Error messages include OpenRouter in provider suggestions
    - Help text mentions OpenRouter in examples
    """
    pass


# Error Hierarchy Contract
class OpenRouterError(RuntimeError):
    """Base exception for OpenRouter provider errors."""

    pass


class OpenRouterAuthError(OpenRouterError):
    """Authentication/authorization errors."""

    pass


class OpenRouterRateLimitError(OpenRouterError):
    """Rate limiting errors."""

    pass


class OpenRouterModelError(OpenRouterError):
    """Model-related errors."""

    pass


class OpenRouterTimeoutError(OpenRouterError):
    """Request timeout errors."""

    pass


class OpenRouterConnectionError(OpenRouterError):
    """Network connection errors."""

    pass
