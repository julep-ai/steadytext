"""Base class for remote model providers.

AIDEV-NOTE: Remote models provide best-effort determinism via seed parameters
but cannot guarantee the same reproducibility as local GGUF models.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Iterator, Dict, Any, List, Union
import logging

logger = logging.getLogger("steadytext.providers")


class UnsafeModeWarning(UserWarning):
    """Warning issued when using remote models with best-effort determinism."""

    pass


class RemoteModelProvider(ABC):
    """Abstract base class for remote model providers.

    AIDEV-NOTE: All providers must implement seed-based generation for best-effort
    determinism. Providers should document their determinism limitations clearly.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize provider with optional API key.

        Args:
            api_key: API key for the provider. If None, will try to read from
                    environment variables specific to each provider.
        """
        self.api_key = api_key
        self._warned = False

        # Validate API key format if provided
        if self.api_key and not self._is_valid_api_key_format(self.api_key):
            logger.warning(
                f"API key format may be invalid for {self.__class__.__name__}. "
                f"Please check your API key."
            )

    def _issue_warning(self):
        """Issue a warning about best-effort determinism if not already warned."""
        if not self._warned:
            warning_msg = (
                f"UNSAFE MODE WARNING: Using {self.provider_name} remote model\n"
                f"For TRUE determinism, use local GGUF models (default SteadyText behavior).\n"
            )
            warnings.warn(warning_msg, UnsafeModeWarning, stacklevel=3)
            self._warned = True

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name for display."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available (API key set, dependencies installed)."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
        schema: Optional[Union[Dict[str, Any], type, object]] = None,
        **kwargs,
    ) -> str:
        """Generate text with best-effort determinism using seed.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            seed: Seed for best-effort determinism
            temperature: Temperature for sampling (0.0 = deterministic, higher = more random)
            response_format: Response format specification (e.g., {"type": "json_object"})
            schema: JSON schema, Pydantic model, or Python type for structured output
            **kwargs: Additional provider-specific parameters (passed directly to the API)
                     These can include custom options like top_p, presence_penalty, etc.

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_iter(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        seed: int = 42,
        temperature: float = 0.0,
        **kwargs,
    ) -> Iterator[str]:
        """Generate text iteratively with best-effort determinism.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            seed: Seed for best-effort determinism
            temperature: Temperature for sampling (0.0 = deterministic, higher = more random)
            **kwargs: Additional provider-specific parameters (passed directly to the API)
                     These can include custom options like top_p, presence_penalty, etc.

        Yields:
            Generated tokens/chunks
        """
        pass

    def embed(
        self, text: Union[str, List[str]], seed: int = 42, **kwargs
    ) -> Optional[Any]:
        """Generate embeddings if supported by provider.

        Most providers don't support seeded embeddings.

        Args:
            text: Text or list of texts to embed
            seed: Seed (may be ignored by provider)
            **kwargs: Additional provider-specific parameters

        Returns:
            Embeddings or None if not supported
        """
        logger.warning(f"{self.provider_name} does not support embeddings")
        return None

    def supports_embeddings(self) -> bool:
        """Check if provider supports embeddings."""
        return False

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming generation."""
        return True

    def get_supported_models(self) -> List[str]:
        """Get list of supported model names."""
        return []

    def _is_valid_api_key_format(self, api_key: str) -> bool:
        """Validate API key format (can be overridden by providers).

        Base implementation just checks for non-empty string.
        Providers can override for specific format validation.

        Args:
            api_key: API key to validate

        Returns:
            True if format appears valid
        """
        return bool(api_key and api_key.strip())
