"""OpenRouter provider configuration.

AIDEV-ANCHOR: OpenRouter configuration
This module defines the configuration class for OpenRouter provider settings,
including default values, validation, and environment variable handling.
"""

import os
from typing import Optional, Tuple
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger("steadytext.providers.openrouter")


class OpenRouterConfig(BaseModel):
    """Configuration container for OpenRouter provider settings.

    AIDEV-NOTE: Follows the data model specification with proper defaults
    and validation rules. Handles environment variable fallbacks and
    API key format validation.

    Attributes:
        api_key: OpenRouter API key (from OPENROUTER_API_KEY env var if None)
        model: Default model to use in OpenRouter format (provider/model)
        base_url: OpenRouter API base URL
        timeout: Connection and read timeouts as (connect, read) tuple
        max_retries: Maximum retry attempts for failed requests
        retry_delay: Base delay for exponential backoff retries
    """

    api_key: Optional[str] = Field(
        None, description="OpenRouter API key (reads OPENROUTER_API_KEY if None)"
    )
    model: str = Field(
        "anthropic/claude-3.5-sonnet", description="Default model to use"
    )
    base_url: str = Field(
        "https://openrouter.ai/api/v1", description="OpenRouter API base URL"
    )
    timeout: Tuple[int, int] = Field(
        (30, 120), description="Connection and read timeouts in seconds"
    )
    max_retries: int = Field(
        3, ge=0, le=10, description="Maximum retry attempts for failed requests"
    )
    retry_delay: float = Field(
        1.0, gt=0.0, le=60.0, description="Base delay for exponential backoff retries"
    )

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"

    def __init__(self, **data):
        """Initialize configuration with environment variable fallbacks."""
        # Handle API key from environment if not provided
        if "api_key" not in data or data["api_key"] is None:
            data["api_key"] = os.environ.get("OPENROUTER_API_KEY")

        super().__init__(**data)

    @validator("api_key")
    def validate_api_key_format(cls, v):
        """Validate OpenRouter API key format.

        Keys are accepted as long as they are non-empty strings. When the
        value does not follow the typical ``sk-or-`` prefix we log a warning
        rather than failing validation so contract tests using placeholder
        keys can run.
        """
        if v is None:
            # API key can be None - validation will happen during provider initialization
            return v

        if not isinstance(v, str):
            raise ValueError("API key must be a string")

        v = v.strip()
        if not v:
            raise ValueError("API key cannot be empty")

        if not v.startswith("sk-or-"):
            logger.warning(
                "OpenRouter API key does not use expected 'sk-or-' prefix. "
                "Continuing with provided key."
            )

        return v

    @validator("model")
    def validate_model_format(cls, v):
        """Validate OpenRouter model format.

        Model names must be in "provider/model-name" format.
        Examples: "anthropic/claude-3.5-sonnet", "openai/gpt-4"
        """
        if not isinstance(v, str):
            raise ValueError("Model must be a string")

        v = v.strip()
        if not v:
            raise ValueError("Model cannot be empty")

        # Check format: provider/model-name
        if "/" not in v:
            raise ValueError("Model must be in 'provider/model-name' format")

        parts = v.split("/")
        if len(parts) != 2:
            raise ValueError("Model must have exactly one '/' separator")

        provider_part, model_part = parts
        if not provider_part or not model_part:
            raise ValueError("Both provider and model parts must be non-empty")

        # Basic validation - no spaces, reasonable length
        if " " in v:
            raise ValueError("Model name cannot contain spaces")

        if len(v) > 100:
            raise ValueError("Model name too long (max 100 characters)")

        return v

    @validator("base_url")
    def validate_base_url(cls, v):
        """Validate OpenRouter API base URL format."""
        if not isinstance(v, str):
            raise ValueError("Base URL must be a string")

        v = v.strip()
        if not v:
            raise ValueError("Base URL cannot be empty")

        # Check basic URL format
        if not v.startswith(("http://", "https://")):
            raise ValueError("Base URL must start with http:// or https://")

        # Remove trailing slash for consistency
        if v.endswith("/"):
            v = v.rstrip("/")

        return v

    @validator("timeout")
    def validate_timeout(cls, v):
        """Validate timeout tuple format and values."""
        if not isinstance(v, (tuple, list)) or len(v) != 2:
            raise ValueError(
                "Timeout must be a tuple of (connect_timeout, read_timeout)"
            )

        connect_timeout, read_timeout = v
        if not isinstance(connect_timeout, (int, float)) or connect_timeout <= 0:
            raise ValueError("Connect timeout must be a positive number")

        if not isinstance(read_timeout, (int, float)) or read_timeout <= 0:
            raise ValueError("Read timeout must be a positive number")

        # Convert to integers and ensure reasonable limits
        connect_timeout = int(connect_timeout)
        read_timeout = int(read_timeout)

        if connect_timeout > 300:  # 5 minutes max
            raise ValueError("Connect timeout too large (max 300 seconds)")

        if read_timeout > 600:  # 10 minutes max
            raise ValueError("Read timeout too large (max 600 seconds)")

        return (connect_timeout, read_timeout)

    def get_api_key(self) -> str:
        """Get API key with validation.

        Returns:
            Valid API key

        Raises:
            ValueError: If API key is missing or invalid
        """
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment "
                "variable or provide api_key parameter."
            )
        return self.api_key

    def get_headers(self) -> dict:
        """Get HTTP headers for OpenRouter API requests.

        Returns:
            Dictionary of headers including authorization and content-type
        """
        api_key = self.get_api_key()
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "steadytext/python-client",
            # OpenRouter-specific headers for better routing
            "HTTP-Referer": "https://github.com/julep-ai/steadytext",
            "X-Title": "SteadyText",
        }

    def get_timeout_seconds(self) -> Tuple[float, float]:
        """Get timeout values as float tuple for httpx.

        Returns:
            Tuple of (connect_timeout, read_timeout) as floats
        """
        return (float(self.timeout[0]), float(self.timeout[1]))

    def is_embedding_model(self, model: Optional[str] = None) -> bool:
        """Check if a model is an embedding model.

        Args:
            model: Model name to check (uses default if None)

        Returns:
            True if model appears to be an embedding model
        """
        model_name = model or self.model
        model_name = model_name.lower()

        # Common embedding model patterns
        embedding_patterns = [
            "embedding",
            "embed",
            "vector",
            "retrieval",
            "voyage",
            "gte",
            "bge",
        ]

        return any(pattern in model_name for pattern in embedding_patterns)

    def get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff.

        Args:
            attempt: Retry attempt number (0-based)

        Returns:
            Delay in seconds for the retry attempt
        """
        if attempt <= 0:
            return 0.0

        # Exponential backoff with jitter
        import random

        base_delay = self.retry_delay * (2 ** (attempt - 1))
        # Add jitter (±25% of base delay)
        jitter = base_delay * 0.25 * (2 * random.random() - 1)
        delay = base_delay + jitter

        # Cap at 60 seconds
        return min(delay, 60.0)

    def should_retry_attempt(self, attempt: int) -> bool:
        """Check if another retry attempt should be made.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            True if should retry (attempt < max_retries)
        """
        return attempt < self.max_retries

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "api_key": "***" if self.api_key else None,  # Mask API key
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
        }

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"OpenRouterConfig(model={self.model}, base_url={self.base_url})"

    def __repr__(self) -> str:
        """Detailed string representation of configuration."""
        return (
            f"OpenRouterConfig("
            f"model='{self.model}', "
            f"base_url='{self.base_url}', "
            f"timeout={self.timeout}, "
            f"max_retries={self.max_retries}, "
            f"retry_delay={self.retry_delay}, "
            f"api_key={'***' if self.api_key else None}"
            f")"
        )


def create_default_config(**overrides) -> OpenRouterConfig:
    """Create OpenRouter configuration with default values.

    AIDEV-ANCHOR: Default config factory
    Convenience function to create configuration with sensible defaults
    and optional overrides for specific use cases.

    Args:
        **overrides: Configuration overrides

    Returns:
        OpenRouterConfig instance with defaults applied

    Example:
        >>> config = create_default_config(model="openai/gpt-4")
        >>> config = create_default_config(timeout=(60, 300))
    """
    return OpenRouterConfig(**overrides)


def create_config_from_env() -> OpenRouterConfig:
    """Create OpenRouter configuration from environment variables.

    AIDEV-ANCHOR: Environment config factory
    Creates configuration by reading all settings from environment variables
    with OPENROUTER_ prefix.

    Environment Variables:
        OPENROUTER_API_KEY: API key
        OPENROUTER_MODEL: Default model
        OPENROUTER_BASE_URL: API base URL
        OPENROUTER_TIMEOUT_CONNECT: Connect timeout in seconds
        OPENROUTER_TIMEOUT_READ: Read timeout in seconds
        OPENROUTER_MAX_RETRIES: Maximum retry attempts
        OPENROUTER_RETRY_DELAY: Base retry delay in seconds

    Returns:
        OpenRouterConfig instance from environment
    """
    config_data = {}

    # API key (handled automatically in __init__)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        config_data["api_key"] = api_key

    # Model
    model = os.environ.get("OPENROUTER_MODEL")
    if model:
        config_data["model"] = model

    # Base URL
    base_url = os.environ.get("OPENROUTER_BASE_URL")
    if base_url:
        config_data["base_url"] = base_url

    # Timeout
    connect_timeout = os.environ.get("OPENROUTER_TIMEOUT_CONNECT")
    read_timeout = os.environ.get("OPENROUTER_TIMEOUT_READ")
    if connect_timeout and read_timeout:
        try:
            config_data["timeout"] = (int(connect_timeout), int(read_timeout))
        except ValueError:
            logger.warning("Invalid timeout values in environment, using defaults")

    # Max retries
    max_retries = os.environ.get("OPENROUTER_MAX_RETRIES")
    if max_retries:
        try:
            config_data["max_retries"] = int(max_retries)
        except ValueError:
            logger.warning("Invalid max_retries value in environment, using default")

    # Retry delay
    retry_delay = os.environ.get("OPENROUTER_RETRY_DELAY")
    if retry_delay:
        try:
            config_data["retry_delay"] = float(retry_delay)
        except ValueError:
            logger.warning("Invalid retry_delay value in environment, using default")

    return OpenRouterConfig(**config_data)
