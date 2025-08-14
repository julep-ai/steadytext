"""Registry for remote model providers.

AIDEV-NOTE: Central registry for managing remote providers and model routing.
"""

import os
from typing import Optional, Dict, Type, List, Tuple
import logging

from .base import RemoteModelProvider
from .openai import OpenAIProvider
from .cerebras import CerebrasProvider
from .voyageai import VoyageAIProvider
from .jina import JinaProvider

logger = logging.getLogger("steadytext.providers.registry")

# AIDEV-NOTE: Registry of available providers
PROVIDER_REGISTRY: Dict[str, Type[RemoteModelProvider]] = {
    "openai": OpenAIProvider,
    "cerebras": CerebrasProvider,
    "voyageai": VoyageAIProvider,
    "jina": JinaProvider,
}


def is_unsafe_mode_enabled() -> bool:
    """Check if unsafe mode is enabled via environment variable."""
    return os.environ.get("STEADYTEXT_UNSAFE_MODE", "false").lower() in [
        "true",
        "1",
        "yes",
    ]


def is_remote_model(model: Optional[str]) -> bool:
    """Check if a model string refers to a remote model.

    Remote models are specified as "provider:model" (e.g., "openai:gpt-4").

    Args:
        model: Model string to check

    Returns:
        True if model is a remote model specification
    """
    if not model:
        return False

    # Check if model contains provider prefix
    if ":" in model:
        provider_name = model.split(":", 1)[0]
        return provider_name in PROVIDER_REGISTRY

    return False


def parse_remote_model(model: str) -> Tuple[str, str]:
    """Parse remote model string into provider and model name.

    Args:
        model: Model string like "openai:gpt-4o-mini"

    Returns:
        Tuple of (provider_name, model_name)

    Raises:
        ValueError: If model string is invalid
    """
    if ":" not in model:
        raise ValueError(
            f"Invalid remote model format: {model}. "
            f"Expected format: provider:model (e.g., openai:gpt-4o-mini)"
        )

    provider_name, model_name = model.split(":", 1)

    if provider_name not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown provider: {provider_name}. Available providers: {available}"
        )

    return provider_name, model_name


def get_provider(model: str, api_key: Optional[str] = None) -> RemoteModelProvider:
    """Get a remote model provider instance.

    Args:
        model: Model string like "openai:gpt-4o-mini"
        api_key: Optional API key (uses env vars if not provided)

    Returns:
        Configured provider instance

    Raises:
        ValueError: If provider not found or model format invalid
        RuntimeError: If unsafe mode not enabled
    """
    if not is_unsafe_mode_enabled():
        raise RuntimeError(
            "Remote models require unsafe mode. "
            "Set STEADYTEXT_UNSAFE_MODE=true to enable. "
            "WARNING: Remote models provide only best-effort determinism!"
        )

    provider_name, model_name = parse_remote_model(model)

    # AIDEV-NOTE: Early API key validation to fail fast without importing heavy dependencies
    actual_key = api_key  # Default to passed api_key
    if provider_name == "openai":
        actual_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not actual_key:
            raise RuntimeError(
                f"Provider {provider_name} is not available. Check API key and dependencies."
            )
    elif provider_name == "cerebras":
        actual_key = api_key or os.environ.get("CEREBRAS_API_KEY")
        if not actual_key:
            raise RuntimeError(
                f"Provider {provider_name} is not available. Check API key and dependencies."
            )
    elif provider_name == "voyageai":
        actual_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not actual_key:
            raise RuntimeError(
                f"Provider {provider_name} is not available. Check API key and dependencies."
            )
    elif provider_name == "jina":
        actual_key = api_key or os.environ.get("JINA_API_KEY")
        if not actual_key:
            raise RuntimeError(
                f"Provider {provider_name} is not available. Check API key and dependencies."
            )

    provider_class = PROVIDER_REGISTRY[provider_name]

    # Handle provider-specific constructors
    # AIDEV-NOTE: Pass the resolved actual_key to ensure consistency
    if provider_name in ["openai", "cerebras", "voyageai", "jina"]:
        # These providers accept a model parameter
        provider = provider_class(api_key=actual_key, model=model_name)  # type: ignore[call-arg]
    else:
        # Future providers might only need api_key
        provider = provider_class(api_key=actual_key)

    if not provider.is_available():
        raise RuntimeError(
            f"Provider {provider_name} is not available. "
            f"Check API key and dependencies."
        )

    return provider


def list_providers() -> List[str]:
    """Get list of available provider names."""
    return list(PROVIDER_REGISTRY.keys())


def list_remote_models() -> Dict[str, List[str]]:
    """Get all available remote models grouped by provider.

    Returns:
        Dict mapping provider names to lists of supported models
    """
    models = {}

    for provider_name, provider_class in PROVIDER_REGISTRY.items():
        try:
            # Create temporary instance to get model list
            provider = provider_class()
            models[provider_name] = provider.get_supported_models()
        except Exception as e:
            logger.warning(f"Failed to get models for {provider_name}: {e}")
            models[provider_name] = []

    return models
