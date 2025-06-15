"""
SteadyText: Deterministic text generation and embedding with zero configuration.

AIDEV-NOTE: Fixed "Never Fails" - embed() now catches TypeErrors & returns zero vectors
"""

# Version of the steadytext package - should match pyproject.toml
__version__ = "0.1.0"

# Import core functions and classes for public API
import numpy as np
from typing import Optional, Any, Union, Tuple, Dict, Iterator
from .core.generator import DeterministicGenerator
from .core.embedder import create_embedding
from .utils import (
    logger,
    DEFAULT_SEED,
    GENERATION_MAX_NEW_TOKENS,
    EMBEDDING_DIMENSION,
    get_cache_dir,
)
from .models.loader import get_generator_model_instance, get_embedding_model_instance

# Create a global generator instance for the public API
_global_generator = DeterministicGenerator()


def generate(
    prompt: str, return_logprobs: bool = False
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]:
    """Generate text deterministically from a prompt.

    If ``return_logprobs`` is True, a tuple ``(text, logprobs)`` is returned.
    """
    return _global_generator.generate(
        prompt, return_logprobs=return_logprobs
    )


def generate_iter(
    prompt: str
) -> Iterator[str]:
    """Generate text iteratively, yielding tokens as they are produced.
    
    This function streams tokens as they are generated, useful for real-time
    output or when you want to process tokens as they arrive. Falls back to
    yielding words from deterministic output when model is unavailable.
    
    Args:
        prompt: The input prompt to generate from
        
    Yields:
        str: Generated tokens/words as they are produced
    """
    return _global_generator.generate_iter(prompt)


def embed(text_input) -> np.ndarray:
    """Create embeddings for text input."""
    try:
        return create_embedding(text_input)
    except TypeError as e:
        logger.error(f"Invalid input type for embedding: {e}")
        return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)


def preload_models(verbose: bool = False):
    """Preload models to ensure they're available for generation and embedding."""
    if verbose:
        logger.info("Preloading generator model...")
    get_generator_model_instance()

    if verbose:
        logger.info("Preloading embedding model...")
    get_embedding_model_instance()

    if verbose:
        logger.info("Model preloading completed.")


def get_model_cache_dir() -> str:
    """Get the model cache directory path as a string."""
    return str(get_cache_dir())


# Export public API
__all__ = [
    "generate",
    "generate_iter",
    "embed",
    "preload_models",
    "get_model_cache_dir",
    "DEFAULT_SEED",
    "GENERATION_MAX_NEW_TOKENS",
    "EMBEDDING_DIMENSION",
    "logger",
    "__version__",
]
