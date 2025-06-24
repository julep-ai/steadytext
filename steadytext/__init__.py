"""
SteadyText: Deterministic text generation and embedding with zero configuration.

AIDEV-NOTE: Fixed "Never Fails" - embed() now catches TypeErrors & returns zero vectors
"""

# Version of the steadytext package - should match pyproject.toml
__version__ = "1.3.5"

# Import core functions and classes for public API
import os
import numpy as np
from typing import Optional, Any, Union, Tuple, Dict, Iterator
from .core.generator import generate as _generate, generate_iter as _generate_iter
from .core.embedder import create_embedding
from .utils import (
    logger,
    DEFAULT_SEED,
    GENERATION_MAX_NEW_TOKENS,
    EMBEDDING_DIMENSION,
    get_cache_dir,
)
from .models.loader import get_generator_model_instance, get_embedding_model_instance
from .daemon.client import DaemonClient, use_daemon, get_daemon_client
from .cache_manager import get_cache_manager


def generate(
    prompt: str,
    return_logprobs: bool = False,
    eos_string: str = "[EOS]",
    model: Optional[str] = None,
    model_repo: Optional[str] = None,
    model_filename: Optional[str] = None,
    size: Optional[str] = None,
    thinking_mode: bool = False,
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]]]:
    """Generate text deterministically from a prompt.

    Args:
        prompt: The input prompt to generate from
        return_logprobs: If True, a tuple (text, logprobs) is returned
        eos_string: Custom end-of-sequence string. "[EOS]" means use model's default.
                   Otherwise, generation stops when this string is encountered.
        model: Model name from registry (e.g., "qwen2.5-3b", "qwen3-8b")
        model_repo: Custom Hugging Face repository ID
        model_filename: Custom model filename
        size: Size identifier ("small", "medium", "large") - maps to Qwen3 0.6B/1.7B/4B models
        thinking_mode: Enable Qwen3 thinking mode (default: False appends /no_think)

    Returns:
        Generated text string, or tuple (text, logprobs) if return_logprobs=True

    Examples:
        # Use default model
        text = generate("Hello, world!")

        # Use size parameter
        text = generate("Quick response", size="small")

        # Use a model from the registry
        text = generate("Explain quantum computing", model="qwen2.5-3b")

        # Use a custom model
        text = generate(
            "Write a poem",
            model_repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
            model_filename="qwen2.5-7b-instruct-q8_0.gguf"
        )
    """
    # AIDEV-NOTE: Use daemon by default unless explicitly disabled
    if os.environ.get("STEADYTEXT_DISABLE_DAEMON") != "1":
        client = get_daemon_client()
        if client is not None:
            try:
                return client.generate(
                    prompt=prompt,
                    return_logprobs=return_logprobs,
                    eos_string=eos_string,
                    model=model,
                    model_repo=model_repo,
                    model_filename=model_filename,
                    size=size,
                    thinking_mode=thinking_mode,
                )
            except ConnectionError:
                # Fall back to direct generation
                logger.debug("Daemon not available, falling back to direct generation")

    return _generate(
        prompt,
        return_logprobs=return_logprobs,
        eos_string=eos_string,
        model=model,
        model_repo=model_repo,
        model_filename=model_filename,
        size=size,
        thinking_mode=thinking_mode,
    )


def generate_iter(
    prompt: str,
    eos_string: str = "[EOS]",
    include_logprobs: bool = False,
    model: Optional[str] = None,
    model_repo: Optional[str] = None,
    model_filename: Optional[str] = None,
    size: Optional[str] = None,
    thinking_mode: bool = False,
) -> Iterator[Union[str, Dict[str, Any]]]:
    """Generate text iteratively, yielding tokens as they are produced.

    This function streams tokens as they are generated, useful for real-time
    output or when you want to process tokens as they arrive. Falls back to
    yielding words from deterministic output when model is unavailable.

    Args:
        prompt: The input prompt to generate from
        eos_string: Custom end-of-sequence string. "[EOS]" means use model's default.
                   Otherwise, generation stops when this string is encountered.
        include_logprobs: If True, yield dicts with token and logprob info
        model: Model name from registry (e.g., "qwen2.5-3b")
        model_repo: Custom Hugging Face repository ID
        model_filename: Custom model filename
        size: Size identifier ("small", "medium", "large") - maps to Qwen3 0.6B/1.7B/4B models
        thinking_mode: Enable Qwen3 thinking mode (default: False appends /no_think)

    Yields:
        str: Generated tokens/words as they are produced (if include_logprobs=False)
        dict: Token info with 'token' and 'logprobs' keys (if include_logprobs=True)
    """
    # AIDEV-NOTE: Use daemon by default for streaming unless explicitly disabled
    if os.environ.get("STEADYTEXT_DISABLE_DAEMON") != "1":
        client = get_daemon_client()
        if client is not None:
            try:
                yield from client.generate_iter(
                    prompt=prompt,
                    eos_string=eos_string,
                    include_logprobs=include_logprobs,
                    model=model,
                    model_repo=model_repo,
                    model_filename=model_filename,
                    size=size,
                    thinking_mode=thinking_mode,
                )
                return
            except ConnectionError:
                # Fall back to direct generation
                logger.debug(
                    "Daemon not available, falling back to direct streaming generation"
                )

    yield from _generate_iter(
        prompt,
        eos_string=eos_string,
        include_logprobs=include_logprobs,
        model=model,
        model_repo=model_repo,
        model_filename=model_filename,
        size=size,
        thinking_mode=thinking_mode,
    )


def embed(text_input) -> np.ndarray:
    """Create embeddings for text input."""
    # AIDEV-NOTE: Use daemon by default for embeddings unless explicitly disabled
    if os.environ.get("STEADYTEXT_DISABLE_DAEMON") != "1":
        client = get_daemon_client()
        if client is not None:
            try:
                return client.embed(text_input)
            except ConnectionError:
                # Fall back to direct embedding
                logger.debug("Daemon not available, falling back to direct embedding")

    try:
        return create_embedding(text_input)
    except TypeError as e:
        logger.error(f"Invalid input type for embedding: {e}")
        return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)


def preload_models(verbose: bool = False, size: Optional[str] = None):
    """Preload models to ensure they're available for generation and embedding.

    Args:
        verbose: Whether to log progress messages
        size: Model size to preload ("small", "medium", "large")
    """
    if verbose:
        if size:
            logger.info(f"Preloading {size} generator model...")
        else:
            logger.info("Preloading generator model...")

    # If size is specified, preload that specific model
    if size:
        from .utils import resolve_model_params

        repo_id, filename = resolve_model_params(size=size)
        # Force the model to load by doing a dummy generation
        generate("test", size=size)
    else:
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
    "use_daemon",
    "DaemonClient",
    "get_cache_manager",
    "DEFAULT_SEED",
    "GENERATION_MAX_NEW_TOKENS",
    "EMBEDDING_DIMENSION",
    "logger",
    "__version__",
]
