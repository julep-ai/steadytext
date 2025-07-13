"""
SteadyText: Deterministic text generation and embedding with zero configuration.

AIDEV-NOTE: Fixed "Never Fails" - embed() now catches TypeErrors & returns zero vectors
"""

# Version of the steadytext package - should match pyproject.toml
__version__ = "2.4.0"

# Import core functions and classes for public API
import os
import numpy as np
from typing import Optional, Any, Union, Tuple, Dict, Iterator, List, Type
from .core.generator import (
    core_generate as _generate,
    core_generate_iter as _generate_iter,
)
from .core.embedder import core_embed
from .core.reranker import core_rerank
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

# Import structured generation functions
from .core.structured import (
    generate_json,
    generate_regex,
    generate_choice,
    generate_format,
)


def generate(
    prompt: str,
    max_new_tokens: Optional[int] = None,
    return_logprobs: bool = False,
    eos_string: str = "[EOS]",
    model: Optional[str] = None,
    model_repo: Optional[str] = None,
    model_filename: Optional[str] = None,
    size: Optional[str] = None,
    seed: int = DEFAULT_SEED,
    response_format: Optional[Dict[str, Any]] = None,
    schema: Optional[Union[Dict[str, Any], Type, object]] = None,
    regex: Optional[str] = None,
    choices: Optional[List[str]] = None,
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]], None, Tuple[None, None]]:
    """Generate text deterministically from a prompt with optional structured output.

    Args:
        prompt: The input prompt to generate from
        max_new_tokens: The maximum number of new tokens to generate.
        return_logprobs: If True, a tuple (text, logprobs) is returned
        eos_string: Custom end-of-sequence string. "[EOS]" means use model's default.
                   Otherwise, generation stops when this string is encountered.
        model: Model name from registry (e.g., "gemma-3n-2b")
        model_repo: Custom Hugging Face repository ID
        model_filename: Custom model filename
        size: Size identifier ("small", "large")
        seed: Seed for deterministic generation
        response_format: Dict specifying output format (e.g., {"type": "json_object"})
        schema: JSON schema, Pydantic model, or Python type for structured output
        regex: Regular expression pattern for output format
        choices: List of allowed string choices for output

    Returns:
        Generated text string, or tuple (text, logprobs) if return_logprobs=True
        For structured output, JSON is wrapped in <json-output> tags

    Examples:
        # Use default model
        text = generate("Hello, world!")

        # Use size parameter
        text = generate("Quick response", size="small")

        # Use a model from the registry
        text = generate("Explain quantum computing", model="gemma-3n-2b")

        # Use a custom model
        text = generate(
            "Write a poem",
            model_repo="ggml-org/gemma-3n-E4B-it-GGUF",
            model_filename="gemma-3n-E4B-it-Q8_0.gguf"
        )

        # Generate JSON with schema
        from pydantic import BaseModel
        class Person(BaseModel):
            name: str
            age: int

        result = generate("Create a person", schema=Person)
        # Returns: "Let me create...<json-output>{"name": "John", "age": 30}</json-output>"

        # Generate with regex pattern
        phone = generate("My phone is", regex=r"\d{3}-\d{3}-\d{4}")

        # Generate with choices
        answer = generate("Is Python good?", choices=["yes", "no", "maybe"])
    """
    # AIDEV-NOTE: This is the primary public API. It orchestrates the daemon-first logic.
    # If the daemon is enabled (default), it attempts to use the client.
    # On any ConnectionError, it transparently falls back to direct, in-process generation.
    if os.environ.get("STEADYTEXT_DISABLE_DAEMON") != "1":
        client = get_daemon_client()
        if client is not None:
            try:
                logger.debug("Attempting to use daemon for text generation")
                return client.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    return_logprobs=return_logprobs,
                    eos_string=eos_string,
                    model=model,
                    model_repo=model_repo,
                    model_filename=model_filename,
                    size=size,
                    seed=seed,
                    response_format=response_format,
                    schema=schema,
                    regex=regex,
                    choices=choices,
                )
            except ConnectionError as e:
                # Fall back to direct generation
                logger.debug(
                    f"Daemon not available ({e}), falling back to direct generation"
                )

    result = _generate(
        prompt,
        max_new_tokens=max_new_tokens,
        return_logprobs=return_logprobs,
        eos_string=eos_string,
        model=model,
        model_repo=model_repo,
        model_filename=model_filename,
        size=size,
        seed=seed,
        response_format=response_format,
        schema=schema,
        regex=regex,
        choices=choices,
    )

    # AIDEV-NOTE: Return None if generation failed
    if result is None:
        logger.error("Text generation failed - model not available or invalid input")

    return result


def generate_iter(
    prompt: str,
    max_new_tokens: Optional[int] = None,
    eos_string: str = "[EOS]",
    include_logprobs: bool = False,
    model: Optional[str] = None,
    model_repo: Optional[str] = None,
    model_filename: Optional[str] = None,
    size: Optional[str] = None,
    seed: int = DEFAULT_SEED,
) -> Iterator[Union[str, Dict[str, Any]]]:
    """Generate text iteratively, yielding tokens as they are produced.

    This function streams tokens as they are generated, useful for real-time
    output or when you want to process tokens as they arrive. Falls back to
    yielding words from deterministic output when model is unavailable.

    Args:
        prompt: The input prompt to generate from
        max_new_tokens: The maximum number of new tokens to generate.
        eos_string: Custom end-of-sequence string. "[EOS]" means use model's default.
                   Otherwise, generation stops when this string is encountered.
        include_logprobs: If True, yield dicts with token and logprob info
        model: Model name from registry (e.g., "gemma-3n-2b")
        model_repo: Custom Hugging Face repository ID
        model_filename: Custom model filename
        size: Size identifier ("small", "large")

    Yields:
        str: Generated tokens/words as they are produced (if include_logprobs=False)
        dict: Token info with 'token' and 'logprobs' keys (if include_logprobs=True)
    """
    # AIDEV-NOTE: Use daemon by default for streaming unless explicitly disabled
    if os.environ.get("STEADYTEXT_DISABLE_DAEMON") != "1":
        client = get_daemon_client()
        if client is not None:
            try:
                logger.debug("Attempting to use daemon for streaming text generation")
                yield from client.generate_iter(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    eos_string=eos_string,
                    include_logprobs=include_logprobs,
                    model=model,
                    model_repo=model_repo,
                    model_filename=model_filename,
                    size=size,
                    seed=seed,
                )
                return
            except ConnectionError as e:
                # Fall back to direct generation
                logger.debug(
                    f"Daemon not available ({e}), falling back to direct streaming generation"
                )

    yield from _generate_iter(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_string=eos_string,
        include_logprobs=include_logprobs,
        model=model,
        model_repo=model_repo,
        model_filename=model_filename,
        size=size,
        seed=seed,
    )


def embed(
    text_input: Union[str, List[str]], seed: int = DEFAULT_SEED
) -> Optional[np.ndarray]:
    """Create embeddings for text input."""
    # AIDEV-NOTE: Use daemon by default for embeddings unless explicitly disabled
    if os.environ.get("STEADYTEXT_DISABLE_DAEMON") != "1":
        client = get_daemon_client()
        if client is not None:
            try:
                return client.embed(text_input, seed=seed)
            except ConnectionError:
                # Fall back to direct embedding
                logger.debug("Daemon not available, falling back to direct embedding")

    try:
        result = core_embed(text_input, seed=seed)
        if result is None:
            logger.error(
                "Embedding creation failed - model not available or invalid input"
            )
        return result
    except TypeError as e:
        logger.error(f"Invalid input type for embedding: {e}")
        return None


def rerank(
    query: str,
    documents: Union[str, List[str]],
    task: str = "Given a web search query, retrieve relevant passages that answer the query",
    return_scores: bool = True,
    seed: int = DEFAULT_SEED,
) -> Union[List[Tuple[str, float]], List[str]]:
    """Rerank documents based on relevance to a query.

    Uses the Qwen3-Reranker model to score query-document pairs and returns
    documents sorted by relevance. Falls back to simple word overlap scoring
    when the model is unavailable.

    Args:
        query: The search query
        documents: Single document or list of documents to rerank
        task: Task description for the reranking (affects scoring)
        return_scores: If True, return (document, score) tuples; if False, just documents
        seed: Random seed for determinism

    Returns:
        If return_scores=True: List of (document, score) tuples sorted by score descending
        If return_scores=False: List of documents sorted by relevance descending
        Empty list if no documents provided or on error

    Examples:
        # Basic reranking with scores
        results = rerank("What is Python?", [
            "Python is a programming language",
            "Snakes are reptiles",
            "Java is also a programming language"
        ])
        # Returns: [("Python is a programming language", 0.95), ...]

        # Get just sorted documents
        docs = rerank("climate change", documents, return_scores=False)

        # Custom task description
        results = rerank(
            "symptoms of flu",
            medical_documents,
            task="Given a medical query, find relevant clinical information"
        )
    """
    # AIDEV-NOTE: Use daemon by default for reranking unless explicitly disabled
    if os.environ.get("STEADYTEXT_DISABLE_DAEMON") != "1":
        client = get_daemon_client()
        if client is not None:
            try:
                return client.rerank(
                    query=query,
                    documents=documents,
                    task=task,
                    return_scores=return_scores,
                    seed=seed,
                )
            except ConnectionError:
                # Fall back to direct reranking
                logger.debug("Daemon not available, falling back to direct reranking")

    try:
        result = core_rerank(
            query=query,
            documents=documents,
            task=task,
            return_scores=return_scores,
            seed=seed,
        )
        if result is None:
            logger.error("Reranking failed - model not available or invalid input")
            return []
        return result
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        return []


def preload_models(verbose: bool = False, size: Optional[str] = None):
    """Preload models to ensure they're available for generation and embedding.

    Args:
        verbose: Whether to log progress messages
        size: Model size to preload ("small", "medium", "large")
    """
    # AIDEV-NOTE: Skip model loading if STEADYTEXT_SKIP_MODEL_LOAD is set
    # This prevents hanging during tests when models aren't available
    if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
        if verbose:
            logger.info("Model preloading skipped (STEADYTEXT_SKIP_MODEL_LOAD=1)")
        return

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
    "rerank",
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
    # Structured generation
    "generate_json",
    "generate_regex",
    "generate_choice",
    "generate_format",
]
