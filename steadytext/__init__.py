"""
SteadyText: Deterministic text generation and embedding with zero configuration.

AIDEV-NOTE: Fixed "Never Fails" - embed() now catches TypeErrors & returns zero vectors
"""

# Version of the steadytext package - should match pyproject.toml
# AIDEV-NOTE: Always update this when bumping the lib version
# AIDEV-NOTE: Using date-based versioning (yyyy.mm.dd) as of 2025.8.15
__version__ = "2025.8.27"

# Import core functions and classes for public API
import os
import numpy as np
from typing import (
    Optional,
    Any,
    Union,
    Tuple,
    Dict,
    Iterator,
    List,
    Type,
)
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
    generate_pydantic,
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
    temperature: float = 0.0,
    response_format: Optional[Dict[str, Any]] = None,
    schema: Optional[Union[Dict[str, Any], Type, object]] = None,
    regex: Optional[str] = None,
    choices: Optional[List[str]] = None,
    unsafe_mode: bool = False,
    return_pydantic: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> Union[str, Tuple[str, Optional[Dict[str, Any]]], None, Tuple[None, None], object]:
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
        temperature: Temperature for sampling (0.0 = deterministic, higher = more random)
        response_format: Dict specifying output format (e.g., {"type": "json_object"})
        schema: JSON schema, Pydantic model, or Python type for structured output
        regex: Regular expression pattern for output format
        choices: List of allowed string choices for output
        unsafe_mode: Enable remote models with best-effort determinism (non-deterministic)
        return_pydantic: If True and schema is a Pydantic model, return the instantiated model
        options: Additional provider-specific options (for remote models)

    Returns:
        Generated text string, or tuple (text, logprobs) if return_logprobs=True
        For structured output, JSON is wrapped in <json-output> tags
        If return_pydantic=True and schema is a Pydantic model, returns the instantiated model

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

        # Or get a Pydantic model instance directly
        person = generate("Create a person", schema=Person, return_pydantic=True)
        # Returns: Person(name="John", age=30)

        # Generate with regex pattern
        phone = generate("My phone is", regex=r"\\d{3}-\\d{3}-\\d{4}")

        # Generate with choices
        answer = generate("Is Python good?", choices=["yes", "no", "maybe"])
    """
    # AIDEV-NOTE: This is the primary public API. It orchestrates the daemon-first logic.
    # If the daemon is enabled (default), it attempts to use the client.
    # On any ConnectionError, it transparently falls back to direct, in-process generation.
    # Skip daemon for remote models to avoid unnecessary delays
    is_remote = model and ":" in model
    if os.environ.get("STEADYTEXT_DISABLE_DAEMON") != "1" and not is_remote:
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
                    temperature=temperature,
                    response_format=response_format,
                    schema=schema,
                    regex=regex,
                    choices=choices,
                    unsafe_mode=unsafe_mode,
                    return_pydantic=return_pydantic,
                    options=options,
                )
            except ConnectionError as e:
                # Fall back to direct generation
                logger.info(
                    f"Daemon not available ({e}), falling back to direct generation. "
                    "For better performance, start the daemon with 'st daemon start'"
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
        temperature=temperature,
        response_format=response_format,
        schema=schema,
        regex=regex,
        choices=choices,
        unsafe_mode=unsafe_mode,
        return_pydantic=return_pydantic,
        options=options,
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
    temperature: float = 0.0,
    unsafe_mode: bool = False,
    options: Optional[Dict[str, Any]] = None,
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
        temperature: Temperature for sampling (0.0 = deterministic, higher = more random)
        unsafe_mode: Enable remote models with best-effort determinism (non-deterministic)
        options: Additional provider-specific options (for remote models)

    Yields:
        str: Generated tokens/words as they are produced (if include_logprobs=False)
        dict: Token info with 'token' and 'logprobs' keys (if include_logprobs=True)
    """
    # AIDEV-NOTE: Use daemon by default for streaming unless explicitly disabled
    # Skip daemon for remote models to avoid unnecessary delays
    is_remote = model and ":" in model
    if os.environ.get("STEADYTEXT_DISABLE_DAEMON") != "1" and not is_remote:
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
                    temperature=temperature,
                    unsafe_mode=unsafe_mode,
                    options=options,
                )
                return
            except ConnectionError as e:
                # Fall back to direct generation
                logger.info(
                    f"Daemon not available ({e}), falling back to direct streaming generation. "
                    "For better performance, start the daemon with 'st daemon start'"
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
        temperature=temperature,
        unsafe_mode=unsafe_mode,
        options=options,
    )


def embed(
    text_input: Union[str, List[str]],
    seed: int = DEFAULT_SEED,
    model: Optional[str] = None,
    unsafe_mode: bool = False,
    mode: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Create embeddings for text input.

    Args:
        text_input: Text or list of texts to embed
        seed: Seed for deterministic behavior (ignored by most remote providers)
        model: Optional remote model string (e.g., "openai:text-embedding-3-small")
        unsafe_mode: Enable remote models with best-effort determinism
        mode: Embedding mode for Jina v4 ("query" or "passage"). Defaults to "query".

    Returns:
        Numpy array of embeddings or None if error

    Examples:
        # Query embedding (default)
        query_emb = embed("What is machine learning?")

        # Passage embedding for documents
        doc_emb = embed("Machine learning is a subset of AI...", mode="passage")

        # Multiple texts with mode
        doc_embs = embed(["Doc 1", "Doc 2"], mode="passage")
    """
    # AIDEV-NOTE: Skip daemon for remote models to avoid loading local embedding model
    # Remote models (containing ':' in the name) are handled directly by core_embed
    is_remote_model = model and ":" in model and unsafe_mode

    # AIDEV-NOTE: Use daemon by default for local embeddings unless explicitly disabled
    if not is_remote_model and os.environ.get("STEADYTEXT_DISABLE_DAEMON") != "1":
        client = get_daemon_client()
        if client is not None:
            try:
                return client.embed(
                    text_input,
                    seed=seed,
                    model=model,
                    unsafe_mode=unsafe_mode,
                    mode=mode,
                )
            except ConnectionError:
                # Fall back to direct embedding
                logger.info(
                    "Daemon not available, falling back to direct embedding. "
                    "For better performance, start the daemon with 'st daemon start'"
                )

    try:
        result = core_embed(
            text_input, seed=seed, model=model, unsafe_mode=unsafe_mode, mode=mode
        )
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
                logger.info(
                    "Daemon not available, falling back to direct reranking. "
                    "For better performance, start the daemon with 'st daemon start'"
                )

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


def preload_models(
    verbose: bool = False, size: Optional[str] = None, skip_embeddings: bool = False
):
    """Preload models to ensure they're available for generation and embedding.

    Args:
        verbose: Whether to log progress messages
        size: Model size to preload ("small", "medium", "large")
        skip_embeddings: Skip preloading embedding model (useful when only using remote embeddings)
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

    # AIDEV-NOTE: Skip embedding model preload when using only remote embeddings
    # This prevents unnecessary local model loading when daemon is used for remote models only
    if not skip_embeddings:
        if verbose:
            logger.info("Preloading embedding model...")
        get_embedding_model_instance()
    elif verbose:
        logger.info("Skipping embedding model preload (skip_embeddings=True)")

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
    "generate_pydantic",
]
