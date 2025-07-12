import os
import random  # Imported by user's new utils.py
import numpy as np
from pathlib import Path
import logging
import platform  # For get_cache_dir
from typing import Dict, Any, List, Optional, Final  # For type hints
import sys
from contextlib import contextmanager

# AIDEV-NOTE: Core utility functions for SteadyText, handling deterministic environment setup, model configuration, and cross-platform cache directory management.

# --- Logger Setup ---
logger = logging.getLogger("steadytext")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
# AIDEV-NOTE: Don't set a default level - let CLI control it
# This prevents INFO messages from appearing in quiet mode

# --- Model Configuration ---
# AIDEV-NOTE: Switched from Qwen3 to Gemma-3n for generation and are using Qwen3-Embedding-0.6B for embeddings. Users can override the models via environment variables. The ggml-org repository is used for the latest GGUF versions.
DEFAULT_GENERATION_MODEL_REPO = "ggml-org/gemma-3n-E2B-it-GGUF"
DEFAULT_EMBEDDING_MODEL_REPO = "Qwen/Qwen3-Embedding-0.6B-GGUF"
DEFAULT_RERANKING_MODEL_REPO = "QuantFactory/Qwen3-Reranker-4B-GGUF"
GENERATION_MODEL_FILENAME = "gemma-3n-E2B-it-Q8_0.gguf"
EMBEDDING_MODEL_FILENAME = "Qwen3-Embedding-0.6B-Q8_0.gguf"
RERANKING_MODEL_FILENAME = "Qwen3-Reranker-4B.Q8_0.gguf"

# AIDEV-NOTE: Model registry for validated alternative models
# Each entry contains repo_id and filename for known working models
MODEL_REGISTRY = {
    # Gemma-3n models (may have compatibility issues with inference-sh fork)
    "gemma-3n-2b": {
        "repo": "ggml-org/gemma-3n-E2B-it-GGUF",
        "filename": "gemma-3n-E2B-it-Q8_0.gguf",
        "compatibility_warning": "May not be compatible with inference-sh llama-cpp-python fork",
    },
    "gemma-3n-4b": {
        "repo": "ggml-org/gemma-3n-E4B-it-GGUF",
        "filename": "gemma-3n-E4B-it-Q8_0.gguf",
        "compatibility_warning": "May not be compatible with inference-sh llama-cpp-python fork",
    },
    # Qwen models (known to work with inference-sh fork)
    "qwen2.5-3b": {
        "repo": "lmstudio-community/Qwen2.5-3B-Instruct-GGUF",
        "filename": "Qwen2.5-3B-Instruct-Q8_0.gguf",
        "verified": True,
    },
    "qwen3-1.7b": {
        "repo": "lmstudio-community/qwen3-1.7b-llama-cpp-python-GGUF",
        "filename": "qwen3-1.7b-q8_0.gguf",
        "verified": True,
    },
    # Reranking models
    "qwen3-reranker-4b": {
        "repo": "QuantFactory/Qwen3-Reranker-4B-GGUF",
        "filename": "Qwen3-Reranker-4B.Q8_0.gguf",
        "verified": True,
    },
}

# AIDEV-NOTE: Size to model mapping for convenient size-based selection
SIZE_TO_MODEL = {
    "small": "qwen3-1.7b",  # fallback to known working model
    "medium": "qwen2.5-3b",  # fallback to known working model
    "large": "gemma-3n-4b",  # may have compatibility issues
}

# Get model configuration from environment or use defaults
# AIDEV-NOTE: If gemma-3n models fail, set STEADYTEXT_USE_FALLBACK_MODEL=true
USE_FALLBACK_MODEL = (
    os.environ.get("STEADYTEXT_USE_FALLBACK_MODEL", "false").lower() == "true"
)

if USE_FALLBACK_MODEL:
    # Use known working Qwen model as fallback
    DEFAULT_GENERATION_MODEL_REPO = "lmstudio-community/Qwen2.5-3B-Instruct-GGUF"
    GENERATION_MODEL_FILENAME = "Qwen2.5-3B-Instruct-Q8_0.gguf"
    logger.info("Using fallback Qwen model due to STEADYTEXT_USE_FALLBACK_MODEL=true")

GENERATION_MODEL_REPO = os.environ.get(
    "STEADYTEXT_GENERATION_MODEL_REPO", DEFAULT_GENERATION_MODEL_REPO
)
GENERATION_MODEL_FILENAME = os.environ.get(
    "STEADYTEXT_GENERATION_MODEL_FILENAME", GENERATION_MODEL_FILENAME
)
EMBEDDING_MODEL_REPO = os.environ.get(
    "STEADYTEXT_EMBEDDING_MODEL_REPO", DEFAULT_EMBEDDING_MODEL_REPO
)
EMBEDDING_MODEL_FILENAME = os.environ.get(
    "STEADYTEXT_EMBEDDING_MODEL_FILENAME", EMBEDDING_MODEL_FILENAME
)
RERANKING_MODEL_REPO = os.environ.get(
    "STEADYTEXT_RERANKING_MODEL_REPO", DEFAULT_RERANKING_MODEL_REPO
)
RERANKING_MODEL_FILENAME = os.environ.get(
    "STEADYTEXT_RERANKING_MODEL_FILENAME", RERANKING_MODEL_FILENAME
)

# --- Determinism & Seeds ---
DEFAULT_SEED: Final[int] = 42


def validate_seed(seed: int) -> None:
    """Validate that seed is a non-negative integer.

    Args:
        seed: The seed value to validate

    Raises:
        ValueError: If seed is not a non-negative integer
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be a non-negative integer, got {seed}")


# AIDEV-NOTE: Critical function for ensuring deterministic behavior
# across all operations
def set_deterministic_environment(seed: int):
    """Sets various seeds for deterministic operations."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Note: llama.cpp itself is seeded at model load time via its parameters.
    # TF/PyTorch seeds would be set here if used directly.
    # Only log if logger level allows INFO messages
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"Deterministic environment set with seed: {seed}")


# AIDEV-NOTE: Removed automatic call on import - now called explicitly where needed
# (in generator.py, daemon server, and model loader)


# --- Llama.cpp Model Parameters ---
# These are now structured as per the new loader.py's expectation
# AIDEV-NOTE: Context window configuration now supports dynamic sizing
# Default to None to let get_optimal_context_window() determine the best value
DEFAULT_CONTEXT_WINDOW = None  # Will be set dynamically based on model capabilities

# Maximum context window sizes for known models
# AIDEV-NOTE: These are conservative estimates to ensure stability
MODEL_MAX_CONTEXT_WINDOWS = {
    "gemma-3n-2b": 8192,
    "gemma-3n-4b": 8192,
    "qwen2.5-3b": 32768,
    "qwen3-1.7b": 8192,
    "qwen3-reranker-4b": 8192,
}


def get_optimal_context_window(
    model_name: Optional[str] = None,
    model_repo: Optional[str] = None,
    requested_size: Optional[int] = None,
) -> int:
    """Determine the optimal context window size for a model.

    AIDEV-NOTE: This function determines the context window size with the following priority:
    1. User-specified via STEADYTEXT_MAX_CONTEXT_WINDOW env var
    2. Requested size parameter
    3. Known model maximum from MODEL_MAX_CONTEXT_WINDOWS
    4. Safe default of 4096

    Args:
        model_name: Optional model name from registry
        model_repo: Optional repository ID
        requested_size: Optional requested context size

    Returns:
        Optimal context window size in tokens
    """
    # Check environment variable first
    env_ctx = os.environ.get("STEADYTEXT_MAX_CONTEXT_WINDOW")
    if env_ctx:
        try:
            ctx_size = int(env_ctx)
            if ctx_size > 0:
                logger.info(f"Using context window from env var: {ctx_size}")
                return ctx_size
        except ValueError:
            logger.warning(
                f"Invalid STEADYTEXT_MAX_CONTEXT_WINDOW value: {env_ctx}. "
                "Using default."
            )

    # Use requested size if provided
    if requested_size and requested_size > 0:
        logger.info(f"Using requested context window: {requested_size}")
        return requested_size

    # Look up known model limits
    if model_name and model_name in MODEL_MAX_CONTEXT_WINDOWS:
        ctx_size = MODEL_MAX_CONTEXT_WINDOWS[model_name]
        logger.info(f"Using known context window for {model_name}: {ctx_size}")
        return ctx_size

    # Try to infer from repo name
    if model_repo:
        repo_lower = model_repo.lower()
        for model, max_ctx in MODEL_MAX_CONTEXT_WINDOWS.items():
            if model.replace("-", "").replace(".", "") in repo_lower.replace(
                "-", ""
            ).replace(".", ""):
                logger.info(
                    f"Inferred context window from repo {model_repo}: {max_ctx}"
                )
                return max_ctx

    # Default to a safe value
    default_ctx = 4096
    logger.info(f"Using default context window: {default_ctx}")
    return default_ctx


LLAMA_CPP_BASE_PARAMS: Dict[str, Any] = {
    "n_ctx": DEFAULT_CONTEXT_WINDOW,  # Set dynamically via get_optimal_context_window()
    "n_gpu_layers": 0,  # CPU-only for zero-config
    "verbose": False,
}

LLAMA_CPP_MAIN_PARAMS_DETERMINISTIC: Dict[str, Any] = {
    **LLAMA_CPP_BASE_PARAMS,
    # Parameters for generation
    # explicit 'embedding': False will be set in loader for gen model
    # logits_all will be set dynamically based on whether logprobs are needed
}

# --- Output Configuration (from previous full utils.py) ---
GENERATION_MAX_NEW_TOKENS = 1024
EMBEDDING_DIMENSION = 1024  # Setting to 1024 as per objective

LLAMA_CPP_EMBEDDING_PARAMS_DETERMINISTIC: Dict[str, Any] = {
    **LLAMA_CPP_BASE_PARAMS,
    "embedding": True,
    "logits_all": False,  # Not needed for embeddings
    # n_batch for embeddings can often be smaller if processing one by one
    "n_batch": 512,  # Default, can be tuned
    # "n_embd_trunc": EMBEDDING_DIMENSION, # Removed as per objective
}

# --- Sampling Parameters for Generation (from previous full utils.py) ---
# These are passed to model() or create_completion() not Llama constructor usually
LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC: Dict[str, Any] = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "max_tokens": GENERATION_MAX_NEW_TOKENS,  # Max tokens to generate
    # stop sequences will be handled by core.generator using DEFAULT_STOP_SEQUENCES
}

# --- Stop Sequences (from previous full utils.py) ---
DEFAULT_STOP_SEQUENCES: List[str] = [
    "<|im_end|>",
    "<|im_start|>",
    "</s>",
    "<|endoftext|>",
]


# --- Cache Directory Logic (from previous full utils.py) ---
DEFAULT_CACHE_DIR_NAME = "steadytext"


# AIDEV-NOTE: Complex cross-platform cache directory logic with fallback handling
def get_cache_dir() -> Path:
    system = platform.system()
    if system == "Windows":
        cache_home_str = os.environ.get("LOCALAPPDATA")
        if cache_home_str is None:
            cache_home = Path.home() / "AppData" / "Local"
        else:
            cache_home = Path(cache_home_str)
        cache_dir = (
            cache_home / DEFAULT_CACHE_DIR_NAME / DEFAULT_CACHE_DIR_NAME / "models"
        )
    else:
        cache_home_str = os.environ.get("XDG_CACHE_HOME")
        if cache_home_str is None:
            cache_home = Path.home() / ".cache"
        else:
            cache_home = Path(cache_home_str)
        cache_dir = cache_home / DEFAULT_CACHE_DIR_NAME / "models"

    # AIDEV-NOTE: Directory creation is now deferred to the components that need it,
    # such as the model cache downloader. This prevents I/O operations during import.
    # try:
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    # except OSError as e:
    #     # ... (rest of the error handling block is now dead code)
    return cache_dir


# AIDEV-NOTE: Add validate_normalized_embedding function that's referenced
# in embedder.py
def validate_normalized_embedding(  # noqa E501
    embedding: np.ndarray, dim: int = EMBEDDING_DIMENSION, tolerance: float = 1e-5
) -> bool:
    """Validates that an embedding has correct shape, dtype, and is properly normalized."""
    if embedding.shape != (dim,):
        return False
    if embedding.dtype != np.float32:
        return False
    norm = np.linalg.norm(embedding)
    # Allow zero vectors (norm=0) or properly normalized vectors (norm approx 1)
    return bool(norm < tolerance or abs(norm - 1.0) < tolerance)  # noqa E501


# AIDEV-NOTE: Helper functions for model configuration and switching
def get_model_config(model_name: str) -> Dict[str, str]:
    """Get model configuration from registry by name.

    Args:
        model_name: Name of the model (e.g., "qwen2.5-3b", "qwen3-8b")

    Returns:
        Dict with 'repo' and 'filename' keys

    Raises:
        ValueError: If model_name is not in registry
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Model '{model_name}' not found in registry. Available models: {available}"
        )
    return MODEL_REGISTRY[model_name]


def resolve_model_params(
    model: Optional[str] = None,
    repo: Optional[str] = None,
    filename: Optional[str] = None,
    size: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve model parameters with precedence: explicit params > model name > size > env vars > defaults.

    Args:
        model: Model name from registry (e.g., "qwen2.5-3b")
        repo: Explicit repository ID (overrides model lookup)
        filename: Explicit filename (overrides model lookup)
        size: Size identifier ("small", "large")

    Returns:
        Tuple of (repo_id, filename) to use for model loading
    """
    # If explicit repo and filename provided, use them
    if repo and filename:
        return repo, filename

    # If model name provided, look it up
    if model:
        try:
            config = get_model_config(model)
            return config["repo"], config["filename"]
        except ValueError:
            logger.warning(
                f"Invalid model name '{model}' provided. Falling back to default model."
            )
            # Fall through to default

    # If size provided, convert to model name and look it up
    if size:
        if size in SIZE_TO_MODEL:
            model_name = SIZE_TO_MODEL[size]
            config = get_model_config(model_name)
            return config["repo"], config["filename"]
        else:
            logger.warning(
                f"Invalid size '{size}' provided. Falling back to default model."
            )
            # Fall through to default

    # Otherwise use environment variables or defaults
    return GENERATION_MODEL_REPO, GENERATION_MODEL_FILENAME


# AIDEV-NOTE: A context manager to suppress llama.cpp's direct stdout/stderr output. This is used during model loading to prevent verbose warnings in quiet mode.
@contextmanager
def suppress_llama_output():
    """Context manager to suppress stdout/stderr during llama.cpp operations.

    This is needed because llama.cpp writes some messages directly to stdout/stderr,
    bypassing Python's logging system. Only used when logger is set to ERROR or higher.
    """
    # Only suppress if logger level is ERROR or higher (quiet mode)
    if logger.isEnabledFor(logging.INFO):
        # In verbose mode, don't suppress anything
        yield
        return

    # Save original stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        # Redirect to devnull
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        # Always restore original streams
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        try:
            devnull.close()
        except Exception:
            pass


# AIDEV-NOTE: Centralized cache key generation to ensure consistency
# across all caching operations and prevent duplicate logic
def generate_cache_key(prompt: str, eos_string: str = "[EOS]") -> str:
    """Generate a consistent cache key for generation requests.

    Args:
        prompt: The input prompt text
        eos_string: The end-of-sequence string, defaults to "[EOS]"

    Returns:
        A cache key string that includes eos_string if it's not the default

    AIDEV-NOTE: This centralizes the cache key generation logic that was previously duplicated. The key format ensures that different eos_string values do not collide in the cache.
    """
    prompt_str = prompt if isinstance(prompt, str) else str(prompt)
    return prompt_str if eos_string == "[EOS]" else f"{prompt_str}::EOS::{eos_string}"


def should_use_cache_for_generation(
    return_logprobs: bool, repo_id: Optional[str], filename: Optional[str]
) -> bool:
    """Determine if generation result should be cached.

    Args:
        return_logprobs: Whether logprobs were requested
        repo_id: Custom repository ID (None for default model)
        filename: Custom filename (None for default model)

    Returns:
        True if the result should be cached, False otherwise

    AIDEV-NOTE: A centralized caching decision logic. Only non-logprobs requests using the default model are cached.
    """
    return not return_logprobs and repo_id is None and filename is None


def should_use_cache_for_streaming(
    include_logprobs: bool,
    model: Optional[str],
    model_repo: Optional[str],
    model_filename: Optional[str],
    size: Optional[str],
) -> bool:
    """Determine if streaming generation result should be cached.

    Args:
        include_logprobs: Whether logprobs were requested
        model: Model name parameter
        model_repo: Custom repository parameter
        model_filename: Custom filename parameter
        size: Size parameter

    Returns:
        True if the result should be cached, False otherwise

    AIDEV-NOTE: A specialized caching logic for streaming generation that checks all model selection parameters to ensure only default model results are cached.
    """
    return (
        not include_logprobs
        and model is None
        and model_repo is None
        and model_filename is None
        and size is None
    )


def check_model_compatibility(repo_id: str, filename: str) -> tuple[bool, str]:
    """Check if a model is known to be compatible with the current llama-cpp-python version.

    Args:
        repo_id: Repository ID of the model
        filename: Filename of the model

    Returns:
        Tuple of (is_compatible, warning_message)

    AIDEV-NOTE: This helps detect potential compatibility issues with the inference-sh fork of llama-cpp-python
    """
    # Check if this is a known problematic model
    for model_name, config in MODEL_REGISTRY.items():
        if config["repo"] == repo_id and config["filename"] == filename:
            if "compatibility_warning" in config:
                return False, config["compatibility_warning"]
            elif config.get("verified", False):
                return True, ""
            break

    # Check for known problematic patterns
    if "gemma-3n" in filename.lower():
        return (
            False,
            "Gemma-3n models may not be compatible with inference-sh llama-cpp-python fork. Consider using STEADYTEXT_USE_FALLBACK_MODEL=true",
        )

    # Unknown model - proceed with caution
    return True, ""
