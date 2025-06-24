import os
import random  # Imported by user's new utils.py
import numpy as np
from pathlib import Path
import logging
import platform  # For get_cache_dir
from typing import Dict, Any, List, Optional  # For type hints
import sys
from contextlib import contextmanager

# AIDEV-NOTE: Core utility functions for SteadyText - handles deterministic
# environment setup, model configuration, and cross-platform cache directory
# management

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
# AIDEV-NOTE: Switched from BitCPM4-1B to Qwen3-1.7B for better performance
# Qwen3-1.7B offers improved reasoning while maintaining reasonable size (1.83GB)
# AIDEV-NOTE: Added environment variable support for model switching
# Users can override models via STEADYTEXT_GENERATION_MODEL_REPO and STEADYTEXT_GENERATION_MODEL_FILENAME
DEFAULT_GENERATION_MODEL_REPO = "Qwen/Qwen3-1.7B-GGUF"
DEFAULT_EMBEDDING_MODEL_REPO = "Qwen/Qwen3-Embedding-0.6B-GGUF"
GENERATION_MODEL_FILENAME = "Qwen3-1.7B-Q8_0.gguf"
EMBEDDING_MODEL_FILENAME = "Qwen3-Embedding-0.6B-Q8_0.gguf"

# AIDEV-NOTE: Model registry for validated alternative models
# Each entry contains repo_id and filename for known working models
MODEL_REGISTRY = {
    # Qwen3 models
    "qwen3-0.6b": {"repo": "Qwen/Qwen3-0.6B-GGUF", "filename": "Qwen3-0.6B-Q8_0.gguf"},
    "qwen3-1.7b": {"repo": "Qwen/Qwen3-1.7B-GGUF", "filename": "Qwen3-1.7B-Q8_0.gguf"},
    "qwen3-4b": {
        "repo": "Qwen/Qwen3-4B-GGUF",
        "filename": "Qwen3-4B-Q8_0.gguf",
    },
    "qwen3-8b": {"repo": "Qwen/Qwen3-8B-GGUF", "filename": "qwen3-8b-q8_0.gguf"},
    # Qwen2.5 models - newer series with better performance
    "qwen2.5-0.5b": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q8_0.gguf",
    },
    "qwen2.5-1.5b": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q8_0.gguf",
    },
    "qwen2.5-3b": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q8_0.gguf",
    },
    "qwen2.5-7b": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename": "qwen2.5-7b-instruct-q8_0.gguf",
    },
}

# AIDEV-NOTE: Size to model mapping for convenient size-based selection
SIZE_TO_MODEL = {
    "small": "qwen3-0.6b",
    "medium": "qwen3-1.7b",  # default
    "large": "qwen3-4b",
}

# Get model configuration from environment or use defaults
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

# --- Determinism & Seeds ---
DEFAULT_SEED = 42


# AIDEV-NOTE: Critical function for ensuring deterministic behavior
# across all operations
def set_deterministic_environment(seed: int = DEFAULT_SEED):
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
LLAMA_CPP_BASE_PARAMS: Dict[str, Any] = {
    "n_ctx": 3072,  # Increased context for Qwen3 thinking mode support
    "n_gpu_layers": 0,  # CPU-only for zero-config
    "seed": DEFAULT_SEED,
    "verbose": False,
}

LLAMA_CPP_MAIN_PARAMS_DETERMINISTIC: Dict[str, Any] = {
    **LLAMA_CPP_BASE_PARAMS,
    # Parameters for generation
    # explicit 'embedding': False will be set in loader for gen model
    # logits_all will be set dynamically based on whether logprobs are needed
}

# --- Output Configuration (from previous full utils.py) ---
# AIDEV-NOTE: Increased default max tokens for generation from 512 to 1024 for Qwen3 thinking support
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

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # AIDEV-NOTE: Enhanced permission error handling with OS-specific guidance
        system = platform.system()

        # Provide OS-specific guidance
        if system == "Windows":
            guidance = (
                f"Permission denied creating cache directory at: {cache_dir}\n"
                f"Possible solutions:\n"
                f"  1. Run as Administrator\n"
                f"  2. Set LOCALAPPDATA environment variable to a writable location\n"
                f"  3. Set XDG_CACHE_HOME to a custom cache directory\n"
                f"  4. Ensure your user has write permissions to: {cache_dir.parent}"
            )
        elif system == "Darwin":  # macOS
            guidance = (
                f"Permission denied creating cache directory at: {cache_dir}\n"
                f"Possible solutions:\n"
                f"  1. Fix permissions: chmod -R u+w ~/.cache\n"
                f"  2. Set XDG_CACHE_HOME to a writable directory\n"
                f"  3. Check disk permissions with Disk Utility\n"
                f"  4. Ensure your user owns the directory: sudo chown -R $(whoami) ~/.cache"
            )
        else:  # Linux and others
            guidance = (
                f"Permission denied creating cache directory at: {cache_dir}\n"
                f"Possible solutions:\n"
                f"  1. Fix permissions: chmod -R u+w ~/.cache\n"
                f"  2. Set XDG_CACHE_HOME to a writable directory\n"
                f"  3. Check if home directory is mounted read-only\n"
                f"  4. Ensure your user owns the directory: sudo chown -R $(whoami):$(whoami) ~/.cache"
            )

        logger.error(f"{guidance}\nOriginal error: {e}")

        import tempfile

        fallback_dir = Path(tempfile.gettempdir()) / DEFAULT_CACHE_DIR_NAME / "models"
        logger.warning(
            f"Attempting to use temporary fallback cache directory: {fallback_dir}\n"
            f"Note: Models cached here may be deleted on system restart."
        )

        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            return fallback_dir
        except OSError as fe:
            logger.critical(
                f"Failed to create even fallback cache directory at {fallback_dir}.\n"
                f"This may indicate severe permission issues or a full disk.\n"
                f"Error: {fe}\n"
                f"Please resolve the permission issues or set XDG_CACHE_HOME to a writable location."
            )
            # Return original cache_dir to maintain API contract
            return cache_dir
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
        size: Size identifier ("small", "medium", "large")

    Returns:
        Tuple of (repo_id, filename) to use for model loading
    """
    # If explicit repo and filename provided, use them
    if repo and filename:
        return repo, filename

    # If model name provided, look it up
    if model:
        config = get_model_config(model)
        return config["repo"], config["filename"]

    # If size provided, convert to model name and look it up
    if size:
        if size not in SIZE_TO_MODEL:
            available = ", ".join(sorted(SIZE_TO_MODEL.keys()))
            raise ValueError(
                f"Size '{size}' not recognized. Available sizes: {available}"
            )
        model_name = SIZE_TO_MODEL[size]
        config = get_model_config(model_name)
        return config["repo"], config["filename"]

    # Otherwise use environment variables or defaults
    return GENERATION_MODEL_REPO, GENERATION_MODEL_FILENAME


# AIDEV-NOTE: Context manager to suppress llama.cpp's direct stdout/stderr output
# Used during model loading to prevent verbose warnings in quiet mode
# AIDEV-NOTE: Quiet mode fix (v1.3.2+) - Suppresses llama_context warnings
# - Logger no longer sets default INFO level (controlled by CLI)
# - set_deterministic_environment() no longer called on import
# - All INFO logs check logger.isEnabledFor() before logging
# - llama.cpp stdout/stderr suppressed during model loading in quiet mode
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
