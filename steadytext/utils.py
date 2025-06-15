import os
import random  # Imported by user's new utils.py
import numpy as np
from pathlib import Path
import logging
import platform  # For get_cache_dir
from typing import Dict, Any, List  # For type hints

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
logger.setLevel(logging.INFO)

# --- Model Configuration ---
DEFAULT_GENERATION_MODEL_REPO = "DevQuasar/openbmb.BitCPM4-1B-GGUF"
DEFAULT_EMBEDDING_MODEL_REPO = "Qwen/Qwen3-Embedding-0.6B-GGUF"
GENERATION_MODEL_FILENAME = "openbmb.BitCPM4-1B.Q8_0.gguf"
EMBEDDING_MODEL_FILENAME = "Qwen3-Embedding-0.6B-Q8_0.gguf"

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
    logger.info(f"Deterministic environment set with seed: {seed}")


# Call it on import to ensure early setup
set_deterministic_environment(DEFAULT_SEED)


# --- Llama.cpp Model Parameters ---
# These are now structured as per the new loader.py's expectation
LLAMA_CPP_BASE_PARAMS: Dict[str, Any] = {
    "n_ctx": 2048,  # Default context, Qwen0.5B supports more but this is a safe default
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
# AIDEV-NOTE: Increased default max tokens for generation from 256 to 512
GENERATION_MAX_NEW_TOKENS = 512
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
        # AIDEV-TODO: Consider adding more robust permission checking and user guidance
        logger.error(
            f"Failed to create cache directory at {cache_dir}: {e}. Check permissions."
        )
        import tempfile

        fallback_dir = Path(tempfile.gettempdir()) / DEFAULT_CACHE_DIR_NAME / "models"
        logger.warning(f"Using temporary fallback cache directory: {fallback_dir}.")
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            return fallback_dir
        except OSError as fe:
            logger.critical(
                f"Failed to create even fallback cache directory at {fallback_dir}: {fe}."
            )
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
