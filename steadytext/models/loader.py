# AIDEV-NOTE: Thread-safe singleton model loader with caching and validation
# Handles both generator and embedder models with proper cleanup

try:
    from llama_cpp import Llama
except ImportError as import_err:  # pragma: no cover - import failure path
    # AIDEV-NOTE: Allow tests to run without llama_cpp installed
    Llama = None  # type: ignore
    import logging

    logging.getLogger(__name__).error("llama_cpp not available: %s", import_err)
from pathlib import Path
import threading
import logging
from typing import Optional, Dict
from ..utils import (
    LLAMA_CPP_EMBEDDING_PARAMS_DETERMINISTIC,
    LLAMA_CPP_MAIN_PARAMS_DETERMINISTIC,
    EMBEDDING_DIMENSION,
    logger,
    set_deterministic_environment,
    suppress_llama_output,
    DEFAULT_SEED,
    check_model_compatibility,
    GENERATION_MODEL_REPO,
    GENERATION_MODEL_FILENAME,
    get_optimal_context_window,
)
from .cache import get_generation_model_path, get_embedding_model_path


# AIDEV-NOTE: This singleton cache is the core of resource management, ensuring that a model is loaded only once. The thread-lock is critical for safe initialization in multi-threaded contexts.
class _ModelInstanceCache:
    _instance = None
    _lock = threading.Lock()

    _generator_model: Optional[Llama] = None
    _embedder_model: Optional[Llama] = None
    _generator_path: Optional[Path] = None
    _embedder_path: Optional[Path] = None

    # AIDEV-NOTE: Cache for multiple generator models to support switching
    _generator_models_cache: Dict[str, Optional[Llama]] = {}
    _generator_paths_cache: Dict[str, Optional[Path]] = {}

    @classmethod
    def __getInstance(cls) -> "_ModelInstanceCache":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    # AIDEV-NOTE: Defer deterministic environment setup until the first use.
                    # set_deterministic_environment()
        return cls._instance

    def __init__(self) -> None:
        raise RuntimeError("Call __getInstance() instead")

    @staticmethod
    def _get_model_registry() -> Dict[str, Dict[str, str]]:
        """Get the model registry from utils.

        AIDEV-NOTE: Helper method to access MODEL_REGISTRY from utils module.
        """
        from ..utils import MODEL_REGISTRY

        return MODEL_REGISTRY

    @classmethod
    def _ensure_init(cls):
        """Ensure the singleton instance and deterministic environment are set up."""
        if cls._instance is None:
            cls.__getInstance()
            # AIDEV-NOTE: Now we set the environment, only on the first real use.
            set_deterministic_environment(DEFAULT_SEED)

    # AIDEV-NOTE: Generator model loading with parameter configuration, error handling, and support for different models.
    @classmethod
    def get_generator(
        cls,
        force_reload: bool = False,
        enable_logits: bool = False,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Optional[Llama]:
        cls._ensure_init()  # AIDEV-NOTE: Lazy initialization
        if Llama is None:
            logger.error(
                "llama_cpp.Llama not available; generator model cannot be loaded"
            )
            return None
        inst = cls.__getInstance()
        # AIDEV-NOTE: This lock is crucial for thread-safe access to the generator model.
        with inst._lock:
            # Create cache key for this specific model
            cache_key = f"{repo_id or 'default'}::{filename or 'default'}"

            # Use default model if no custom params and it's already loaded
            if (
                repo_id is None
                and filename is None
                and inst._generator_model is not None
            ):
                if not force_reload and enable_logits == getattr(
                    inst, "_generator_logits_enabled", False
                ):
                    return inst._generator_model

            # Check if this model is already cached
            if cache_key in inst._generator_models_cache and not force_reload:
                cached_model = inst._generator_models_cache[cache_key]
                if cached_model is not None:
                    # Check if logits configuration matches
                    cached_logits = getattr(cached_model, "_logits_enabled", False)
                    if cached_logits == enable_logits:
                        return cached_model

            model_path = get_generation_model_path(repo_id, filename)
            if model_path is None:
                logger.error(
                    f"Generator model file not found by cache for {cache_key}."
                )
                return None

            # Check model compatibility
            actual_repo = repo_id or GENERATION_MODEL_REPO
            actual_filename = filename or GENERATION_MODEL_FILENAME
            is_compatible, warning_msg = check_model_compatibility(
                actual_repo, actual_filename
            )
            if not is_compatible:
                logger.warning(f"Model compatibility issue: {warning_msg}")

            # For custom models, always load fresh (or use cache)
            needs_reload = True

            if needs_reload:
                # Clean up existing model in cache if reloading
                if cache_key in inst._generator_models_cache:
                    old_model = inst._generator_models_cache[cache_key]
                    if old_model is not None:
                        # AIDEV-NOTE: Explicit cleanup to prevent memory leaks
                        if hasattr(old_model, "close"):
                            old_model.close()
                        del old_model
                    inst._generator_models_cache[cache_key] = None
                    inst._generator_paths_cache[cache_key] = None

                # Only log if logger level allows INFO messages
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        f"Loading generator model from: {model_path} (logits_all={enable_logits})"
                    )
                try:
                    params = {**LLAMA_CPP_MAIN_PARAMS_DETERMINISTIC}
                    params["embedding"] = False
                    if enable_logits:
                        params["logits_all"] = True

                    # AIDEV-NOTE: Set optimal context window dynamically based on model
                    # Resolve model name to pass to get_optimal_context_window
                    model_name = None
                    if repo_id is None and filename is None:
                        # Using default model - check if it matches a known model
                        for name, config in cls._get_model_registry().items():
                            if (
                                config.get("repo") == GENERATION_MODEL_REPO
                                and config.get("filename") == GENERATION_MODEL_FILENAME
                            ):
                                model_name = name
                                break
                    else:
                        # Check if custom model matches a known model
                        for name, config in cls._get_model_registry().items():
                            if (
                                config.get("repo") == repo_id
                                and config.get("filename") == filename
                            ):
                                model_name = name
                                break

                    # Get optimal context window
                    optimal_ctx = get_optimal_context_window(
                        model_name=model_name,
                        model_repo=repo_id or GENERATION_MODEL_REPO,
                    )
                    params["n_ctx"] = optimal_ctx

                    # Suppress llama.cpp output in quiet mode
                    with suppress_llama_output():
                        new_model = Llama(model_path=str(model_path), **params)
                    new_model._logits_enabled = enable_logits  # type: ignore[attr-defined]

                    # Verify model loaded successfully
                    if new_model is None or not hasattr(new_model, "n_ctx"):
                        raise ValueError("Model loaded but appears to be invalid")

                    # Store in cache
                    inst._generator_models_cache[cache_key] = new_model
                    inst._generator_paths_cache[cache_key] = model_path

                    # Update default model reference if this is the default
                    if repo_id is None and filename is None:
                        inst._generator_model = new_model
                        inst._generator_path = model_path
                        inst._generator_logits_enabled = enable_logits

                    # Only log if logger level allows INFO messages
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(
                            f"Generator model loaded successfully for {cache_key}."
                        )
                    return new_model
                except ValueError as ve:
                    # Specific GGUF format error from llama_cpp
                    error_msg = str(ve)
                    if "Failed to load model from file" in error_msg:
                        logger.error(
                            f"Failed to load generator model {cache_key}: {error_msg}\n"
                            f"This may be a GGUF format compatibility issue with the inference-sh fork of llama-cpp-python.\n"
                            f"Try setting STEADYTEXT_USE_FALLBACK_MODEL=true to use a known working model."
                        )
                    else:
                        logger.error(
                            f"Failed to load generator model {cache_key}: {ve}",
                            exc_info=True,
                        )
                    inst._generator_models_cache[cache_key] = None
                    inst._generator_paths_cache[cache_key] = None
                    return None
                except Exception as e:
                    logger.error(
                        f"Unexpected error loading generator model {cache_key}: {e}\n"
                        f"Model file: {model_path}\n"
                        f"llama-cpp-python version: {getattr(Llama, '__version__', 'unknown')}",
                        exc_info=True,
                    )
                    inst._generator_models_cache[cache_key] = None
                    inst._generator_paths_cache[cache_key] = None
                    return None

            # Return cached model
            return inst._generator_models_cache.get(cache_key)

    # AIDEV-NOTE: Embedder model loading with dimension validation - critical
    # for consistency
    @classmethod
    def get_embedder(cls, force_reload: bool = False) -> Optional[Llama]:
        cls._ensure_init()  # AIDEV-NOTE: Lazy initialization
        if Llama is None:
            logger.error(
                "llama_cpp.Llama not available; embedder model cannot be loaded"
            )
            return None
        inst = cls.__getInstance()
        # AIDEV-NOTE: This lock is crucial for thread-safe access to the embedder model.
        with inst._lock:
            model_path = get_embedding_model_path()
            if model_path is None:
                logger.error("Embedder model file not found by cache.")
                return None

            if (
                inst._embedder_model is None
                or inst._embedder_path != model_path
                or force_reload
            ):
                if inst._embedder_model is not None:
                    # AIDEV-NOTE: Explicit cleanup to prevent memory leaks
                    if hasattr(inst._embedder_model, "close"):
                        inst._embedder_model.close()
                    del inst._embedder_model
                    inst._embedder_model = None

                # Only log if logger level allows INFO messages
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f"Loading embedder model from: {model_path}")
                try:
                    params = {**LLAMA_CPP_EMBEDDING_PARAMS_DETERMINISTIC}

                    # AIDEV-NOTE: Set optimal context window for embedding model too
                    # Embedding models typically need less context than generation models
                    # but we still use the dynamic system for consistency
                    optimal_ctx = get_optimal_context_window(
                        model_name=None,  # Could add embedding model detection later
                        model_repo=None,
                        requested_size=None,
                    )
                    params["n_ctx"] = optimal_ctx

                    logger.debug(f"Embedding Llama params: {params}")  # ADDED LOGGING
                    # Suppress llama.cpp output in quiet mode
                    with suppress_llama_output():
                        inst._embedder_model = Llama(
                            model_path=str(model_path), **params
                        )

                    model_n_embd = (
                        inst._embedder_model.n_embd()
                        if hasattr(inst._embedder_model, "n_embd")
                        else 0
                    )
                    # AIDEV-NOTE: This dimension check is a critical validation step to ensure the loaded embedding model produces vectors of the expected dimension.
                    if model_n_embd != EMBEDDING_DIMENSION:
                        logger.error(
                            f"Embedder model n_embd ({model_n_embd}) does not "
                            f"match expected EMBEDDING_DIMENSION "
                            f"({EMBEDDING_DIMENSION})."
                        )
                        if inst._embedder_model is not None:  # Safety check
                            # AIDEV-NOTE: Explicit cleanup to prevent memory leaks
                            if hasattr(inst._embedder_model, "close"):
                                inst._embedder_model.close()
                            del inst._embedder_model
                            inst._embedder_model = None
                        inst._embedder_path = None  # Also clear path
                    else:
                        inst._embedder_path = model_path
                        # Only log if logger level allows INFO messages
                        if logger.isEnabledFor(logging.INFO):
                            logger.info("Embedder model loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load embedder model: {e}", exc_info=True)
                    inst._embedder_model = None
                    inst._embedder_path = None
            return inst._embedder_model


def get_generator_model_instance(
    force_reload: bool = False,
    enable_logits: bool = False,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[Llama]:
    """Get a generator model instance with optional custom model specification.

    AIDEV-NOTE: Extended to support dynamic model loading while maintaining backward compatibility.
    """
    return _ModelInstanceCache.get_generator(
        force_reload, enable_logits, repo_id, filename
    )


def get_embedding_model_instance(force_reload: bool = False) -> Optional[Llama]:
    return _ModelInstanceCache.get_embedder(force_reload)


# AIDEV-NOTE: Cache clearing utility for testing - ensures clean state for mock patching
def clear_model_cache() -> None:
    """Clear all cached model instances and paths.

    This function is primarily intended for testing to ensure clean state
    when using mock models. It clears both generator and embedder caches.

    AIDEV-NOTE: This is essential for proper mock testing, as the singleton pattern caches real model instances across test runs.
    """
    inst = _ModelInstanceCache._ModelInstanceCache__getInstance()  # type: ignore[attr-defined]
    with inst._lock:
        # Clear generator model and state
        if inst._generator_model is not None:
            # AIDEV-NOTE: Explicit cleanup to prevent memory leaks
            if hasattr(inst._generator_model, "close"):
                inst._generator_model.close()
            del inst._generator_model
            inst._generator_model = None
        inst._generator_path = None
        inst._generator_logits_enabled = False

        # Clear all cached generator models
        for key, model in inst._generator_models_cache.items():
            if model is not None:
                # AIDEV-NOTE: Explicit cleanup to prevent memory leaks
                if hasattr(model, "close"):
                    model.close()
                del model
        inst._generator_models_cache.clear()
        inst._generator_paths_cache.clear()

        # Clear embedder model and state
        if inst._embedder_model is not None:
            # AIDEV-NOTE: Explicit cleanup to prevent memory leaks
            if hasattr(inst._embedder_model, "close"):
                inst._embedder_model.close()
            del inst._embedder_model
            inst._embedder_model = None
        inst._embedder_path = None

        logger.debug("Model cache cleared for testing")
