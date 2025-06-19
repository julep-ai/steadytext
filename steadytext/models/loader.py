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
from typing import Optional
from ..utils import (
    logger,
    LLAMA_CPP_MAIN_PARAMS_DETERMINISTIC,
    LLAMA_CPP_EMBEDDING_PARAMS_DETERMINISTIC,
    EMBEDDING_DIMENSION,
    set_deterministic_environment,
)
from .cache import get_generation_model_path, get_embedding_model_path


# AIDEV-NOTE: Critical singleton pattern implementation for model caching
# Thread-safe with proper resource management and dimension validation
class _ModelInstanceCache:
    _instance = None
    _lock = threading.Lock()

    _generator_model: Optional[Llama] = None
    _embedder_model: Optional[Llama] = None
    _generator_path: Optional[Path] = None
    _embedder_path: Optional[Path] = None

    @classmethod
    def __getInstance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls.__new__(cls)
                    set_deterministic_environment()
        return cls._instance

    def __init__(self):
        raise RuntimeError("Call __getInstance() instead")

    # AIDEV-NOTE: Generator model loading with parameter configuration and
    # error handling
    @classmethod
    def get_generator(
        cls, force_reload: bool = False, enable_logits: bool = False
    ) -> Optional[Llama]:
        if Llama is None:
            logger.error(
                "llama_cpp.Llama not available; generator model cannot be loaded"
            )
            return None
        inst = cls.__getInstance()
        # AIDEV-NOTE: This lock is crucial for thread-safe access to the generator model.
        with inst._lock:
            model_path = get_generation_model_path()
            if model_path is None:
                logger.error("Generator model file not found by cache.")
                return None

            # Check if we need to reload due to logits configuration change
            current_logits_enabled = getattr(inst, "_generator_logits_enabled", False)
            needs_reload = (
                inst._generator_model is None
                or inst._generator_path != model_path
                or force_reload
                or (enable_logits != current_logits_enabled)
            )

            if needs_reload:
                if inst._generator_model is not None:
                    del inst._generator_model
                    inst._generator_model = None

                logger.info(
                    f"Loading generator model from: {model_path} (logits_all={enable_logits})"
                )
                try:
                    params = {**LLAMA_CPP_MAIN_PARAMS_DETERMINISTIC}
                    params["embedding"] = False
                    if enable_logits:
                        params["logits_all"] = True

                    inst._generator_model = Llama(model_path=str(model_path), **params)
                    inst._generator_path = model_path
                    inst._generator_logits_enabled = enable_logits
                    logger.info("Generator model loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load generator model: {e}", exc_info=True)
                    inst._generator_model = None
                    inst._generator_path = None
                    inst._generator_logits_enabled = False
            return inst._generator_model

    # AIDEV-NOTE: Embedder model loading with dimension validation - critical
    # for consistency
    @classmethod
    def get_embedder(cls, force_reload: bool = False) -> Optional[Llama]:
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
                    del inst._embedder_model
                    inst._embedder_model = None

                logger.info(f"Loading embedder model from: {model_path}")
                try:
                    params = {**LLAMA_CPP_EMBEDDING_PARAMS_DETERMINISTIC}
                    logger.debug(f"Embedding Llama params: {params}")  # ADDED LOGGING
                    inst._embedder_model = Llama(model_path=str(model_path), **params)

                    model_n_embd = (
                        inst._embedder_model.n_embd()
                        if hasattr(inst._embedder_model, "n_embd")
                        else 0
                    )
                    # Restoring original dimension check logic
                    # AIDEV-NOTE: This is an important validation step for the embedding model's dimensions.
                    if model_n_embd != EMBEDDING_DIMENSION:
                        logger.error(
                            f"Embedder model n_embd ({model_n_embd}) does not "
                            f"match expected EMBEDDING_DIMENSION "
                            f"({EMBEDDING_DIMENSION})."
                        )
                        if inst._embedder_model is not None:  # Safety check
                            del inst._embedder_model
                            inst._embedder_model = None
                        inst._embedder_path = None  # Also clear path
                    else:
                        inst._embedder_path = model_path
                        logger.info("Embedder model loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load embedder model: {e}", exc_info=True)
                    inst._embedder_model = None
                    inst._embedder_path = None
            return inst._embedder_model


def get_generator_model_instance(
    force_reload: bool = False, enable_logits: bool = False
) -> Optional[Llama]:
    return _ModelInstanceCache.get_generator(force_reload, enable_logits)


def get_embedding_model_instance(force_reload: bool = False) -> Optional[Llama]:
    return _ModelInstanceCache.get_embedder(force_reload)


# AIDEV-NOTE: Cache clearing utility for testing - ensures clean state for mock patching
def clear_model_cache():
    """Clear all cached model instances and paths.

    This function is primarily intended for testing to ensure clean state
    when using mock models. It clears both generator and embedder caches.

    AIDEV-NOTE: This is essential for proper mock testing because the singleton
    pattern caches real model instances across test runs. Without clearing,
    patches may not take effect when cached models exist.
    """
    inst = _ModelInstanceCache._ModelInstanceCache__getInstance()
    with inst._lock:
        # Clear generator model and state
        if inst._generator_model is not None:
            del inst._generator_model
            inst._generator_model = None
        inst._generator_path = None
        inst._generator_logits_enabled = False

        # Clear embedder model and state
        if inst._embedder_model is not None:
            del inst._embedder_model
            inst._embedder_model = None
        inst._embedder_path = None

        logger.debug("Model cache cleared for testing")
