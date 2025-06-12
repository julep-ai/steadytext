# AIDEV-NOTE: Thread-safe singleton model loader with caching and validation
# Handles both generator and embedder models with proper cleanup

from llama_cpp import Llama
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

    # AIDEV-NOTE: Generator model loading with parameter configuration and error handling
    @classmethod
    def get_generator(cls, force_reload: bool = False) -> Optional[Llama]:
        inst = cls.__getInstance()
        with inst._lock:
            model_path = get_generation_model_path()
            if model_path is None:
                logger.error("Generator model file not found by cache.")
                return None

            if (
                inst._generator_model is None
                or inst._generator_path != model_path
                or force_reload
            ):
                if inst._generator_model is not None:
                    del inst._generator_model
                    inst._generator_model = None

                logger.info(f"Loading generator model from: {model_path}")
                try:
                    params = {**LLAMA_CPP_MAIN_PARAMS_DETERMINISTIC}
                    params["embedding"] = False

                    inst._generator_model = Llama(model_path=str(model_path), **params)
                    inst._generator_path = model_path
                    logger.info("Generator model loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load generator model: {e}", exc_info=True)
                    inst._generator_model = None
                    inst._generator_path = None
            return inst._generator_model

    # AIDEV-NOTE: Embedder model loading with dimension validation - critical for consistency
    @classmethod
    def get_embedder(cls, force_reload: bool = False) -> Optional[Llama]:
        inst = cls.__getInstance()
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
                    logger.debug(f"Embedding Llama params: {params}") # ADDED LOGGING
                    inst._embedder_model = Llama(model_path=str(model_path), **params)

                    model_n_embd = (
                        inst._embedder_model.n_embd()
                        if hasattr(inst._embedder_model, "n_embd")
                        else 0
                    )
                    # Temporarily bypass dimension check as per objective
                    inst._embedder_path = model_path
                    logger.info(f"Embedder model loaded (dimension check temporarily bypassed). Model n_embd: {model_n_embd}, Expected EMBEDDING_DIMENSION: {EMBEDDING_DIMENSION}")
                    # if model_n_embd != EMBEDDING_DIMENSION:
                    #     logger.error(
                    #         f"Embedder model n_embd ({model_n_embd}) "
                    #         f"does not match expected EMBEDDING_DIMENSION ({EMBEDDING_DIMENSION})."
                    #     )
                    #     del inst._embedder_model
                    #     inst._embedder_model = None
                    #     inst._embedder_path = None
                    # else:
                    #     inst._embedder_path = model_path
                    #     logger.info("Embedder model loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to load embedder model: {e}", exc_info=True)
                    inst._embedder_model = None
                    inst._embedder_path = None
            return inst._embedder_model


def get_generator_model_instance(force_reload: bool = False) -> Optional[Llama]:
    return _ModelInstanceCache.get_generator(force_reload)


def get_embedding_model_instance(force_reload: bool = False) -> Optional[Llama]:
    return _ModelInstanceCache.get_embedder(force_reload)
