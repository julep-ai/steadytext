from steadytext.utils import (
    logger as steadytext_logger,
    EMBEDDING_DIMENSION,
    LLAMA_CPP_EMBEDDING_PARAMS_DETERMINISTIC,
)

# Using create_embedding for more direct test
from steadytext.core.embedder import create_embedding
from steadytext.models.loader import get_embedding_model_instance, _ModelInstanceCache
import logging
import sys
import os

# Ensure steadytext is importable from the local project
sys.path.insert(0, os.getcwd())


# Set logging to DEBUG to see the params passed to Llama constructor
steadytext_logger.setLevel(logging.DEBUG)
# Also set level for other relevant loggers if they exist
logging.getLogger("steadytext.models.loader").setLevel(logging.DEBUG)
logging.getLogger("steadytext.core.embedder").setLevel(logging.DEBUG)  # Added

print("--- Starting Embedding Dimension Test ---")
print(f"Expected EMBEDDING_DIMENSION (from utils.py): {EMBEDDING_DIMENSION}")
print(
    f"LLAMA_CPP_EMBEDDING_PARAMS_DETERMINISTIC (from utils.py): "
    f"{LLAMA_CPP_EMBEDDING_PARAMS_DETERMINISTIC}"
)

# Force a reload to ensure new params are used and logging is triggered
# Clear any cached model first for a clean load attempt
if _ModelInstanceCache._embedder_model is not None:
    print("Clearing cached embedder model instance.")
    del _ModelInstanceCache._embedder_model
    _ModelInstanceCache._embedder_model = None
    _ModelInstanceCache._embedder_path = None

model = None
try:
    print("Attempting to load embedding model...")
    # Pass force_reload=True to ensure it tries to load based on current params
    model = get_embedding_model_instance(force_reload=True)
except Exception as e:
    print(f"Error during get_embedding_model_instance: {e}")
    steadytext_logger.exception("Exception in get_embedding_model_instance")


if model:
    print("Embedding model loaded successfully.")

    model_internal_n_embd = 0
    if hasattr(model, "n_embd") and callable(model.n_embd):
        try:
            model_internal_n_embd = model.n_embd()
            print(f"Model's internal n_embd (model.n_embd()): {model_internal_n_embd}")
        except Exception as e:
            print(f"Error calling model.n_embd(): {e}")
    else:
        print("Model object does not have a callable n_embd method.")

    print("Calling create_embedding with 'hello world'...")
    try:
        # Using core.embedder.create_embedding for a more direct test
        # create_embedding itself uses get_embedding_model_instance, so model
        # should be the same instance
        embedding_output = create_embedding("hello world")
        print(f"Output embedding shape: {embedding_output.shape}")
        print(f"Output embedding dtype: {embedding_output.dtype}")

        if embedding_output.shape == (EMBEDDING_DIMENSION,):
            print(
                f"SUCCESS: Embedding dimension ({embedding_output.shape[0]}) "
                f"matches expected EMBEDDING_DIMENSION ({EMBEDDING_DIMENSION})."
            )
        else:
            print(
                f"FAILURE: Embedding dimension ({embedding_output.shape[0]}) "
                f"does NOT match expected EMBEDDING_DIMENSION ({EMBEDDING_DIMENSION})."
            )

        # Additional check: Is model_internal_n_embd consistent with
        # EMBEDDING_DIMENSION if truncation worked? This depends on whether
        # model.n_embd() reflects the truncated dimension.
        if model_internal_n_embd == EMBEDDING_DIMENSION:
            print(
                f"INFO: model.n_embd() ({model_internal_n_embd}) "
                f"also matches expected EMBEDDING_DIMENSION."
            )
        else:
            print(
                f"WARNING: model.n_embd() ({model_internal_n_embd}) "
                f"does NOT match expected EMBEDDING_DIMENSION "
                f"({EMBEDDING_DIMENSION}). This might be okay if truncation "
                f"works at output level but n_embd() reports original."
            )

    except Exception as e:
        print(f"Error during create_embedding: {e}")
        steadytext_logger.exception("Exception in create_embedding")
else:
    print("Failed to load embedding model.")

print("--- Embedding Dimension Test Finished ---")
