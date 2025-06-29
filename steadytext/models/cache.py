# AIDEV-NOTE: Model downloading and caching from Hugging Face Hub
# Handles download resumption and path validation

import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from ..utils import (
    logger,
    get_cache_dir,
    DEFAULT_EMBEDDING_MODEL_REPO,
    GENERATION_MODEL_REPO,
    GENERATION_MODEL_FILENAME,
    EMBEDDING_MODEL_FILENAME,
)
from typing import Optional


# AIDEV-NOTE: Core download function with path validation, error handling, and dynamic model loading.
def _download_model_if_needed(
    repo_id: str, filename: str, cache_dir: Path
) -> Optional[Path]:
    # AIDEV-NOTE: Check if model downloads are disabled (for testing/collection)
    if os.environ.get("STEADYTEXT_ALLOW_MODEL_DOWNLOADS", "true").lower() == "false":
        logger.debug(f"Model downloads disabled, skipping download of {filename}")
        return None

    # AIDEV-NOTE: Ensure cache directory exists before attempting to download.
    # This is the lazy-initialization of the cache directory.
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(
            f"Failed to create cache directory at {cache_dir}: {e}", exc_info=True
        )
        return None

    model_path = cache_dir / filename
    if not model_path.exists():
        logger.info(
            f"Model {filename} not found in cache. Downloading from {repo_id}..."
        )
        try:
            actual_downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                local_dir=cache_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            if Path(actual_downloaded_path) != model_path:
                logger.warning(
                    f"Model {filename} downloaded to {actual_downloaded_path}, "
                    f"not {model_path}. Using actual path."
                )
                model_path = Path(actual_downloaded_path)

            if not model_path.exists():
                logger.error(
                    f"Model {filename} downloaded but not found at expected "
                    f"path {model_path}."
                )
                return None
            logger.info(f"Model {filename} downloaded successfully to {model_path}.")
        except Exception as e:
            logger.error(
                f"Failed to download model {filename} from {repo_id}: {e}",
                exc_info=True,
            )
            return None
    else:
        logger.debug(f"Model {filename} found in cache: {model_path}")
    return model_path


def get_generation_model_path(
    repo_id: Optional[str] = None, filename: Optional[str] = None
) -> Optional[Path]:
    """Get path to generation model, with support for dynamic model switching.

    AIDEV-NOTE: Now accepts optional parameters to support loading different models.
    Falls back to environment variables or defaults if not specified.

    Args:
        repo_id: Hugging Face repository ID (e.g., "Qwen/Qwen2.5-3B-Instruct-GGUF")
        filename: Model filename (e.g., "qwen2.5-3b-instruct-q8_0.gguf")

    Returns:
        Path to downloaded model or None if download fails
    """
    cache = get_cache_dir()
    # Use provided params or fall back to configured defaults
    repo = repo_id or GENERATION_MODEL_REPO
    fname = filename or GENERATION_MODEL_FILENAME
    return _download_model_if_needed(repo, fname, cache)


def get_embedding_model_path() -> Optional[Path]:
    cache = get_cache_dir()
    return _download_model_if_needed(
        DEFAULT_EMBEDDING_MODEL_REPO, EMBEDDING_MODEL_FILENAME, cache
    )
