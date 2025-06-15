# AIDEV-NOTE: Model downloading and caching from Hugging Face Hub
# Handles download resumption and path validation

from pathlib import Path
from huggingface_hub import hf_hub_download
from ..utils import (
    logger,
    get_cache_dir,
    DEFAULT_GENERATION_MODEL_REPO,
    DEFAULT_EMBEDDING_MODEL_REPO,
    GENERATION_MODEL_FILENAME,
    EMBEDDING_MODEL_FILENAME,
)
from typing import Optional


# AIDEV-NOTE: Core download function with path validation and error handling
def _download_model_if_needed(
    repo_id: str, filename: str, cache_dir: Path
) -> Optional[Path]:
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


def get_generation_model_path() -> Optional[Path]:
    cache = get_cache_dir()
    return _download_model_if_needed(
        DEFAULT_GENERATION_MODEL_REPO, GENERATION_MODEL_FILENAME, cache
    )


def get_embedding_model_path() -> Optional[Path]:
    cache = get_cache_dir()
    return _download_model_if_needed(
        DEFAULT_EMBEDDING_MODEL_REPO, EMBEDDING_MODEL_FILENAME, cache
    )
