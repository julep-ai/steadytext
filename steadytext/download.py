# AIDEV-NOTE: CLI for downloading SteadyText models
# Run with: python -m steadytext.download

import argparse
import sys
from .models.cache import get_generation_model_path, get_embedding_model_path
from .utils import (
    logger,
    get_cache_dir,
    DEFAULT_GENERATION_MODEL_REPO,
    DEFAULT_EMBEDDING_MODEL_REPO,
    GENERATION_MODEL_FILENAME,
    EMBEDDING_MODEL_FILENAME,
)


def download_models(generation=True, embedding=True):
    """Download specified models."""
    cache_dir = get_cache_dir()
    logger.info(f"Using cache directory: {cache_dir}")

    success = True

    if generation:
        logger.info("Downloading generation model...")
        logger.info(f"  Repository: {DEFAULT_GENERATION_MODEL_REPO}")
        logger.info(f"  File: {GENERATION_MODEL_FILENAME}")
        gen_path = get_generation_model_path()
        if gen_path:
            logger.info(f"✓ Generation model ready at: {gen_path}")
        else:
            logger.error("✗ Failed to download generation model")
            success = False

    if embedding:
        logger.info("Downloading embedding model...")
        logger.info(f"  Repository: {DEFAULT_EMBEDDING_MODEL_REPO}")
        logger.info(f"  File: {EMBEDDING_MODEL_FILENAME}")
        emb_path = get_embedding_model_path()
        if emb_path:
            logger.info(f"✓ Embedding model ready at: {emb_path}")
        else:
            logger.error("✗ Failed to download embedding model")
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Download SteadyText models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m steadytext.download              # Download all models
  python -m steadytext.download --generation # Download only generation model
  python -m steadytext.download --embedding  # Download only embedding model
        """,
    )

    parser.add_argument(
        "--generation",
        action="store_true",
        help="Download only the generation model",
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Download only the embedding model",
    )

    args = parser.parse_args()

    # If no specific model requested, download both
    if not args.generation and not args.embedding:
        generation = True
        embedding = True
    else:
        generation = args.generation
        embedding = args.embedding

    logger.info("SteadyText Model Downloader")
    logger.info("==========================")

    success = download_models(generation=generation, embedding=embedding)

    if success:
        logger.info("\nAll requested models downloaded successfully!")
        sys.exit(0)
    else:
        logger.error("\nSome models failed to download. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
