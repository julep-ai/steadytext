import click
import json

from ...utils import (
    get_cache_dir,
    DEFAULT_GENERATION_MODEL_REPO,
    DEFAULT_EMBEDDING_MODEL_REPO,
    GENERATION_MODEL_FILENAME,
    EMBEDDING_MODEL_FILENAME,
)
from ...models.cache import get_generation_model_path, get_embedding_model_path

# Define model information structure for CLI commands
MODELS = {
    "generation": {
        "filename": GENERATION_MODEL_FILENAME,
        "repo_id": DEFAULT_GENERATION_MODEL_REPO,
    },
    "embedding": {
        "filename": EMBEDDING_MODEL_FILENAME,
        "repo_id": DEFAULT_EMBEDDING_MODEL_REPO,
    },
}


@click.group()
def models():
    """Manage SteadyText models."""
    pass


@models.command()
def status():
    """Check model download status."""
    model_dir = get_cache_dir()
    status_data = {"model_directory": str(model_dir), "models": {}}

    for model_type, model_info in MODELS.items():
        model_path = model_dir / model_info["filename"]
        status_data["models"][model_type] = {
            "filename": model_info["filename"],
            "repo_id": model_info["repo_id"],
            "downloaded": model_path.exists(),
            "size_mb": model_path.stat().st_size / (1024 * 1024)
            if model_path.exists()
            else None,
        }

    click.echo(json.dumps(status_data, indent=2))


@models.command()
def download():
    """Pre-download models."""
    click.echo("Downloading models...")

    # Download generation model
    click.echo("Checking generation model...", nl=False)
    try:
        path = get_generation_model_path()
        if path:
            click.echo(" ✓ Ready")
        else:
            click.echo(" ✗ Failed to download")
    except Exception as e:
        click.echo(f" ✗ Failed: {e}")

    # Download embedding model
    click.echo("Checking embedding model...", nl=False)
    try:
        path = get_embedding_model_path()
        if path:
            click.echo(" ✓ Ready")
        else:
            click.echo(" ✗ Failed to download")
    except Exception as e:
        click.echo(f" ✗ Failed: {e}")


@models.command()
def path():
    """Show model cache directory."""
    click.echo(str(get_cache_dir()))
