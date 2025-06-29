import click
import json
from typing import Optional

from ...utils import (
    get_cache_dir,
    DEFAULT_GENERATION_MODEL_REPO,
    DEFAULT_EMBEDDING_MODEL_REPO,
    GENERATION_MODEL_FILENAME,
    EMBEDDING_MODEL_FILENAME,
    MODEL_REGISTRY,
    SIZE_TO_MODEL,
    resolve_model_params,
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


@models.command("list")
def list_models():
    """Check model download status."""
    model_dir = get_cache_dir()
    status_data = {"model_directory": str(model_dir), "models": {}}

    # Show default models
    for model_type, model_info in MODELS.items():
        model_path = model_dir / model_info["filename"]
        status_data["models"][model_type] = {
            "filename": model_info["filename"],
            "repo_id": model_info["repo_id"],
            "downloaded": model_path.exists(),
            "size_mb": (
                model_path.stat().st_size / (1024 * 1024)
                if model_path.exists()
                else None
            ),
            "default": True,
        }

    # Show all available generation models from registry
    status_data["available_generation_models"] = {}
    for model_name, model_info in MODEL_REGISTRY.items():
        model_path = model_dir / model_info["filename"]
        status_data["available_generation_models"][model_name] = {
            "filename": model_info["filename"],
            "repo_id": model_info["repo"],
            "downloaded": model_path.exists(),
            "size_mb": (
                model_path.stat().st_size / (1024 * 1024)
                if model_path.exists()
                else None
            ),
        }

    # Show size mappings
    status_data["size_mappings"] = SIZE_TO_MODEL

    click.echo(json.dumps(status_data, indent=2))


@models.command()
@click.option(
    "--size",
    type=click.Choice(["small", "large"]),
    help="Download specific model size (small=2B, large=4B)",
)
@click.option(
    "--model",
    help="Download specific model from registry (e.g., 'qwen2.5-3b')",
)
@click.option(
    "--all",
    is_flag=True,
    help="Download all available models",
)
def download(size: Optional[str], model: Optional[str], all: bool):
    """Pre-download models."""
    import os

    if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
        click.echo("Preloading models... (skipped in test environment)")
        return

    if all:
        click.echo("Downloading all available models...")
        # Download all models from registry
        for model_name, model_info in MODEL_REGISTRY.items():
            click.echo(f"Downloading {model_name} ({model_info['repo']})...", nl=False)
            try:
                path = get_generation_model_path(
                    model_info["repo"], model_info["filename"]
                )
                if path:
                    click.echo(" ✓ Ready")
                else:
                    click.echo(" ✗ Failed to download")
            except Exception as e:
                click.echo(f" ✗ Failed: {e}")

        # Also download embedding model
        click.echo("Downloading embedding model...", nl=False)
        try:
            path = get_embedding_model_path()
            if path:
                click.echo(" ✓ Ready")
            else:
                click.echo(" ✗ Failed to download")
        except Exception as e:
            click.echo(f" ✗ Failed: {e}")
    elif size or model:
        # Download specific model
        if size and model:
            click.echo("Error: Cannot specify both --size and --model", err=True)
            return

        try:
            if model:
                # Download specific model by name
                if model not in MODEL_REGISTRY:
                    available = ", ".join(sorted(MODEL_REGISTRY.keys()))
                    click.echo(
                        f"Error: Unknown model '{model}'. Available models: {available}",
                        err=True,
                    )
                    return
                repo_id = MODEL_REGISTRY[model]["repo"]
                filename = MODEL_REGISTRY[model]["filename"]
                click.echo(f"Downloading {model} ({repo_id})...", nl=False)
            else:
                # Download model by size
                repo_id, filename = resolve_model_params(size=size)
                click.echo(f"Downloading {size} model ({repo_id})...", nl=False)

            path = get_generation_model_path(repo_id, filename)
            if path:
                click.echo(" ✓ Ready")
            else:
                click.echo(" ✗ Failed to download")
        except Exception as e:
            click.echo(f" ✗ Failed: {e}")
    else:
        # Default behavior - download default models
        click.echo("Downloading default models...")

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


@models.command()
def list():
    """List available models."""
    # Show size shortcuts
    click.echo("Size Shortcuts:")
    for size, model_name in SIZE_TO_MODEL.items():
        model_info = MODEL_REGISTRY.get(model_name, {})
        click.echo(f"  {size} → {model_name}")

    click.echo("\nAvailable Models:")
    for model_name, model_info in sorted(MODEL_REGISTRY.items()):
        click.echo(f"  {model_name}")
        click.echo(f"    Repository: {model_info['repo']}")
        click.echo(f"    Filename: {model_info['filename']}")


@models.command()
@click.option(
    "--size", type=click.Choice(["small", "large"]), help="Model size to preload"
)
@click.pass_context
def preload(ctx, size: Optional[str]):
    """Preload models into memory."""
    # Check if we're in test environment
    import os

    if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
        click.echo("Preloading models... (skipped in test environment)")
        return

    # In normal environment, we would preload models
    click.echo("Preloading models...")
    try:
        # Import the actual model loading functions
        from ...models.loader import (
            get_generator_model_instance,
            get_embedding_model_instance,
        )

        # Preload generation model
        if size:
            repo_id, filename = resolve_model_params(size=size)
            click.echo(f"Loading {size} generation model...", nl=False)
        else:
            click.echo("Loading default generation model...", nl=False)

        gen_model = get_generator_model_instance(
            repo_id=repo_id if size else None, filename=filename if size else None
        )
        if gen_model:
            click.echo(" ✓")
        else:
            click.echo(" ✗ (using fallback)")

        # Preload embedding model
        click.echo("Loading embedding model...", nl=False)
        embed_model = get_embedding_model_instance()
        if embed_model:
            click.echo(" ✓")
        else:
            click.echo(" ✗ (using fallback)")

        click.echo("Models preloaded successfully")
    except Exception as e:
        click.echo(f"\nError preloading models: {e}", err=True)


@models.command()
@click.option(
    "--size",
    type=click.Choice(["small", "large"]),
    help="Delete specific model size (small=2B, large=4B)",
)
@click.option(
    "--model",
    help="Delete specific model from registry (e.g., 'qwen2.5-3b')",
)
@click.option(
    "--all",
    is_flag=True,
    help="Delete all available models",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force deletion without confirmation",
)
def delete(size: Optional[str], model: Optional[str], all: bool, force: bool):
    """Delete cached models."""
    model_dir = get_cache_dir()

    def _delete_file(path, model_name):
        if path.exists():
            if force or click.confirm(f"Delete {model_name} ({path.name})?"):
                try:
                    path.unlink()
                    click.echo(f"Deleted {model_name} ({path.name})")
                except OSError as e:
                    click.echo(f"Error deleting {path.name}: {e}", err=True)
        else:
            click.echo(f"{model_name} ({path.name}) not found.")

    if all:
        if force or click.confirm("Delete all cached models?"):
            # Delete all generation models
            for model_name, model_info in MODEL_REGISTRY.items():
                model_path = model_dir / model_info["filename"]
                _delete_file(model_path, model_name)
            # Delete embedding model
            embed_path = model_dir / EMBEDDING_MODEL_FILENAME
            _delete_file(embed_path, "embedding model")
    elif size or model:
        if size and model:
            click.echo("Error: Cannot specify both --size and --model", err=True)
            return

        if model:
            if model not in MODEL_REGISTRY:
                available = ", ".join(sorted(MODEL_REGISTRY.keys()))
                click.echo(
                    f"Error: Unknown model '{model}'. Available models: {available}",
                    err=True,
                )
                return
            filename = MODEL_REGISTRY[model]["filename"]
            model_path = model_dir / filename
            _delete_file(model_path, model)
        else:  # size
            repo_id, filename = resolve_model_params(size=size)
            model_path = model_dir / filename
            _delete_file(model_path, f"{size} model")
    else:
        click.echo("Specify a model to delete or use --all.", err=True)
