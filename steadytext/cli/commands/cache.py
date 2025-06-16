import click
import os
import json
from pathlib import Path

from ...disk_backed_frecency_cache import DiskBackedFrecencyCache
from ...utils import get_cache_dir


@click.group()
def cache():
    """Manage SteadyText caches."""
    pass


@cache.command()
def stats():
    """Show cache statistics."""
    cache_dir = get_cache_dir() / "caches"

    stats_data = {
        "generation_cache": {},
        "embedding_cache": {},
        "cache_directory": str(cache_dir),
    }

    # Check generation cache
    gen_cache_path = cache_dir / "generation_cache.pkl"
    if gen_cache_path.exists():
        try:
            gen_cache = DiskBackedFrecencyCache(
                str(gen_cache_path),
                capacity=int(
                    os.environ.get("STEADYTEXT_GENERATION_CACHE_CAPACITY", "256")
                ),
                max_size_mb=float(
                    os.environ.get("STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB", "50.0")
                ),
            )
            stats_data["generation_cache"] = {
                "entries": len(gen_cache.cache),
                "file_size_mb": gen_cache_path.stat().st_size / (1024 * 1024),
                "capacity": gen_cache.capacity,
                "max_size_mb": gen_cache.max_size_mb,
            }
        except Exception as e:
            stats_data["generation_cache"]["error"] = str(e)
    else:
        stats_data["generation_cache"]["status"] = "not found"

    # Check embedding cache
    embed_cache_path = cache_dir / "embedding_cache.pkl"
    if embed_cache_path.exists():
        try:
            embed_cache = DiskBackedFrecencyCache(
                str(embed_cache_path),
                capacity=int(
                    os.environ.get("STEADYTEXT_EMBEDDING_CACHE_CAPACITY", "512")
                ),
                max_size_mb=float(
                    os.environ.get("STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB", "100.0")
                ),
            )
            stats_data["embedding_cache"] = {
                "entries": len(embed_cache.cache),
                "file_size_mb": embed_cache_path.stat().st_size / (1024 * 1024),
                "capacity": embed_cache.capacity,
                "max_size_mb": embed_cache.max_size_mb,
            }
        except Exception as e:
            stats_data["embedding_cache"]["error"] = str(e)
    else:
        stats_data["embedding_cache"]["status"] = "not found"

    click.echo(json.dumps(stats_data, indent=2))


@cache.command()
@click.option("--generation", is_flag=True, help="Clear only generation cache")
@click.option("--embedding", is_flag=True, help="Clear only embedding cache")
@click.confirmation_option(prompt="Are you sure you want to clear the cache(s)?")
def clear(generation: bool, embedding: bool):
    """Clear all caches or specific caches."""
    cache_dir = get_cache_dir() / "caches"

    # If neither flag is set, clear both
    if not generation and not embedding:
        generation = embedding = True

    cleared = []

    if generation:
        gen_cache_path = cache_dir / "generation_cache.pkl"
        if gen_cache_path.exists():
            gen_cache_path.unlink()
            cleared.append("generation")

    if embedding:
        embed_cache_path = cache_dir / "embedding_cache.pkl"
        if embed_cache_path.exists():
            embed_cache_path.unlink()
            cleared.append("embedding")

    if cleared:
        click.echo(f"Cleared caches: {', '.join(cleared)}")
    else:
        click.echo("No caches found to clear")


@cache.command()
@click.argument("output_file", type=click.Path())
def export(output_file: str):
    """Export cache to file."""
    cache_dir = get_cache_dir() / "caches"
    output_path = Path(output_file)

    export_data = {}

    # Export generation cache
    gen_cache_path = cache_dir / "generation_cache.pkl"
    if gen_cache_path.exists():
        try:
            gen_cache = DiskBackedFrecencyCache(
                str(gen_cache_path),
                capacity=int(
                    os.environ.get("STEADYTEXT_GENERATION_CACHE_CAPACITY", "256")
                ),
                max_size_mb=float(
                    os.environ.get("STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB", "50.0")
                ),
            )
            export_data["generation"] = dict(gen_cache.cache)
        except Exception as e:
            click.echo(f"Warning: Could not export generation cache: {e}", err=True)

    # Export embedding cache
    embed_cache_path = cache_dir / "embedding_cache.pkl"
    if embed_cache_path.exists():
        try:
            embed_cache = DiskBackedFrecencyCache(
                str(embed_cache_path),
                capacity=int(
                    os.environ.get("STEADYTEXT_EMBEDDING_CACHE_CAPACITY", "512")
                ),
                max_size_mb=float(
                    os.environ.get("STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB", "100.0")
                ),
            )
            # Convert numpy arrays to lists for JSON serialization
            embed_data = {}
            for key, (value, freq, time) in embed_cache.cache.items():
                embed_data[key] = {
                    "value": value.tolist() if hasattr(value, "tolist") else value,
                    "frequency": freq,
                    "timestamp": time,
                }
            export_data["embedding"] = embed_data
        except Exception as e:
            click.echo(f"Warning: Could not export embedding cache: {e}", err=True)

    # Save to file
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    click.echo(f"Exported cache to {output_path}")


@cache.command("import")
@click.argument("input_file", type=click.Path(exists=True))
def import_cache(input_file: str):
    """Import cache from file."""
    # AIDEV-TODO: Implement cache import functionality
    click.echo("Cache import not yet implemented")
