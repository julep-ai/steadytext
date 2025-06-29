import click
import os
import json
from pathlib import Path
from typing import Dict, Any

from ...cache_manager import get_cache_manager
from ...utils import get_cache_dir
from ...disk_backed_frecency_cache import DiskBackedFrecencyCache


@click.group()
def cache():
    """Manage SteadyText caches."""
    pass


@cache.command()
def path():
    """Show the cache directory path."""
    cache_dir = get_cache_dir() / "caches"
    click.echo(str(cache_dir))


@cache.command()
def status():
    """Show cache status."""
    cache_manager = get_cache_manager()

    # Force cache initialization to get proper stats
    cache_manager.get_generation_cache()
    cache_manager.get_embedding_cache()

    stats_data = cache_manager.get_cache_stats()

    click.echo("Generation Cache:")
    gen_stats = stats_data.get("generation", {})
    click.echo(f"  {gen_stats.get('size', 0)} entries")
    click.echo(f"  Capacity: {gen_stats.get('capacity', 0)}")

    click.echo("\nEmbedding Cache:")
    embed_stats = stats_data.get("embedding", {})
    click.echo(f"  {embed_stats.get('size', 0)} entries")
    click.echo(f"  Capacity: {embed_stats.get('capacity', 0)}")


@cache.command()
def stats():
    """Show cache statistics."""
    cache_dir = get_cache_dir() / "caches"

    # AIDEV-NOTE: Use the centralized cache manager for consistent stats. Now uses the SQLite backend.
    cache_manager = get_cache_manager()

    # Force cache initialization to get proper stats
    cache_manager.get_generation_cache()
    cache_manager.get_embedding_cache()

    stats_data = cache_manager.get_cache_stats()

    # Add cache directory info
    stats_data["cache_directory"] = str(cache_dir)

    # Check for actual database files
    gen_db_path = cache_dir / "generation_cache.db"
    embed_db_path = cache_dir / "embedding_cache.db"

    if gen_db_path.exists():
        stats_data.setdefault("generation", {})["file_size_mb"] = (
            gen_db_path.stat().st_size / (1024 * 1024)
        )

    if embed_db_path.exists():
        stats_data.setdefault("embedding", {})["file_size_mb"] = (
            embed_db_path.stat().st_size / (1024 * 1024)
        )

    click.echo(json.dumps(stats_data, indent=2))


@cache.command()
@click.option("--generation", is_flag=True, help="Clear only generation cache")
@click.option("--embedding", is_flag=True, help="Clear only embedding cache")
@click.confirmation_option(prompt="Are you sure you want to clear the cache(s)?")
def clear(generation: bool, embedding: bool):
    """Clear all caches or specific caches."""
    cache_manager = get_cache_manager()

    # If neither flag is set, clear both
    if not generation and not embedding:
        cache_manager.clear_all_caches()
        click.echo("All caches cleared")
        return

    cleared = []

    if generation:
        try:
            gen_cache = cache_manager.get_generation_cache()
            gen_cache.clear()
            cleared.append("generation")
        except Exception as e:
            click.echo(f"Error clearing generation cache: {e}", err=True)

    if embedding:
        try:
            embed_cache = cache_manager.get_embedding_cache()
            embed_cache.clear()
            cleared.append("embedding")
        except Exception as e:
            click.echo(f"Error clearing embedding cache: {e}", err=True)

    if cleared:
        click.echo(f"Cleared caches: {', '.join(cleared)}")
    else:
        click.echo("No caches were cleared")


@cache.command()
@click.argument("output_file", type=click.Path())
def export(output_file: str):
    """Export cache to file."""
    cache_dir = get_cache_dir().parent / "caches"
    output_path = Path(output_file)

    export_data = {"version": "1.0", "caches": {}}

    # Export generation cache
    try:
        DiskBackedFrecencyCache(
            capacity=int(os.environ.get("STEADYTEXT_GENERATION_CACHE_CAPACITY", "256")),
            cache_name="generation_cache",
            max_size_mb=float(
                os.environ.get("STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB", "50.0")
            ),
            cache_dir=cache_dir,
        )

        # Collect all cache entries
        gen_entries: Dict[str, Any] = {}
        # Note: We need a way to iterate cache entries. For now, we'll document this limitation
        click.echo(
            "Note: Cache export currently exports an empty structure. Full export coming soon."
        )
        export_data["caches"]["generation"] = gen_entries
    except Exception as e:
        click.echo(f"Warning: Could not export generation cache: {e}", err=True)

    # Export embedding cache
    try:
        DiskBackedFrecencyCache(
            capacity=int(os.environ.get("STEADYTEXT_EMBEDDING_CACHE_CAPACITY", "512")),
            cache_name="embedding_cache",
            max_size_mb=float(
                os.environ.get("STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB", "100.0")
            ),
            cache_dir=cache_dir,
        )
        # Collect all cache entries
        embed_entries: Dict[str, Any] = {}
        export_data["caches"]["embedding"] = embed_entries
    except Exception as e:
        click.echo(f"Warning: Could not export embedding cache: {e}", err=True)

    # Save to file
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    click.echo(f"Exported cache structure to {output_path}")


@cache.command("import")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--merge/--replace",
    default=True,
    help="Merge with existing cache (default) or replace entirely",
)
def import_cache(input_file: str, merge: bool):
    """Import cache from file."""
    # AIDEV-NOTE: A basic cache import implementation. Currently supports importing cache structure but not entries.

    import numpy as np

    cache_dir = get_cache_dir().parent / "caches"
    input_path = Path(input_file)

    if not input_path.exists():
        click.echo(f"Error: Input file not found: {input_path}", err=True)
        return

    try:
        with open(input_path, "r") as f:
            import_data = json.load(f)
    except Exception as e:
        click.echo(f"Error reading import file: {e}", err=True)
        return

    # Check version compatibility
    version = import_data.get("version", "0.0")
    if version != "1.0":
        click.echo(
            f"Warning: Import file version {version} may not be compatible", err=True
        )

    imported_caches = []

    # Import generation cache entries if present
    if "caches" in import_data and "generation" in import_data["caches"]:
        gen_entries = import_data["caches"]["generation"]
        if gen_entries:
            try:
                gen_cache = DiskBackedFrecencyCache(
                    capacity=int(
                        os.environ.get("STEADYTEXT_GENERATION_CACHE_CAPACITY", "256")
                    ),
                    cache_name="generation_cache",
                    max_size_mb=float(
                        os.environ.get(
                            "STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB", "50.0"
                        )
                    ),
                    cache_dir=cache_dir,
                )

                if not merge:
                    gen_cache.clear()

                # Import each entry
                for key, value in gen_entries.items():
                    gen_cache.set(key, value)

                imported_caches.append(f"generation ({len(gen_entries)} entries)")
            except Exception as e:
                click.echo(f"Warning: Could not import generation cache: {e}", err=True)

    # Import embedding cache entries if present
    if "caches" in import_data and "embedding" in import_data["caches"]:
        embed_entries = import_data["caches"]["embedding"]
        if embed_entries:
            try:
                embed_cache = DiskBackedFrecencyCache(
                    capacity=int(
                        os.environ.get("STEADYTEXT_EMBEDDING_CACHE_CAPACITY", "512")
                    ),
                    cache_name="embedding_cache",
                    max_size_mb=float(
                        os.environ.get(
                            "STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB", "100.0"
                        )
                    ),
                    cache_dir=cache_dir,
                )

                if not merge:
                    embed_cache.clear()

                # Import each entry, converting lists back to numpy arrays
                for key, entry_data in embed_entries.items():
                    if isinstance(entry_data, dict) and "value" in entry_data:
                        value = entry_data["value"]
                        if isinstance(value, list):
                            value = np.array(value, dtype=np.float32)
                    else:
                        value = entry_data
                    embed_cache.set(key, value)

                imported_caches.append(f"embedding ({len(embed_entries)} entries)")
            except Exception as e:
                click.echo(f"Warning: Could not import embedding cache: {e}", err=True)

    # Handle legacy format (direct cache dictionaries)
    elif "generation" in import_data or "embedding" in import_data:
        click.echo("Detected legacy cache format, attempting import...")

        if "generation" in import_data:
            gen_entries = import_data["generation"]
            if gen_entries:
                try:
                    gen_cache = DiskBackedFrecencyCache(
                        capacity=int(
                            os.environ.get(
                                "STEADYTEXT_GENERATION_CACHE_CAPACITY", "256"
                            )
                        ),
                        cache_name="generation_cache",
                        max_size_mb=float(
                            os.environ.get(
                                "STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB", "50.0"
                            )
                        ),
                        cache_dir=cache_dir,
                    )

                    if not merge:
                        gen_cache.clear()

                    for key, value in gen_entries.items():
                        gen_cache.set(key, value)

                    imported_caches.append(f"generation ({len(gen_entries)} entries)")
                except Exception as e:
                    click.echo(
                        f"Warning: Could not import generation cache: {e}", err=True
                    )

        if "embedding" in import_data:
            embed_entries = import_data["embedding"]
            if embed_entries:
                try:
                    embed_cache = DiskBackedFrecencyCache(
                        capacity=int(
                            os.environ.get("STEADYTEXT_EMBEDDING_CACHE_CAPACITY", "512")
                        ),
                        cache_name="embedding_cache",
                        max_size_mb=float(
                            os.environ.get(
                                "STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB", "100.0"
                            )
                        ),
                        cache_dir=cache_dir,
                    )

                    if not merge:
                        embed_cache.clear()

                    for key, entry_data in embed_entries.items():
                        if isinstance(entry_data, dict) and "value" in entry_data:
                            value = entry_data["value"]
                            if isinstance(value, list):
                                value = np.array(value, dtype=np.float32)
                        else:
                            value = entry_data
                        embed_cache.set(key, value)

                    imported_caches.append(f"embedding ({len(embed_entries)} entries)")
                except Exception as e:
                    click.echo(
                        f"Warning: Could not import embedding cache: {e}", err=True
                    )

    if imported_caches:
        mode = "Merged into" if merge else "Replaced"
        click.echo(f"{mode} caches: {', '.join(imported_caches)}")
    else:
        click.echo("No cache entries found to import")
