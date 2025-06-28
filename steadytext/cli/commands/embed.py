import click
import json
import numpy as np
import os


# AIDEV-NOTE: Fixed CLI consistency issue (2025-06-28) - Changed from single --format option
# to individual flags (--json, --numpy, --hex) to match generate command pattern
@click.command()
@click.argument("text", nargs=-1)
@click.option(
    "--json",
    "output_format",
    flag_value="json",
    default=True,
    help="JSON output (default)",
)
@click.option("--numpy", "output_format", flag_value="numpy", help="Numpy array output")
@click.option("--hex", "output_format", flag_value="hex", help="Hex-encoded output")
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for deterministic embedding.",
    show_default=True,
)
def embed(text: tuple, output_format: str, seed: int):
    """Generate embedding vector for text."""
    if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
        embedding = np.zeros(1024, dtype=np.float32)
        if output_format == "json":
            output = {
                "embedding": embedding.tolist(),
                "model": "mock_model",
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }
            click.echo(json.dumps(output))
        elif output_format == "numpy":
            click.echo(str(embedding))
        else:  # hex
            click.echo(embedding.tobytes().hex())
        return

    from ...core.embedder import core_embed as create_embedding
    from ...utils import set_deterministic_environment

    set_deterministic_environment(seed)
    embedding = create_embedding(list(text), seed=seed)

    if output_format == "json":
        output = {
            "embedding": embedding.tolist(),
            "model": "embedding_model",
            "usage": {
                "prompt_tokens": len(list(text)),
                "total_tokens": len(list(text)),
            },
        }
        click.echo(json.dumps(output))
    elif output_format == "numpy":
        click.echo(str(embedding))
    else:  # hex
        click.echo(embedding.tobytes().hex())
