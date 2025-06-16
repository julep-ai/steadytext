import click
import sys
import json
import numpy as np
import time

from ...core.embedder import create_embedding


@click.command()
@click.argument("text", default="-", required=False)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["numpy", "json", "hex"]),
    default="json",
    help="Output format",
)
def embed(text: str, output_format: str):
    """Generate embedding vector for text.

    Examples:
        st embed "hello world"
        st embed "hello world" --format numpy
        echo "text to embed" | st embed
    """
    # Handle stdin input
    if text == "-":
        if sys.stdin.isatty():
            click.echo(
                "Error: No input provided. Use 'st embed --help' for usage.", err=True
            )
            sys.exit(1)
        text = sys.stdin.read().strip()

    if not text:
        click.echo("Error: Empty text provided.", err=True)
        sys.exit(1)

    # AIDEV-NOTE: Create embedding directly using core function
    start_time = time.time()
    embedding = create_embedding(text)

    if output_format == "numpy":
        # Output as numpy text representation
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        click.echo(np.array2string(embedding, separator=", "))
    elif output_format == "hex":
        # Output as hex string
        hex_str = embedding.tobytes().hex()
        click.echo(hex_str)
    else:
        # JSON format (default)
        output = {
            "text": text,
            "embedding": embedding.tolist(),
            "dimension": len(embedding),
            "time_taken": time.time() - start_time,
        }
        click.echo(json.dumps(output, indent=2))
