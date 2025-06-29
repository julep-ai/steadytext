import click
import json
import numpy as np


# AIDEV-NOTE: Fixed CLI consistency issue (2025-06-28) - Changed from single --format option
# to individual flags (--json, --numpy, --hex) to match generate command pattern
@click.command()
@click.argument("text", nargs=-1)
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--numpy", "output_numpy", is_flag=True, help="Output as numpy array")
@click.option("--hex", "output_hex", is_flag=True, help="Output as hex string")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["numpy", "json", "hex"]),
    default=None,
    help="Output format (deprecated, use --json/--numpy/--hex)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for deterministic embedding.",
    show_default=True,
)
def embed(text, output_json, output_numpy, output_hex, output_format, seed):
    """Generate embedding vector for text.

    Examples:
        st embed "hello world"
        st embed "hello world" --json
        st embed "text one" "text two" --json
        echo "text to embed" | st embed
    """
    import sys
    import time
    from ...core.embedder import core_embed as create_embedding

    # Determine output format
    if output_format:
        # Legacy --format option
        format_choice = output_format
    elif output_numpy:
        format_choice = "numpy"
    elif output_hex:
        format_choice = "hex"
    else:
        # Default to hex for single text without flags, json for multiple or with --json
        format_choice = "json" if output_json or len(text) > 1 else "hex"

    # Handle input text
    if not text:
        # Read from stdin
        if sys.stdin.isatty():
            click.echo(
                "Error: No input provided. Use 'st embed --help' for usage.", err=True
            )
            sys.exit(1)
        input_text = sys.stdin.read().strip()
    else:
        # Join multiple text arguments
        input_text = " ".join(text)

    if not input_text:
        click.echo("Error: Empty text provided.", err=True)
        sys.exit(1)

    # AIDEV-NOTE: Create embedding directly using core function
    start_time = time.time()
    embedding = create_embedding(input_text, seed=seed)
    elapsed_time = time.time() - start_time

    if format_choice == "numpy":
        # Output as numpy text representation
        np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
        click.echo(np.array2string(embedding, separator=", "))
    elif format_choice == "hex":
        # Output as hex string
        hex_str = embedding.tobytes().hex()
        click.echo(hex_str)
    else:
        # JSON format
        output = {
            "text": input_text,
            "embedding": embedding.tolist(),
            "model": "Qwen3-Embedding-0.6B",
            "usage": {
                "prompt_tokens": len(input_text.split()),
                "total_tokens": len(input_text.split()),
            },
            "dimension": len(embedding),
            "time_taken": elapsed_time,
        }
        click.echo(json.dumps(output))
