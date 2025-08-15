import click
import json
import numpy as np


# AIDEV-NOTE: Fixed CLI consistency issue (2025-06-28) - Changed from single --format option
# to individual flags (--json, --numpy, --hex) to match generate command pattern
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
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
@click.option(
    "--model",
    "-m",
    type=str,
    help="Remote model to use (e.g., openai:text-embedding-3-small, voyageai:voyage-3-lite)",
)
@click.option(
    "--size",
    type=click.Choice(["mini"]),
    default=None,
    help="Model size (mini=130MB BGE model for CI/testing)",
)
@click.option(
    "--unsafe-mode",
    is_flag=True,
    help="Enable unsafe mode for remote models (non-deterministic)",
)
@click.option(
    "--mode",
    type=click.Choice(["query", "passage"]),
    default="query",
    help="Embedding mode for Jina v4 models (query for searches, passage for documents)",
    show_default=True,
)
def embed(
    text,
    output_json,
    output_numpy,
    output_hex,
    output_format,
    seed,
    model,
    size,
    unsafe_mode,
    mode,
):
    """Generate embedding vector for text.

    Examples:
        st embed "hello world"
        st embed "hello world" --json
        st embed "text one" "text two" --json
        echo "text to embed" | st embed
        st embed "search query" --mode query
        st embed "document text" --mode passage
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
    # Now supports remote models with unsafe_mode, mini models, and Jina v4 mode parameter
    import os
    from ...utils import resolve_embedding_model_params

    # Set environment variables for mini model if specified
    original_repo = os.environ.get("STEADYTEXT_EMBEDDING_MODEL_REPO")
    original_filename = os.environ.get("STEADYTEXT_EMBEDDING_MODEL_FILENAME")

    if size == "mini" and not model:
        repo, filename = resolve_embedding_model_params(size=size)
        # Temporarily set environment variables for mini model
        os.environ["STEADYTEXT_EMBEDDING_MODEL_REPO"] = repo
        os.environ["STEADYTEXT_EMBEDDING_MODEL_FILENAME"] = filename

    try:
        start_time = time.time()
        embedding = create_embedding(
            input_text, seed=seed, model=model, unsafe_mode=unsafe_mode, mode=mode
        )
        elapsed_time = time.time() - start_time
    finally:
        # Restore original environment variables
        if size == "mini" and not model:
            if original_repo is not None:
                os.environ["STEADYTEXT_EMBEDDING_MODEL_REPO"] = original_repo
            else:
                os.environ.pop("STEADYTEXT_EMBEDDING_MODEL_REPO", None)
            if original_filename is not None:
                os.environ["STEADYTEXT_EMBEDDING_MODEL_FILENAME"] = original_filename
            else:
                os.environ.pop("STEADYTEXT_EMBEDDING_MODEL_FILENAME", None)

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
        # Determine which model was used
        if model:
            model_name = model
        else:
            model_name = "jina-embeddings-v4-text-retrieval"

        output = {
            "text": input_text,
            "embedding": embedding.tolist(),
            "model": model_name,
            "usage": {
                "prompt_tokens": len(input_text.split()),
                "total_tokens": len(input_text.split()),
            },
            "dimension": len(embedding),
            "time_taken": elapsed_time,
        }
        click.echo(json.dumps(output))
