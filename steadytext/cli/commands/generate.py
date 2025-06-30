import click
import sys
import json
import time
from pathlib import Path

from ... import generate as steady_generate, generate_iter as steady_generate_iter
from .index import search_index_for_context, get_default_index_path


@click.command()
@click.argument("prompt", default="-", required=False)
@click.option(
    "--raw",
    "output_format",
    flag_value="raw",
    default=True,
    help="No formatting, just generated text (default)",
)
@click.option(
    "--json", "output_format", flag_value="json", help="JSON output with metadata"
)
@click.option(
    "--wait",
    is_flag=True,
    help="Wait for full generation before output (disables streaming)",
)
@click.option("--logprobs", is_flag=True, help="Include log probabilities in output")
@click.option(
    "--eos-string",
    default="[EOS]",
    help="Custom end-of-sequence string (default: [EOS] for model's default)",
)
@click.option("--no-index", is_flag=True, help="Disable automatic index search")
@click.option(
    "--index-file", type=click.Path(exists=True), help="Use specific index file"
)
@click.option(
    "--top-k", default=3, help="Number of context chunks to retrieve from index"
)
@click.option(
    "--quiet", is_flag=True, default=True, help="Silence informational output (default)"
)
@click.option("--verbose", is_flag=True, help="Enable informational output")
@click.option(
    "--model", default=None, help="Model name from registry (e.g., 'qwen2.5-3b')"
)
@click.option(
    "--model-repo",
    default=None,
    help="Custom model repository (e.g., 'Qwen/Qwen2.5-3B-Instruct-GGUF')",
)
@click.option(
    "--model-filename",
    default=None,
    help="Custom model filename (e.g., 'qwen2.5-3b-instruct-q8_0.gguf')",
)
@click.option(
    "--size",
    type=click.Choice(["small", "large"]),
    default=None,
    help="Model size (small=2B, large=4B)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Seed for deterministic generation.",
    show_default=True,
)
@click.option(
    "--max-new-tokens",
    type=int,
    default=None,
    help="Maximum number of new tokens to generate.",
)
@click.pass_context
def generate(
    ctx,
    prompt: str,
    output_format: str,
    wait: bool,
    logprobs: bool,
    eos_string: str,
    no_index: bool,
    index_file: str,
    top_k: int,
    quiet: bool,
    verbose: bool,
    model: str,
    model_repo: str,
    model_filename: str,
    size: str,
    seed: int,
    max_new_tokens: int,
):
    """Generate text from a prompt (streams by default).

    Examples:
        echo "write a hello world function" | st  # Streams output
        echo "quick task" | st --wait            # Waits for full output
        echo "quick task" | st generate --size small    # Uses Gemma-3n-2B
        echo "complex task" | st generate --size large  # Uses Gemma-3n-4B
        echo "explain quantum computing" | st generate --model gemma-3n-2b
        st -  # Read from stdin
        echo "explain this" | st
        echo "complex task" | st generate --model-repo Qwen/Qwen2.5-7B-Instruct-GGUF --model-filename qwen2.5-7b-instruct-q8_0.gguf
    """
    # Handle verbosity - verbose overrides quiet
    if verbose:
        quiet = False

    # Configure logging based on quiet/verbose flags
    if quiet:
        import logging

        logging.getLogger("steadytext").setLevel(logging.ERROR)
        logging.getLogger("llama_cpp").setLevel(logging.ERROR)
    # Handle stdin input
    if prompt == "-":
        if sys.stdin.isatty():
            click.echo("Error: No input provided. Use 'st --help' for usage.", err=True)
            sys.exit(1)
        prompt = sys.stdin.read().strip()

    if not prompt:
        click.echo("Error: Empty prompt provided.", err=True)
        sys.exit(1)

    # AIDEV-NOTE: Search index for context unless disabled
    context_chunks = []
    if not no_index:
        index_path = Path(index_file) if index_file else get_default_index_path()
        if index_path:
            context_chunks = search_index_for_context(
                prompt, index_path, top_k, seed=seed
            )

    # AIDEV-NOTE: Prepare prompt with context if available
    final_prompt = prompt
    if context_chunks:
        # Build context-enhanced prompt
        context_text = "\n\n".join(
            [f"Context {i + 1}:\n{chunk}" for i, chunk in enumerate(context_chunks)]
        )
        final_prompt = f"Based on the following context, answer the question.\n\n{context_text}\n\nQuestion: {prompt}\n\nAnswer:"
        click.echo(f"Final prompt: {final_prompt}", err=True)

    # AIDEV-NOTE: Model switching support - pass model parameters to core functions

    start_time = time.time()

    # Streaming is now the default - wait flag disables it
    stream = not wait

    if stream:
        # Streaming mode
        generated_text = ""
        logprobs_tokens = []

        for token in steady_generate_iter(
            final_prompt,
            max_new_tokens=max_new_tokens,
            eos_string=eos_string,
            include_logprobs=logprobs,
            model=model,
            model_repo=model_repo,
            model_filename=model_filename,
            size=size,
            seed=seed,
        ):
            if logprobs and isinstance(token, dict):
                # Handle logprobs output
                if output_format == "json":
                    logprobs_tokens.append(token)
                    # Also accumulate the text part of the token
                    generated_text += token.get("token", "")
                else:
                    # For raw output with logprobs, print each token's JSON
                    click.echo(json.dumps(token), nl=True)
            else:
                # Handle raw text output
                token_str = str(token)
                if output_format != "json":
                    click.echo(token_str, nl=False)
                generated_text += token_str

        if output_format == "json":
            # For JSON format in streaming mode, output only the JSON
            metadata = {
                "text": generated_text,
                "model": model or "gemma-3n-E2B-it-GGUF",
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": len(prompt.split()) + len(generated_text.split()),
                },
                "prompt": prompt,
                "generated": generated_text,
                "time_taken": time.time() - start_time,
                "stream": True,
                "used_index": len(context_chunks) > 0,
                "context_chunks": len(context_chunks),
            }
            if logprobs:
                # In fallback mode, logprobs will be None
                # Extract just the logprobs values from token objects
                if logprobs_tokens and all(
                    token.get("logprobs") is None for token in logprobs_tokens
                ):
                    metadata["logprobs"] = None
                else:
                    metadata["logprobs"] = (
                        [token.get("logprobs") for token in logprobs_tokens]
                        if logprobs_tokens
                        else None
                    )
            click.echo(json.dumps(metadata, indent=2))
    else:
        # Non-streaming mode
        if logprobs:
            result = steady_generate(
                final_prompt,
                max_new_tokens=max_new_tokens,
                return_logprobs=True,
                eos_string=eos_string,
                model=model,
                model_repo=model_repo,
                model_filename=model_filename,
                size=size,
                seed=seed,
            )
            # Unpack the tuple result
            if result is not None and isinstance(result, tuple):
                text, logprobs_data = result
            else:
                text, logprobs_data = None, None
            if output_format == "json":
                generated_text = ""
                for token in steady_generate_iter(
                    final_prompt,
                    eos_string=eos_string,
                    include_logprobs=logprobs,
                    model=model,
                    model_repo=model_repo,
                    model_filename=model_filename,
                    size=size,
                    seed=seed,
                ):
                    generated_text += str(
                        token.get("token", "") if isinstance(token, dict) else token
                    )

                # After collecting all text, format the final JSON output
                metadata = {
                    "text": text,
                    "model": model or "gemma-3n-E2B-it-GGUF",
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(text.split()) if text else 0,
                        "total_tokens": len(prompt.split())
                        + (len(text.split()) if text else 0),
                    },
                    "logprobs": logprobs_data,
                    "prompt": prompt,
                    "generated": text,
                    "time_taken": time.time() - start_time,
                    "stream": False,
                    "used_index": len(context_chunks) > 0,
                    "context_chunks": len(context_chunks),
                }
                click.echo(json.dumps(metadata))

            else:
                click.echo(json.dumps({"text": text, "logprobs": logprobs_data}))
        else:
            # Non-logprobs mode
            generated = steady_generate(
                final_prompt,
                max_new_tokens=max_new_tokens,
                eos_string=eos_string,
                model=model,
                model_repo=model_repo,
                model_filename=model_filename,
                size=size,
                seed=seed,
            )
            if output_format == "json":
                metadata = {
                    "text": generated,
                    "model": model or "gemma-3n-E2B-it-GGUF",
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(generated.split()),
                        "total_tokens": len(prompt.split()) + len(generated.split()),
                    },
                    "prompt": prompt,
                    "generated": generated,
                    "time_taken": time.time() - start_time,
                    "stream": False,
                    "used_index": len(context_chunks) > 0,
                    "context_chunks": len(context_chunks),
                }
                click.echo(json.dumps(metadata, indent=2))
            else:
                click.echo(generated)
