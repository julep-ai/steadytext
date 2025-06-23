import click
import sys
import json
import time
from pathlib import Path

from ...core.generator import (
    generate as core_generate,
    generate_iter as core_generate_iter,
)
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
@click.option("--stream", is_flag=True, help="Stream tokens as they generate")
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
    type=click.Choice(["small", "medium", "large"]),
    default=None,
    help="Model size (small=0.6B, medium=1.7B, large=4B)",
)
@click.pass_context
def generate(
    ctx,
    prompt: str,
    output_format: str,
    stream: bool,
    logprobs: bool,
    eos_string: str,
    no_index: bool,
    index_file: str,
    top_k: int,
    model: str,
    model_repo: str,
    model_filename: str,
    size: str,
):
    """Generate text from a prompt.

    Examples:
        st "write a hello world function"
        st "quick task" --size small    # Uses Qwen3-0.6B
        st "complex task" --size large   # Uses Qwen3-4B
        st "explain quantum computing" --model qwen2.5-3b
        st -  # Read from stdin
        echo "explain this" | st
        st "complex task" --model-repo Qwen/Qwen2.5-7B-Instruct-GGUF --model-filename qwen2.5-7b-instruct-q8_0.gguf
    """
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
            context_chunks = search_index_for_context(prompt, index_path, top_k)

    # AIDEV-NOTE: Prepare prompt with context if available
    final_prompt = prompt
    if context_chunks:
        # Build context-enhanced prompt
        context_text = "\n\n".join(
            [f"Context {i + 1}:\n{chunk}" for i, chunk in enumerate(context_chunks)]
        )
        final_prompt = f"Based on the following context, answer the question.\n\n{context_text}\n\nQuestion: {prompt}\n\nAnswer:"

    # AIDEV-NOTE: Model switching support - pass model parameters to core functions

    start_time = time.time()

    if stream:
        # Streaming mode
        generated_text = ""
        for token in core_generate_iter(
            final_prompt,
            eos_string=eos_string,
            include_logprobs=logprobs,
            model=model,
            model_repo=model_repo,
            model_filename=model_filename,
            size=size,
        ):
            if logprobs and isinstance(token, dict):
                click.echo(json.dumps(token), nl=True)
            else:
                click.echo(token, nl=False)
                generated_text += token
        click.echo()  # Final newline

        if output_format == "json" and not logprobs:
            # Output metadata after streaming
            metadata = {
                "prompt": prompt,
                "generated": generated_text,
                "time_taken": time.time() - start_time,
                "stream": True,
                "used_index": len(context_chunks) > 0,
                "context_chunks": len(context_chunks),
            }
            click.echo(json.dumps(metadata, indent=2))
    else:
        # Non-streaming mode
        if logprobs:
            text, logprobs_data = core_generate(
                final_prompt,
                return_logprobs=True,
                eos_string=eos_string,
                model=model,
                model_repo=model_repo,
                model_filename=model_filename,
                size=size,
            )
            if output_format == "json":
                metadata = {
                    "prompt": prompt,
                    "generated": text,
                    "logprobs": logprobs_data if logprobs_data is not None else [],
                    "time_taken": time.time() - start_time,
                    "stream": False,
                    "used_index": len(context_chunks) > 0,
                    "context_chunks": len(context_chunks),
                }
                click.echo(json.dumps(metadata, indent=2))
            else:
                # Raw format with logprobs - output as dict
                result_dict = {
                    "text": text,
                    "logprobs": logprobs_data if logprobs_data is not None else [],
                }
                click.echo(json.dumps(result_dict, indent=2))
        else:
            generated = core_generate(
                final_prompt,
                eos_string=eos_string,
                model=model,
                model_repo=model_repo,
                model_filename=model_filename,
                size=size,
            )

            if output_format == "json":
                metadata = {
                    "prompt": prompt,
                    "generated": generated,
                    "time_taken": time.time() - start_time,
                    "stream": False,
                    "used_index": len(context_chunks) > 0,
                    "context_chunks": len(context_chunks),
                }
                click.echo(json.dumps(metadata, indent=2))
            else:
                # Raw format
                click.echo(generated)
