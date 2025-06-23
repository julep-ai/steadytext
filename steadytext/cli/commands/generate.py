import click
import sys
import json
import time
from pathlib import Path

from ...core.generator import DeterministicGenerator
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
@click.option("--index-file", type=click.Path(exists=True), help="Use specific index file")
@click.option("--top-k", default=3, help="Number of context chunks to retrieve from index")
@click.pass_context
def generate(
    ctx, prompt: str, output_format: str, stream: bool, logprobs: bool, eos_string: str,
    no_index: bool, index_file: str, top_k: int
):
    """Generate text from a prompt.

    Examples:
        st "write a hello world function"
        st -  # Read from stdin
        echo "explain this" | st
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
        context_text = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
        final_prompt = f"Based on the following context, answer the question.\n\n{context_text}\n\nQuestion: {prompt}\n\nAnswer:"

    # AIDEV-NOTE: Initialize generator once for better performance
    generator = DeterministicGenerator()

    start_time = time.time()

    if stream:
        # Streaming mode
        generated_text = ""
        for token in generator.generate_iter(
            final_prompt, eos_string=eos_string, include_logprobs=logprobs
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
            text, logprobs_data = generator.generate(
                final_prompt, return_logprobs=True, eos_string=eos_string
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
            generated = generator.generate(final_prompt, eos_string=eos_string)

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
