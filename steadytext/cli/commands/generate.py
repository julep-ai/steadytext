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
    type=click.Choice(["small", "medium", "large"]),
    default=None,
    help="Model size (small=0.6B, medium=1.7B, large=4B)",
)
@click.option(
    "--think",
    is_flag=True,
    help="Enable Qwen3 thinking mode (shows reasoning process)",
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
    think: bool,
):
    """Generate text from a prompt (streams by default).

    Examples:
        echo "write a hello world function" | st  # Streams output
        echo "quick task" | st --wait            # Waits for full output
        echo "quick task" | st generate --size small    # Uses Qwen3-0.6B
        echo "complex task" | st generate --size large  # Uses Qwen3-4B
        echo "explain quantum computing" | st generate --model qwen2.5-3b
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

    # Streaming is now the default - wait flag disables it
    stream = not wait

    if stream:
        # Streaming mode
        generated_text = ""
        buffer = ""  # Buffer to detect empty think tags
        in_think_tag = False
        think_content = ""

        for token in steady_generate_iter(
            final_prompt,
            eos_string=eos_string,
            include_logprobs=logprobs,
            model=model,
            model_repo=model_repo,
            model_filename=model_filename,
            size=size,
            thinking_mode=think,
        ):
            if logprobs and isinstance(token, dict):
                click.echo(json.dumps(token), nl=True)
            else:
                # Ensure token is a string
                if isinstance(token, dict):
                    # Extract the token text from dict if needed
                    token = token.get("token", "")
                buffer += str(token)

                # Check for think tag patterns
                while True:
                    if not in_think_tag and "<think>" in buffer:
                        # Found opening tag
                        idx = buffer.index("<think>")
                        # Output everything before the tag
                        if idx > 0:
                            click.echo(buffer[:idx], nl=False)
                            generated_text += buffer[:idx]
                        buffer = buffer[idx + 7 :]  # Skip past <think>
                        in_think_tag = True
                        think_content = ""
                    elif in_think_tag and "</think>" in buffer:
                        # Found closing tag
                        idx = buffer.index("</think>")
                        think_content += buffer[:idx]
                        buffer = buffer[idx + 8 :]  # Skip past </think>
                        in_think_tag = False

                        # Only output think tags if they have content
                        if think_content.strip():
                            full_tag = f"<think>{think_content}</think>"
                            click.echo(full_tag, nl=False)
                            generated_text += full_tag
                    else:
                        # No more complete tags to process
                        break

                # If not in a think tag, output buffer content
                if not in_think_tag and buffer and "<think>" not in buffer:
                    click.echo(buffer, nl=False)
                    generated_text += buffer
                    buffer = ""
                elif in_think_tag:
                    # Accumulate think content
                    think_content += buffer
                    buffer = ""

        # Handle any remaining buffer
        if buffer and not in_think_tag:
            click.echo(buffer, nl=False)
            generated_text += buffer
        elif in_think_tag and think_content.strip():
            # Unclosed think tag with content
            full_tag = f"<think>{think_content}"
            click.echo(full_tag, nl=False)
            generated_text += full_tag

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
            result = steady_generate(
                final_prompt,
                return_logprobs=True,
                eos_string=eos_string,
                model=model,
                model_repo=model_repo,
                model_filename=model_filename,
                size=size,
                thinking_mode=think,
            )
            # Unpack the tuple result
            text, logprobs_data = result
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
            generated = steady_generate(
                final_prompt,
                eos_string=eos_string,
                model=model,
                model_repo=model_repo,
                model_filename=model_filename,
                size=size,
                thinking_mode=think,
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
