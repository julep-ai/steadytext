import click
import sys
import json
import time

from ...core.generator import DeterministicGenerator


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
@click.pass_context
def generate(
    ctx, prompt: str, output_format: str, stream: bool, logprobs: bool, eos_string: str
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

    # AIDEV-NOTE: Initialize generator once for better performance
    generator = DeterministicGenerator()

    start_time = time.time()

    if stream:
        # Streaming mode
        generated_text = ""
        for token in generator.generate_iter(
            prompt, eos_string=eos_string, include_logprobs=logprobs
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
            }
            click.echo(json.dumps(metadata, indent=2))
    else:
        # Non-streaming mode
        if logprobs:
            text, logprobs_data = generator.generate(
                prompt, return_logprobs=True, eos_string=eos_string
            )
            if output_format == "json":
                metadata = {
                    "prompt": prompt,
                    "generated": text,
                    "logprobs": logprobs_data if logprobs_data is not None else [],
                    "time_taken": time.time() - start_time,
                    "stream": False,
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
            generated = generator.generate(prompt, eos_string=eos_string)

            if output_format == "json":
                metadata = {
                    "prompt": prompt,
                    "generated": generated,
                    "time_taken": time.time() - start_time,
                    "stream": False,
                }
                click.echo(json.dumps(metadata, indent=2))
            else:
                # Raw format
                click.echo(generated)
