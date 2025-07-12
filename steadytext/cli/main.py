import click
import sys
import logging

from .commands.generate import generate
from .commands.embed import embed
from .commands.rerank import rerank
from .commands.cache import cache
from .commands.models import models
from .commands.vector import vector
from .commands.index import index
from .commands.daemon import daemon
from .commands.completion import completion


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--version", is_flag=True, help="Show version")
@click.option(
    "--quiet", "-q", is_flag=True, default=True, help="Silence log output (default)"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable informational output")
@click.option(
    "--size",
    type=click.Choice(["small", "large"]),
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
    "--wait",
    is_flag=True,
    help="Wait for full generation before output (disables streaming)",
)
@click.option(
    "--schema",
    default=None,
    help="JSON schema for structured output (can be file path or inline JSON)",
)
@click.option(
    "--regex",
    default=None,
    help="Regular expression pattern for structured output",
)
@click.option(
    "--choices",
    default=None,
    help="Comma-separated list of allowed choices for structured output",
)
def cli(ctx, version, quiet, verbose, size, seed, wait, schema, regex, choices):
    """SteadyText: Deterministic text generation and embedding CLI."""
    # Handle verbosity - verbose overrides quiet
    if verbose:
        quiet = False

    if quiet:
        # Set all steadytext loggers to ERROR level to silence INFO/WARNING logs
        logging.getLogger("steadytext").setLevel(logging.ERROR)
        # Also set llama_cpp logger to ERROR if it exists
        logging.getLogger("llama_cpp").setLevel(logging.ERROR)
        # Set root logger to ERROR to catch any other loggers
        logging.getLogger().setLevel(logging.ERROR)

    if version:
        from .. import __version__

        click.echo(f"steadytext {__version__}")
        ctx.exit(0)

    # Store verbosity flags in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet
    ctx.obj["verbose"] = verbose

    if ctx.invoked_subcommand is None and not sys.stdin.isatty():
        # If no subcommand and input is from pipe, assume generate
        ctx.invoke(
            generate,
            prompt="-",
            size=size,
            seed=seed,
            wait=wait,
            schema=schema,
            regex=regex,
            choices=choices,
        )
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Register commands
cli.add_command(generate)
cli.add_command(embed)
cli.add_command(rerank)
cli.add_command(cache)
cli.add_command(models)
cli.add_command(vector)
cli.add_command(index)
cli.add_command(daemon)
cli.add_command(completion)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
