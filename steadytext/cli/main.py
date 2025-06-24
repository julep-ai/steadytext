import click
import sys
import logging

from .commands.generate import generate
from .commands.embed import embed
from .commands.cache import cache
from .commands.models import models
from .commands.vector import vector
from .commands.index import index
from .commands.daemon import daemon


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--version", is_flag=True, help="Show version")
@click.option(
    "--quiet", "-q", is_flag=True, default=True, help="Silence log output (default)"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable informational output")
@click.option(
    "--size",
    type=click.Choice(["small", "medium", "large"]),
    default=None,
    help="Model size (small=0.6B, medium=1.7B, large=4B)",
)
def cli(ctx, version, quiet, verbose, size):
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
        ctx.invoke(generate, prompt="-", size=size)
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Register commands
cli.add_command(generate)
cli.add_command(embed)
cli.add_command(cache)
cli.add_command(models)
cli.add_command(vector)
cli.add_command(index)
cli.add_command(daemon)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
