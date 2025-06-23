import click
import sys
import logging

from .commands.generate import generate
from .commands.embed import embed
from .commands.cache import cache
from .commands.models import models
from .commands.vector import vector
from .commands.index import index


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--version", is_flag=True, help="Show version")
@click.option("--quiet", "-q", is_flag=True, help="Silence log output")
def cli(ctx, version, quiet):
    """SteadyText: Deterministic text generation and embedding CLI."""
    if quiet:
        # Set all steadytext loggers to ERROR level to silence INFO/WARNING logs
        logging.getLogger("steadytext").setLevel(logging.ERROR)
        # Also set llama_cpp logger to ERROR if it exists
        logging.getLogger("llama_cpp").setLevel(logging.ERROR)
    
    if version:
        from .. import __version__

        click.echo(f"steadytext {__version__}")
        ctx.exit(0)
    
    # Store quiet flag in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet

    if ctx.invoked_subcommand is None and not sys.stdin.isatty():
        # If no subcommand and input is from pipe, assume generate
        ctx.invoke(generate, prompt="-")
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Register commands
cli.add_command(generate)
cli.add_command(embed)
cli.add_command(cache)
cli.add_command(models)
cli.add_command(vector)
cli.add_command(index)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
