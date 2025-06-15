import click
import sys
from typing import Optional

from .commands.generate import generate
from .commands.embed import embed
from .commands.cache import cache
from .commands.models import models

@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--version', is_flag=True, help='Show version')
def cli(ctx, version):
    """SteadyText: Deterministic text generation and embedding CLI."""
    if version:
        from .. import __version__
        click.echo(f"steadytext {__version__}")
        ctx.exit(0)
    
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

def main():
    """Main entry point for the CLI."""
    cli()

if __name__ == "__main__":
    main()