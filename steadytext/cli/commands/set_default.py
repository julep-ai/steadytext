"""
Set-default command for managing persistent CLI defaults.

This command allows users to set, view, and reset default parameters
for SteadyText CLI commands.
"""

import click
from typing import Dict, Any

from ...config import get_config_manager, get_config_file


@click.group(name="set-default", invoke_without_command=True)
@click.option(
    "--reset-all", is_flag=True, help="Reset all stored defaults for all commands"
)
@click.pass_context
def set_default(ctx, reset_all):
    """Set, view, or reset default parameters for CLI commands.

    Examples:
        # Set defaults for generate command
        st set-default generate --model gemma-3n-2b --size large

        # View current defaults for generate
        st set-default generate --show

        # Reset generate defaults to built-in values
        st set-default generate

        # Reset all defaults
        st set-default --reset-all
    """
    if reset_all:
        config_manager = get_config_manager()
        config_manager.reset_all_defaults()
        click.echo("All defaults have been reset.")
        return

    # If no subcommand, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@set_default.command()
@click.option("--show", is_flag=True, help="Show current defaults")
@click.option("--model", help="Model name from registry")
@click.option("--model-repo", help="Custom model repository")
@click.option("--model-filename", help="Custom model filename")
@click.option("--size", type=click.Choice(["small", "large"]), help="Model size")
@click.option("--seed", type=int, help="Seed for deterministic generation")
@click.option("--max-new-tokens", type=int, help="Maximum number of new tokens")
@click.option("--wait", is_flag=True, help="Wait for full generation before output")
@click.option("--logprobs", is_flag=True, help="Include log probabilities")
@click.option("--eos-string", help="Custom end-of-sequence string")
@click.option("--no-index", is_flag=True, help="Disable automatic index search")
@click.option("--index-file", help="Use specific index file")
@click.option("--top-k", type=int, help="Number of context chunks to retrieve")
@click.option("--schema", help="JSON schema for structured output")
@click.option("--regex", help="Regular expression pattern for structured output")
@click.option("--choices", help="Comma-separated list of allowed choices")
@click.option("--unsafe-mode", is_flag=True, help="Enable remote models")
@click.option("--raw", "output_format_raw", is_flag=True, help="Raw output format")
@click.option("--json", "output_format_json", is_flag=True, help="JSON output format")
@click.option("--quiet", is_flag=True, help="Silence informational output")
@click.option("--verbose", is_flag=True, help="Enable informational output")
def generate(show, **kwargs):
    """Set defaults for the generate command."""
    _handle_command_defaults("generate", show, kwargs)


@set_default.command()
@click.option("--show", is_flag=True, help="Show current defaults")
@click.option("--seed", type=int, help="Seed for deterministic embedding")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--numpy", "output_numpy", is_flag=True, help="Output as numpy array")
@click.option("--hex", "output_hex", is_flag=True, help="Output as hex string")
def embed(show, **kwargs):
    """Set defaults for the embed command."""
    _handle_command_defaults("embed", show, kwargs)


@set_default.command()
@click.option("--show", is_flag=True, help="Show current defaults")
@click.option("--seed", type=int, help="Seed for deterministic reranking")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--scores/--no-scores", help="Include relevance scores in output")
@click.option("--task", help="Task description for reranking")
@click.option("--top-k", type=int, help="Return only top K documents")
@click.option("--file", "doc_file", help="Read documents from file")
def rerank(show, **kwargs):
    """Set defaults for the rerank command."""
    _handle_command_defaults("rerank", show, kwargs)


@set_default.command()
@click.option("--show", is_flag=True, help="Show current defaults")
@click.option("--seed", type=int, help="Seed for deterministic operations")
@click.option("--chunk-size", type=int, help="Size of text chunks")
@click.option("--chunk-overlap", type=int, help="Overlap between chunks")
@click.option("--batch-size", type=int, help="Batch size for processing")
def index(show, **kwargs):
    """Set defaults for the index command."""
    _handle_command_defaults("index", show, kwargs)


def _handle_command_defaults(command: str, show: bool, params: Dict[str, Any]) -> None:
    """Handle setting, showing, or resetting defaults for a command."""
    config_manager = get_config_manager()

    if show:
        # Show current defaults
        defaults = config_manager.get_command_defaults(command)
        if defaults:
            click.echo(f"Current defaults for '{command}' command:")
            for key, value in defaults.items():
                click.echo(f"  --{key.replace('_', '-')}: {value}")
        else:
            click.echo(f"No defaults set for '{command}' command.")

        # Also show config file location
        config_file = get_config_file()
        click.echo(f"\nConfiguration file: {config_file}")
        return

    # Filter out None values and the 'show' parameter
    filtered_params = {k: v for k, v in params.items() if v is not None and k != "show"}

    # Handle special cases for flag parameters
    # Convert flag parameters to their appropriate values
    flag_params = {}
    for key, value in filtered_params.items():
        if isinstance(value, bool):
            if value:  # Only store True flags, False is the default
                flag_params[key] = value
        else:
            flag_params[key] = value

    if not flag_params:
        # No parameters provided - reset to defaults
        config_manager.reset_command_defaults(command)
        click.echo(f"Defaults for '{command}' command have been reset.")
    else:
        # Set the new defaults
        config_manager.set_command_defaults(command, flag_params)
        click.echo(f"Defaults for '{command}' command have been updated:")
        for key, value in flag_params.items():
            click.echo(f"  --{key.replace('_', '-')}: {value}")

    # Show config file location
    config_file = get_config_file()
    click.echo(f"\nConfiguration file: {config_file}")


@set_default.command()
@click.option("--show", is_flag=True, help="Show all current defaults")
def all(show):
    """Show all current defaults or reset everything."""
    config_manager = get_config_manager()

    if show:
        all_defaults = config_manager.get_all_defaults()
        if all_defaults:
            click.echo("All current defaults:")
            for command, defaults in all_defaults.items():
                click.echo(f"\n[{command}]")
                for key, value in defaults.items():
                    click.echo(f"  --{key.replace('_', '-')}: {value}")
        else:
            click.echo("No defaults are currently set.")

        # Show config file location
        config_file = get_config_file()
        click.echo(f"\nConfiguration file: {config_file}")
    else:
        # Reset all defaults
        config_manager.reset_all_defaults()
        click.echo("All defaults have been reset.")
