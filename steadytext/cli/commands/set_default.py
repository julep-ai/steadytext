"""Command for setting persistent defaults for CLI commands.

AIDEV-NOTE: This command allows users to save default parameters for each CLI command
(generate, embed, rerank) which are then automatically applied unless overridden.
"""

import click
import json
from typing import Dict, Any

from ...config import get_defaults_manager


# AIDEV-NOTE: Define the commands and their allowed parameters
SUPPORTED_COMMANDS = {
    "generate": {
        "model", "model_repo", "model_filename", "size", "seed", "max_new_tokens",
        "eos_string", "wait", "json", "raw", "logprobs", "no_index", "index_file",
        "top_k", "quiet", "verbose", "schema", "regex", "choices", "unsafe_mode"
    },
    "embed": {
        "model", "model_repo", "model_filename", "size", "seed", "quiet", "verbose",
        "json", "numpy", "hex", "unsafe_mode"
    },
    "rerank": {
        "model", "model_repo", "model_filename", "size", "task", "batch_size",
        "quiet", "verbose", "json", "score_format", "unsafe_mode"
    }
}


def parse_cli_args(args: tuple) -> Dict[str, Any]:
    """Parse CLI-style arguments into a dictionary.
    
    AIDEV-NOTE: This function parses arguments in the same format as the original
    commands accept them, converting --flag-name value pairs into a dictionary.
    """
    parsed = {}
    i = 0
    
    while i < len(args):
        arg = args[i]
        
        if not arg.startswith("--"):
            raise click.ClickException(f"Invalid argument: {arg}. Arguments must start with --")
        
        # Remove -- prefix and convert to underscore format
        key = arg[2:].replace("-", "_")
        
        # Check if this is a flag (no value)
        if i + 1 >= len(args) or args[i + 1].startswith("--"):
            # It's a flag
            parsed[key] = True
            i += 1
        else:
            # It has a value
            value = args[i + 1]
            
            # Try to parse the value appropriately
            if value.lower() in ("true", "false"):
                parsed[key] = value.lower() == "true"
            elif value.isdigit():
                parsed[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                parsed[key] = float(value)
            else:
                parsed[key] = value
            
            i += 2
    
    return parsed


@click.command()
@click.argument("command", type=click.Choice(list(SUPPORTED_COMMANDS.keys())))
@click.argument("args", nargs=-1)
@click.option("--show", is_flag=True, help="Show current defaults for the command")
@click.option("--reset-all", is_flag=True, help="Reset all saved defaults")
def set_default(command: str, args: tuple, show: bool, reset_all: bool):
    """Set persistent defaults for CLI commands.
    
    This command allows you to save default parameters that will be automatically
    applied to commands unless explicitly overridden.
    
    Precedence order (highest to lowest):
    1. Command-line arguments
    2. Environment variables
    3. Saved defaults (set by this command)
    4. Built-in defaults
    
    Examples:
        # Set default model and size for generation
        st set-default generate --model gemma-3n-2b --size large
        
        # Set default output format for embed
        st set-default embed --json
        
        # Show current defaults for a command
        st set-default generate --show
        
        # Reset defaults for a command (no args)
        st set-default generate
        
        # Reset all saved defaults
        st set-default generate --reset-all
    
    AIDEV-NOTE: The saved defaults are stored in ~/.config/steadytext/defaults.toml
    on Linux/Mac or %LOCALAPPDATA%/steadytext/config/defaults.toml on Windows.
    """
    manager = get_defaults_manager()
    
    # Handle reset-all flag
    if reset_all:
        manager.reset_defaults()
        click.echo("All saved defaults have been reset.")
        return
    
    # Handle show flag
    if show:
        defaults = manager.get_defaults(command)
        if defaults:
            click.echo(f"Current defaults for '{command}':")
            for key, value in sorted(defaults.items()):
                # Format the key back to CLI style
                cli_key = "--" + key.replace("_", "-")
                if isinstance(value, bool):
                    if value:
                        click.echo(f"  {cli_key}")
                else:
                    click.echo(f"  {cli_key} {value}")
        else:
            click.echo(f"No saved defaults for '{command}'.")
        
        # Also show all defaults if verbose
        all_defaults = manager.get_all_defaults()
        if all_defaults and len(all_defaults) > 1:
            click.echo("\nOther commands with saved defaults:")
            for cmd in sorted(all_defaults.keys()):
                if cmd != command:
                    click.echo(f"  {cmd}")
        return
    
    # Parse the provided arguments
    if args:
        try:
            parsed_args = parse_cli_args(args)
        except click.ClickException as e:
            click.echo(f"Error: {e}", err=True)
            click.echo("\nUsage: st set-default <command> [--option value ...]", err=True)
            return
        
        # Validate that all arguments are supported for this command
        supported = SUPPORTED_COMMANDS[command]
        unsupported = set(parsed_args.keys()) - supported
        if unsupported:
            click.echo(f"Error: Unsupported options for '{command}': {', '.join(sorted(unsupported))}", err=True)
            click.echo(f"\nSupported options: {', '.join(sorted(supported))}", err=True)
            return
        
        # Save the defaults
        manager.set_defaults(command, **parsed_args)
        click.echo(f"Defaults saved for '{command}'.")
        
        # Show what was saved
        click.echo("\nSaved defaults:")
        for key, value in sorted(parsed_args.items()):
            cli_key = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    click.echo(f"  {cli_key}")
            else:
                click.echo(f"  {cli_key} {value}")
    else:
        # No args means reset
        manager.reset_defaults(command)
        click.echo(f"Defaults reset for '{command}'.")
    
    # Show the config file location
    config_path = manager.config_path
    click.echo(f"\nDefaults stored in: {config_path}")