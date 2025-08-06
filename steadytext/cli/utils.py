"""Utilities for CLI commands.

AIDEV-NOTE: This module provides helper functions for integrating persisted defaults
with CLI commands, respecting the proper precedence order.
"""

from typing import Dict, Any, Optional
import os
import click

from ..config import get_defaults_manager


def apply_defaults(command_name: str, ctx: click.Context, **cli_args) -> Dict[str, Any]:
    """Apply saved defaults to CLI arguments with proper precedence.
    
    AIDEV-NOTE: Precedence order (highest to lowest):
    1. Explicitly provided CLI arguments
    2. Environment variables
    3. Saved defaults (from set-default command)
    4. Built-in Click defaults
    
    Args:
        command_name: Name of the command (e.g., 'generate', 'embed')
        ctx: Click context containing parameter information
        **cli_args: Arguments provided via CLI
        
    Returns:
        Dictionary of merged arguments with proper precedence
    """
    manager = get_defaults_manager()
    saved_defaults = manager.get_defaults(command_name)
    
    # Start with an empty dict for the final arguments
    final_args = {}
    
    # First, apply saved defaults
    for key, value in saved_defaults.items():
        final_args[key] = value
    
    # Then, check environment variables and CLI args
    for key, value in cli_args.items():
        # Check if this parameter has an environment variable
        param = None
        for p in ctx.command.params:
            if p.name == key:
                param = p
                break
        
        # AIDEV-NOTE: Environment variables override saved defaults
        # This is handled by Click automatically, so we just need to
        # check if the value differs from the Click default
        if param and hasattr(param, 'default'):
            # If the value is different from Click's default, it was either
            # provided via CLI or environment variable, so use it
            if value != param.default:
                final_args[key] = value
            elif key not in final_args:
                # If not in saved defaults and is the Click default, include it
                final_args[key] = value
        else:
            # No default defined, so any value is explicit
            if value is not None:
                final_args[key] = value
    
    return final_args


def apply_saved_defaults_to_params(command_name: str, ctx: click.Context, **current_values) -> Dict[str, Any]:
    """Apply saved defaults to parameters based on precedence rules.
    
    AIDEV-NOTE: This helper function extracts the repetitive logic for applying
    saved defaults to CLI parameters. It only applies saved defaults when the
    current value matches the Click default (meaning it wasn't explicitly provided).
    
    Args:
        command_name: Name of the command (e.g., 'generate', 'embed')
        ctx: Click context containing parameter information
        **current_values: Current parameter values from the command
        
    Returns:
        Dictionary of updated parameter values with saved defaults applied
    """
    manager = get_defaults_manager()
    saved_defaults = manager.get_defaults(command_name)
    
    if not saved_defaults:
        return current_values
    
    # Build a mapping of parameter names to their Click defaults
    param_defaults = {}
    if ctx and hasattr(ctx, 'command') and hasattr(ctx.command, 'params'):
        for param in ctx.command.params:
            if hasattr(param, 'default'):
                param_defaults[param.name] = param.default
    
    # Apply saved defaults where current values match Click defaults
    result = current_values.copy()
    for key, saved_value in saved_defaults.items():
        if key in current_values:
            current_value = current_values[key]
            # Special handling for output_format (which might be set via --json flag)
            if key == "output_format" and saved_value == "json" and current_value == "raw":
                result[key] = "json"
            # Apply saved default if current value is the Click default
            elif current_value == param_defaults.get(key):
                result[key] = saved_value
    
    return result


def check_defaults_precedence(command_name: str, param_name: str, 
                            cli_value: Any, click_default: Any, 
                            ctx: Optional[click.Context] = None) -> tuple[Any, str]:
    """Determine the effective value and source for a parameter.
    
    AIDEV-NOTE: This helper function determines which value should be used
    based on the precedence rules and returns both the value and its source.
    Now checks the actual Click parameter's envvar attribute if available.
    
    Args:
        command_name: Name of the command
        param_name: Name of the parameter
        cli_value: Value provided via CLI (or Click default)
        click_default: The Click-defined default value
        ctx: Optional Click context to get parameter's envvar attribute
        
    Returns:
        Tuple of (effective_value, source) where source is one of:
        'cli', 'env', 'saved', 'default'
    """
    manager = get_defaults_manager()
    saved_defaults = manager.get_defaults(command_name)
    
    # Check if value was explicitly provided via CLI
    if cli_value != click_default:
        # Could be from CLI or environment variable
        # First try to get the actual envvar from the parameter definition
        env_vars = []
        if ctx and hasattr(ctx, 'command') and hasattr(ctx.command, 'params'):
            for param in ctx.command.params:
                if param.name == param_name and hasattr(param, 'envvar'):
                    if param.envvar:
                        # envvar can be a string or a list of strings
                        if isinstance(param.envvar, (list, tuple)):
                            env_vars.extend(param.envvar)
                        else:
                            env_vars.append(param.envvar)
        
        # If no envvar defined in parameter, use default pattern
        if not env_vars:
            env_vars = [f"STEADYTEXT_{param_name.upper()}"]
        
        # Check if any of the environment variables are set
        for env_var in env_vars:
            if env_var in os.environ:
                return cli_value, 'env'
        
        # Not from env, must be from CLI
        return cli_value, 'cli'
    
    # Check if there's a saved default
    if param_name in saved_defaults:
        return saved_defaults[param_name], 'saved'
    
    # Use Click default
    return click_default, 'default'