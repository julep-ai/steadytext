"""CLI commands for unsafe mode remote models.

AIDEV-NOTE: These commands help users discover and use remote models
with best-effort determinism support.
"""

import click
import sys
import os


@click.group()
def unsafe():
    """Manage unsafe mode for remote models with best-effort determinism."""
    pass


@unsafe.command()
def list_models():
    """List available remote models that support seed parameters.
    
    WARNING: These models provide only best-effort determinism!
    """
    # Check if unsafe mode is enabled
    if os.environ.get("STEADYTEXT_UNSAFE_MODE", "false").lower() != "true":
        click.echo(
            "Note: Unsafe mode is not enabled. Set STEADYTEXT_UNSAFE_MODE=true to use these models.\n",
            err=True
        )
    
    try:
        from ...providers.registry import list_remote_models
        
        models = list_remote_models()
        
        if not models:
            click.echo("No remote models available.")
            return
        
        click.echo("Available remote models (provider:model format):\n")
        
        for provider, model_list in sorted(models.items()):
            click.echo(f"{provider}:")
            if model_list:
                for model in sorted(model_list):
                    click.echo(f"  - {provider}:{model}")
            else:
                click.echo("  (Unable to retrieve model list)")
            click.echo()
        
        click.echo("WARNING: Remote models provide only BEST-EFFORT determinism!")
        click.echo("Results may vary between calls, environments, and over time.")
        click.echo("\nFor TRUE determinism, use local GGUF models (default behavior).")
        
    except Exception as e:
        click.echo(f"Error listing models: {e}", err=True)
        sys.exit(1)


@unsafe.command()
def status():
    """Check unsafe mode status and configuration."""
    unsafe_mode = os.environ.get("STEADYTEXT_UNSAFE_MODE", "false").lower() in ["true", "1", "yes"]
    
    click.echo(f"Unsafe mode enabled: {'Yes' if unsafe_mode else 'No'}")
    
    if unsafe_mode:
        click.echo("\nWARNING: Unsafe mode is ENABLED!")
        click.echo("Remote models can be used but provide only best-effort determinism.")
    else:
        click.echo("\nTo enable unsafe mode: export STEADYTEXT_UNSAFE_MODE=true")
    
    # Check for API keys
    click.echo("\nAPI key status:")
    api_keys = {
        "OpenAI": "OPENAI_API_KEY",
        "Cerebras": "CEREBRAS_API_KEY",
    }
    
    for provider, env_var in api_keys.items():
        has_key = bool(os.environ.get(env_var))
        status = "Set" if has_key else "Not set"
        click.echo(f"  {provider}: {status} ({env_var})")


@unsafe.command()
def enable():
    """Show how to enable unsafe mode."""
    click.echo("To enable unsafe mode for remote models:")
    click.echo("\nBash/Zsh:")
    click.echo("  export STEADYTEXT_UNSAFE_MODE=true")
    click.echo("\nFish:")
    click.echo("  set -x STEADYTEXT_UNSAFE_MODE true")
    click.echo("\nWindows CMD:")
    click.echo("  set STEADYTEXT_UNSAFE_MODE=true")
    click.echo("\nWindows PowerShell:")
    click.echo("  $env:STEADYTEXT_UNSAFE_MODE=\"true\"")
    click.echo("\nWARNING: Remote models provide only best-effort determinism!")