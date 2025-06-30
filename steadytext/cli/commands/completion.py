"""Shell completion support for SteadyText CLI.

AIDEV-NOTE: This module provides shell completion functionality for both bash and zsh.
It uses Click's built-in completion support to generate completion scripts.
AIDEV-NOTE: The completion scripts work for both 'st' and 'steadytext' command names.
"""

import click
import os
import sys
import subprocess
from pathlib import Path


def get_shell():
    """Detect the current shell.

    AIDEV-NOTE: Detection order: SHELL env var, parent process name, fallback to bash
    """
    # Try SHELL environment variable first
    shell = os.environ.get("SHELL", "")
    if shell:
        shell_name = Path(shell).name
        if shell_name in ["bash", "zsh", "fish"]:
            return shell_name

    # Try to detect from parent process (more reliable for subshells)
    try:
        if sys.platform != "win32":
            # Get parent process info
            ppid = os.getppid()
            result = subprocess.run(
                ["ps", "-p", str(ppid), "-o", "comm="], capture_output=True, text=True
            )
            if result.returncode == 0:
                parent_process = result.stdout.strip()
                if parent_process in ["bash", "zsh", "fish"]:
                    return parent_process
    except Exception:
        pass

    # Default to bash
    return "bash"


def get_completion_script(shell, prog_name):
    """Generate completion script for the specified shell.

    AIDEV-NOTE: Uses Click's shell completion API which handles all subcommands
    and options automatically.
    """
    if shell == "bash":
        return f"""# bash completion for {prog_name}
_{prog_name.upper()}_COMPLETE=bash_source {prog_name}"""

    elif shell == "zsh":
        return f"""# zsh completion for {prog_name}
_{prog_name.upper()}_COMPLETE=zsh_source {prog_name}"""

    elif shell == "fish":
        return f"""# fish completion for {prog_name}
_{prog_name.upper()}_COMPLETE=fish_source {prog_name}"""

    else:
        raise ValueError(f"Unsupported shell: {shell}")


def get_install_instructions(shell, prog_name):
    """Get installation instructions for the completion script."""
    script = get_completion_script(shell, prog_name)

    if shell == "bash":
        completion_dir = "~/.local/share/bash-completion/completions"
        instructions = f'''To install bash completions for {prog_name}:

1. Create the completion directory if it doesn't exist:
   mkdir -p {completion_dir}

2. Generate and save the completion script:
   eval "$({script})" > {completion_dir}/{prog_name}

3. Source your ~/.bashrc or start a new shell:
   source ~/.bashrc

Alternative (for current session only):
   eval "$({script})"'''

    elif shell == "zsh":
        instructions = f'''To install zsh completions for {prog_name}:

1. Add to your ~/.zshrc:
   eval "$({script})"

2. Reload your shell configuration:
   source ~/.zshrc

Alternative (for current session only):
   eval "$({script})"'''

    elif shell == "fish":
        completion_dir = "~/.config/fish/completions"
        instructions = f"""To install fish completions for {prog_name}:

1. Create the completion directory if it doesn't exist:
   mkdir -p {completion_dir}

2. Generate and save the completion script:
   {script} > {completion_dir}/{prog_name}.fish

3. Reload your shell or start a new one"""

    else:
        instructions = f"Completions not available for shell: {shell}"

    return instructions


@click.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    help="Shell to generate completions for (auto-detected if not specified)",
)
@click.option(
    "--install", is_flag=True, help="Attempt to install completions automatically"
)
def completion(shell, install):
    """Generate shell completion scripts for steadytext/st commands.

    AIDEV-NOTE: Supports bash, zsh, and fish shells with automatic detection.
    Installation can be manual (default) or automatic (--install flag).
    """
    if not shell:
        shell = get_shell()
        click.echo(f"Detected shell: {shell}")

    # Get the program name (could be 'st' or 'steadytext')
    prog_name = Path(sys.argv[0]).name

    if install:
        # Attempt automatic installation
        try:
            if shell == "bash":
                completion_dir = (
                    Path.home() / ".local/share/bash-completion/completions"
                )
                completion_dir.mkdir(parents=True, exist_ok=True)

                # Generate completion script
                script = get_completion_script(shell, prog_name)
                result = subprocess.run(
                    f'eval "$({script})"',
                    shell=True,
                    capture_output=True,
                    text=True,
                    executable="/bin/bash",
                )

                if result.returncode == 0:
                    completion_file = completion_dir / prog_name
                    completion_file.write_text(result.stdout)
                    click.echo(f"✓ Installed bash completions to {completion_file}")
                    click.echo("  Restart your shell or run: source ~/.bashrc")
                else:
                    raise Exception(f"Failed to generate completions: {result.stderr}")

            elif shell == "zsh":
                # For zsh, we need to add to .zshrc
                zshrc = Path.home() / ".zshrc"
                script = get_completion_script(shell, prog_name)
                completion_line = f'\n# SteadyText completion\neval "$({script})"\n'

                # Check if already installed
                if zshrc.exists():
                    content = zshrc.read_text()
                    if script in content:
                        click.echo("✓ Completions already installed in ~/.zshrc")
                        return

                # Append to .zshrc
                with open(zshrc, "a") as f:
                    f.write(completion_line)

                click.echo("✓ Added completion to ~/.zshrc")
                click.echo("  Restart your shell or run: source ~/.zshrc")

            elif shell == "fish":
                completion_dir = Path.home() / ".config/fish/completions"
                completion_dir.mkdir(parents=True, exist_ok=True)

                # Generate completion script
                script = get_completion_script(shell, prog_name)
                result = subprocess.run(
                    script, shell=True, capture_output=True, text=True
                )

                if result.returncode == 0:
                    completion_file = completion_dir / f"{prog_name}.fish"
                    completion_file.write_text(result.stdout)
                    click.echo(f"✓ Installed fish completions to {completion_file}")
                    click.echo("  Restart your shell or reload completions")
                else:
                    raise Exception(f"Failed to generate completions: {result.stderr}")

            # Also install for the alternate command name
            alt_prog = "steadytext" if prog_name == "st" else "st"
            click.echo(
                f"\nNote: You may also want to install completions for '{alt_prog}'"
            )

        except Exception as e:
            click.echo(f"✗ Auto-installation failed: {e}", err=True)
            click.echo("\nManual installation instructions:")
            click.echo(get_install_instructions(shell, prog_name))
    else:
        # Show manual installation instructions
        click.echo(get_install_instructions(shell, prog_name))

        # Also show for alternate command
        alt_prog = "steadytext" if prog_name == "st" else "st"
        click.echo(f"\n{'=' * 60}")
        click.echo(f"\nFor the '{alt_prog}' command:")
        click.echo(get_install_instructions(shell, alt_prog))
