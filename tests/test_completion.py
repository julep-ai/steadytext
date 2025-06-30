"""Tests for shell completion functionality.

AIDEV-NOTE: Tests the completion command and shell script generation
"""

import os
import pytest
from click.testing import CliRunner
from steadytext.cli.main import cli
from steadytext.cli.commands.completion import get_shell, get_completion_script


def test_completion_command_help():
    """Test that completion command shows help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["completion", "--help"])
    assert result.exit_code == 0
    assert "Generate shell completion scripts" in result.output


def test_completion_command_default():
    """Test completion command with auto-detection."""
    runner = CliRunner()
    result = runner.invoke(cli, ["completion"])
    assert result.exit_code == 0
    # Should show installation instructions
    assert "To install" in result.output or "completion" in result.output.lower()


def test_completion_command_bash():
    """Test completion command for bash."""
    runner = CliRunner()
    result = runner.invoke(cli, ["completion", "--shell", "bash"])
    assert result.exit_code == 0
    assert "bash" in result.output.lower()
    assert "eval" in result.output or "source" in result.output


def test_completion_command_zsh():
    """Test completion command for zsh."""
    runner = CliRunner()
    result = runner.invoke(cli, ["completion", "--shell", "zsh"])
    assert result.exit_code == 0
    assert "zsh" in result.output.lower()
    assert "eval" in result.output or ".zshrc" in result.output


def test_completion_command_fish():
    """Test completion command for fish."""
    runner = CliRunner()
    result = runner.invoke(cli, ["completion", "--shell", "fish"])
    assert result.exit_code == 0
    assert "fish" in result.output.lower()
    assert "completions" in result.output


def test_get_shell_detection():
    """Test shell detection function."""
    # Save original SHELL env var
    original_shell = os.environ.get("SHELL")

    try:
        # Test with SHELL env var
        os.environ["SHELL"] = "/bin/bash"
        assert get_shell() == "bash"

        os.environ["SHELL"] = "/usr/bin/zsh"
        assert get_shell() == "zsh"

        os.environ["SHELL"] = "/usr/local/bin/fish"
        assert get_shell() == "fish"

        # Test with unknown shell (should default to bash)
        os.environ["SHELL"] = "/bin/unknown"
        assert get_shell() == "bash"

        # Test without SHELL env var
        if "SHELL" in os.environ:
            del os.environ["SHELL"]
        shell = get_shell()
        assert shell in ["bash", "zsh", "fish"]  # Should detect or default to bash

    finally:
        # Restore original SHELL env var
        if original_shell:
            os.environ["SHELL"] = original_shell
        elif "SHELL" in os.environ:
            del os.environ["SHELL"]


def test_get_completion_script():
    """Test completion script generation."""
    # Test bash script
    bash_script = get_completion_script("bash", "st")
    assert "_ST_COMPLETE=bash_source st" in bash_script

    # Test zsh script
    zsh_script = get_completion_script("zsh", "st")
    assert "_ST_COMPLETE=zsh_source st" in zsh_script

    # Test fish script
    fish_script = get_completion_script("fish", "steadytext")
    assert "_STEADYTEXT_COMPLETE=fish_source steadytext" in fish_script

    # Test unsupported shell
    with pytest.raises(ValueError, match="Unsupported shell"):
        get_completion_script("tcsh", "st")


def test_completion_install_dry_run():
    """Test completion install flag (dry run only)."""
    runner = CliRunner()

    # We don't actually want to modify the user's shell config in tests
    # So we just verify the command runs without error
    result = runner.invoke(cli, ["completion", "--shell", "bash"])
    assert result.exit_code == 0

    # The output should contain installation instructions
    assert "mkdir" in result.output or "eval" in result.output


# AIDEV-NOTE: We don't test actual installation to avoid modifying test environment
# AIDEV-TODO: Add integration tests that verify generated scripts work with actual shells
