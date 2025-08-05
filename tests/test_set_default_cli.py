"""Tests for the set-default CLI command.

AIDEV-NOTE: This module tests the set-default command functionality including
argument parsing, defaults management, and error handling.
"""

import json
import pytest
from click.testing import CliRunner
from pathlib import Path

from steadytext.cli.main import cli
from steadytext.config import get_defaults_manager


class TestSetDefaultCommand:
    """Test the set-default CLI command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @pytest.fixture
    def isolated_config(self, runner, monkeypatch):
        """Create an isolated config environment for testing."""
        with runner.isolated_filesystem():
            # Create a temporary config directory
            config_dir = Path("test_config")
            config_dir.mkdir()
            
            # Monkeypatch the config directory
            monkeypatch.setattr("steadytext.config.get_config_dir", lambda: config_dir)
            
            # Clear any existing defaults manager instance
            import steadytext.config
            steadytext.config._defaults_manager = None
            
            yield config_dir
    
    def test_set_default_generate_basic(self, runner, isolated_config):
        """Test setting basic defaults for generate command."""
        result = runner.invoke(cli, ["set-default", "generate", "--model", "gemma-3n-2b", "--size", "large"])
        
        assert result.exit_code == 0
        assert "Defaults saved for 'generate'" in result.output
        assert "--model gemma-3n-2b" in result.output
        assert "--size large" in result.output
        
        # Verify defaults were saved
        manager = get_defaults_manager()
        defaults = manager.get_defaults("generate")
        assert defaults["model"] == "gemma-3n-2b"
        assert defaults["size"] == "large"
    
    def test_set_default_with_flags(self, runner, isolated_config):
        """Test setting flag-based defaults."""
        result = runner.invoke(cli, ["set-default", "generate", "--wait", "--logprobs", "--no-index"])
        
        assert result.exit_code == 0
        assert "--wait" in result.output
        assert "--logprobs" in result.output
        assert "--no-index" in result.output
        
        # Verify flags were saved as True
        manager = get_defaults_manager()
        defaults = manager.get_defaults("generate")
        assert defaults["wait"] is True
        assert defaults["logprobs"] is True
        assert defaults["no_index"] is True
    
    def test_set_default_embed(self, runner, isolated_config):
        """Test setting defaults for embed command."""
        result = runner.invoke(cli, ["set-default", "embed", "--seed", "123", "--json"])
        
        assert result.exit_code == 0
        assert "Defaults saved for 'embed'" in result.output
        assert "--seed 123" in result.output
        assert "--json" in result.output
        
        manager = get_defaults_manager()
        defaults = manager.get_defaults("embed")
        assert defaults["seed"] == 123
        assert defaults["json"] is True
    
    def test_set_default_rerank(self, runner, isolated_config):
        """Test setting defaults for rerank command."""
        result = runner.invoke(cli, ["set-default", "rerank", "--task", "Find medical info", "--top-k", "10"])
        
        assert result.exit_code == 0
        assert "Defaults saved for 'rerank'" in result.output
        assert "--task Find medical info" in result.output
        assert "--top-k 10" in result.output
    
    def test_show_defaults(self, runner, isolated_config):
        """Test showing current defaults."""
        # First set some defaults
        runner.invoke(cli, ["set-default", "generate", "--model", "gemma-3n-2b"])
        
        # Show defaults
        result = runner.invoke(cli, ["set-default", "generate", "--show"])
        
        assert result.exit_code == 0
        assert "Current defaults for 'generate':" in result.output
        assert "--model gemma-3n-2b" in result.output
    
    def test_show_no_defaults(self, runner, isolated_config):
        """Test showing when no defaults are set."""
        result = runner.invoke(cli, ["set-default", "generate", "--show"])
        
        assert result.exit_code == 0
        assert "No saved defaults for 'generate'" in result.output
    
    def test_reset_defaults(self, runner, isolated_config):
        """Test resetting defaults for a command."""
        # Set defaults
        runner.invoke(cli, ["set-default", "generate", "--model", "gemma-3n-2b"])
        
        # Reset (no args)
        result = runner.invoke(cli, ["set-default", "generate"])
        
        assert result.exit_code == 0
        assert "Defaults reset for 'generate'" in result.output
        
        # Verify defaults are gone
        manager = get_defaults_manager()
        assert manager.get_defaults("generate") == {}
    
    def test_reset_all_defaults(self, runner, isolated_config):
        """Test resetting all defaults."""
        # Set defaults for multiple commands
        runner.invoke(cli, ["set-default", "generate", "--model", "gemma-3n-2b"])
        runner.invoke(cli, ["set-default", "embed", "--seed", "123"])
        
        # Reset all
        result = runner.invoke(cli, ["set-default", "generate", "--reset-all"])
        
        assert result.exit_code == 0
        assert "All saved defaults have been reset" in result.output
        
        # Verify all defaults are gone
        manager = get_defaults_manager()
        assert manager.get_all_defaults() == {}
    
    def test_invalid_command(self, runner, isolated_config):
        """Test error handling for invalid command."""
        result = runner.invoke(cli, ["set-default", "invalid-command", "--model", "test"])
        
        assert result.exit_code != 0
        assert "Invalid value for 'COMMAND'" in result.output
    
    def test_invalid_option(self, runner, isolated_config):
        """Test error handling for unsupported option."""
        result = runner.invoke(cli, ["set-default", "generate", "--invalid-option", "value"])
        
        assert result.exit_code == 0  # Our command handles this gracefully
        assert "Error: Unsupported options for 'generate': invalid_option" in result.output
    
    def test_malformed_arguments(self, runner, isolated_config):
        """Test error handling for malformed arguments."""
        result = runner.invoke(cli, ["set-default", "generate", "not-an-option"])
        
        assert result.exit_code == 0  # Our command handles this gracefully
        assert "Error: Invalid argument: not-an-option" in result.output
    
    def test_numeric_value_parsing(self, runner, isolated_config):
        """Test parsing of numeric values."""
        result = runner.invoke(cli, ["set-default", "generate", "--seed", "42", "--max-new-tokens", "512"])
        
        assert result.exit_code == 0
        
        manager = get_defaults_manager()
        defaults = manager.get_defaults("generate")
        assert defaults["seed"] == 42
        assert defaults["max_new_tokens"] == 512
    
    def test_boolean_value_parsing(self, runner, isolated_config):
        """Test parsing of boolean values."""
        result = runner.invoke(cli, ["set-default", "generate", "--wait", "true", "--logprobs", "false"])
        
        assert result.exit_code == 0
        
        manager = get_defaults_manager()
        defaults = manager.get_defaults("generate")
        assert defaults["wait"] is True
        assert defaults["logprobs"] is False
    
    def test_complex_values(self, runner, isolated_config):
        """Test setting complex values like schemas."""
        schema = '{"type": "object", "properties": {"name": {"type": "string"}}}'
        result = runner.invoke(cli, ["set-default", "generate", "--schema", schema])
        
        assert result.exit_code == 0
        assert "--schema" in result.output
        
        manager = get_defaults_manager()
        defaults = manager.get_defaults("generate")
        assert defaults["schema"] == schema
    
    def test_choices_value(self, runner, isolated_config):
        """Test setting choices value."""
        result = runner.invoke(cli, ["set-default", "generate", "--choices", "yes,no,maybe"])
        
        assert result.exit_code == 0
        
        manager = get_defaults_manager()
        defaults = manager.get_defaults("generate")
        assert defaults["choices"] == "yes,no,maybe"
    
    def test_show_multiple_commands(self, runner, isolated_config):
        """Test showing defaults when multiple commands have defaults."""
        # Set defaults for multiple commands
        runner.invoke(cli, ["set-default", "generate", "--model", "gemma-3n-2b"])
        runner.invoke(cli, ["set-default", "embed", "--seed", "123"])
        runner.invoke(cli, ["set-default", "rerank", "--top-k", "5"])
        
        # Show generate defaults
        result = runner.invoke(cli, ["set-default", "generate", "--show"])
        
        assert result.exit_code == 0
        assert "Current defaults for 'generate':" in result.output
        assert "Other commands with saved defaults:" in result.output
        assert "embed" in result.output
        assert "rerank" in result.output
    
    def test_config_path_display(self, runner, isolated_config):
        """Test that config path is displayed."""
        result = runner.invoke(cli, ["set-default", "generate", "--model", "gemma-3n-2b"])
        
        assert result.exit_code == 0
        assert "Defaults stored in:" in result.output
        assert "defaults.toml" in result.output