"""Tests for the configuration management module.

AIDEV-NOTE: This module tests the DefaultsManager class and the config module
functionality for persisting CLI defaults.
"""

import json
import os
import tempfile
from pathlib import Path
import pytest

from steadytext.config import DefaultsManager, get_config_dir, get_defaults_path


class TestConfigPaths:
    """Test configuration path functions."""
    
    def test_get_config_dir_linux(self, monkeypatch):
        """Test config directory path on Linux/Mac."""
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg_config")
        
        config_dir = get_config_dir()
        assert config_dir == Path("/tmp/xdg_config/steadytext")
    
    def test_get_config_dir_windows(self, monkeypatch):
        """Test config directory path on Windows."""
        monkeypatch.setattr("os.name", "nt")
        monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\test\\AppData\\Local")
        
        config_dir = get_config_dir()
        assert config_dir == Path("C:\\Users\\test\\AppData\\Local\\steadytext\\config")
    
    def test_get_defaults_path(self):
        """Test defaults file path."""
        config_dir = get_config_dir()
        defaults_path = get_defaults_path()
        assert defaults_path == config_dir / "defaults.toml"


class TestDefaultsManager:
    """Test the DefaultsManager class."""
    
    @pytest.fixture
    def temp_config(self, tmp_path, monkeypatch):
        """Create a temporary config directory."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        
        # Monkeypatch the get_config_dir function
        monkeypatch.setattr("steadytext.config.get_config_dir", lambda: config_dir)
        
        return config_dir
    
    def test_init_creates_empty_defaults(self, temp_config):
        """Test that initialization creates empty defaults."""
        manager = DefaultsManager()
        assert manager.get_all_defaults() == {}
    
    def test_set_and_get_defaults(self, temp_config):
        """Test setting and getting defaults for a command."""
        manager = DefaultsManager()
        
        # Set defaults for generate command
        manager.set_defaults("generate", model="gemma-3n-2b", size="large", seed=123)
        
        # Get defaults
        defaults = manager.get_defaults("generate")
        assert defaults == {"model": "gemma-3n-2b", "size": "large", "seed": 123}
    
    def test_set_defaults_overwrites(self, temp_config):
        """Test that setting defaults overwrites previous values."""
        manager = DefaultsManager()
        
        # Set initial defaults
        manager.set_defaults("generate", model="gemma-3n-2b", size="small")
        
        # Overwrite with new defaults
        manager.set_defaults("generate", model="gemma-3n-4b", seed=42)
        
        # Should have new values only
        defaults = manager.get_defaults("generate")
        assert defaults == {"model": "gemma-3n-4b", "seed": 42}
        assert "size" not in defaults
    
    def test_set_defaults_with_none_values(self, temp_config):
        """Test that None values are not saved."""
        manager = DefaultsManager()
        
        manager.set_defaults("generate", model="gemma-3n-2b", size=None, seed=42)
        
        defaults = manager.get_defaults("generate")
        assert defaults == {"model": "gemma-3n-2b", "seed": 42}
        assert "size" not in defaults
    
    def test_set_defaults_with_special_types(self, temp_config):
        """Test handling of special types (lists, dicts)."""
        manager = DefaultsManager()
        
        # Test with choices list
        manager.set_defaults("generate", choices=["yes", "no", "maybe"])
        defaults = manager.get_defaults("generate")
        assert defaults["choices"] == "yes,no,maybe"
        
        # Test with schema dict
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        manager.set_defaults("generate", schema=schema)
        defaults = manager.get_defaults("generate")
        assert defaults["schema"] == json.dumps(schema)
    
    def test_reset_defaults_single_command(self, temp_config):
        """Test resetting defaults for a single command."""
        manager = DefaultsManager()
        
        # Set defaults for multiple commands
        manager.set_defaults("generate", model="gemma-3n-2b")
        manager.set_defaults("embed", seed=123)
        
        # Reset only generate
        manager.reset_defaults("generate")
        
        # Generate should be empty, embed should remain
        assert manager.get_defaults("generate") == {}
        assert manager.get_defaults("embed") == {"seed": 123}
    
    def test_reset_defaults_all(self, temp_config):
        """Test resetting all defaults."""
        manager = DefaultsManager()
        
        # Set defaults for multiple commands
        manager.set_defaults("generate", model="gemma-3n-2b")
        manager.set_defaults("embed", seed=123)
        manager.set_defaults("rerank", task="custom task")
        
        # Reset all
        manager.reset_defaults()
        
        # All should be empty
        assert manager.get_all_defaults() == {}
    
    def test_empty_args_resets_command(self, temp_config):
        """Test that setting defaults with no args resets the command."""
        manager = DefaultsManager()
        
        # Set initial defaults
        manager.set_defaults("generate", model="gemma-3n-2b", size="large")
        
        # Call with no args (empty dict after filtering None values)
        manager.set_defaults("generate")
        
        # Should be reset
        assert manager.get_defaults("generate") == {}
    
    def test_persistence_across_instances(self, temp_config):
        """Test that defaults persist across manager instances."""
        # First manager sets defaults
        manager1 = DefaultsManager()
        manager1.set_defaults("generate", model="gemma-3n-2b", seed=123)
        
        # Second manager should see the same defaults
        manager2 = DefaultsManager()
        defaults = manager2.get_defaults("generate")
        assert defaults == {"model": "gemma-3n-2b", "seed": 123}
    
    def test_corrupted_config_file(self, temp_config):
        """Test handling of corrupted config file."""
        # Write invalid TOML
        config_path = temp_config / "defaults.toml"
        with open(config_path, "w") as f:
            f.write("invalid toml content [[[")
        
        # Should start with empty defaults
        manager = DefaultsManager()
        assert manager.get_all_defaults() == {}
    
    def test_merge_with_cli_args(self, temp_config):
        """Test merging saved defaults with CLI arguments."""
        manager = DefaultsManager()
        
        # Set some defaults
        manager.set_defaults("generate", model="gemma-3n-2b", size="large", seed=123)
        
        # CLI args with some overrides
        cli_args = {
            "model": "gemma-3n-4b",  # Override
            "size": "large",         # Same as default
            "seed": None,            # Not provided (use default)
            "max_new_tokens": 256    # New parameter
        }
        
        # Merge
        merged = manager.merge_with_cli_args("generate", cli_args)
        
        # Should have CLI override, default seed, and new parameter
        assert merged == {
            "model": "gemma-3n-4b",
            "size": "large",
            "seed": 123,
            "max_new_tokens": 256
        }
    
    def test_multiple_commands(self, temp_config):
        """Test managing defaults for multiple commands."""
        manager = DefaultsManager()
        
        # Set defaults for different commands
        manager.set_defaults("generate", model="gemma-3n-2b", seed=42)
        manager.set_defaults("embed", seed=123, json=True)
        manager.set_defaults("rerank", task="Find relevant docs", top_k=5)
        
        # Verify each command has its own defaults
        assert manager.get_defaults("generate") == {"model": "gemma-3n-2b", "seed": 42}
        assert manager.get_defaults("embed") == {"seed": 123, "json": True}
        assert manager.get_defaults("rerank") == {"task": "Find relevant docs", "top_k": 5}
        
        # Verify all defaults
        all_defaults = manager.get_all_defaults()
        assert len(all_defaults) == 3
        assert "generate" in all_defaults
        assert "embed" in all_defaults
        assert "rerank" in all_defaults