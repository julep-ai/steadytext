"""Configuration management for persistent CLI defaults.

AIDEV-NOTE: This module handles reading and writing CLI defaults to a TOML configuration file
stored in the user's config directory. It provides a centralized way to manage default
parameters for all steadytext CLI commands.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import toml

from .utils import get_cache_dir


def get_config_dir() -> Path:
    """Get the configuration directory for steadytext.
    
    Returns:
        Path to the config directory (~/.config/steadytext on Linux/Mac,
        %LOCALAPPDATA%/steadytext/config on Windows)
    """
    # AIDEV-NOTE: Using similar pattern to cache directory but under config subdirectory
    if os.name == "nt":  # Windows
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        config_dir = Path(base) / "steadytext" / "config"
    else:  # Linux, macOS, etc.
        xdg_config = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
        config_dir = Path(xdg_config) / "steadytext"
    
    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_defaults_path() -> Path:
    """Get the path to the defaults configuration file."""
    return get_config_dir() / "defaults.toml"


class DefaultsManager:
    """Manager for CLI command defaults.
    
    AIDEV-NOTE: This class handles reading, writing, and merging CLI defaults
    from the persistent configuration file. It uses TOML format for human-readable
    configuration storage.
    """
    
    def __init__(self):
        self.config_path = get_defaults_path()
        self._defaults = self._load_defaults()
    
    def _load_defaults(self) -> Dict[str, Dict[str, Any]]:
        """Load defaults from the configuration file."""
        if not self.config_path.exists():
            return {}
        
        try:
            with open(self.config_path, "r") as f:
                return toml.load(f)
        except Exception:
            # If config file is corrupted, start fresh
            return {}
    
    def _save_defaults(self) -> None:
        """Save defaults to the configuration file."""
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, "w") as f:
            toml.dump(self._defaults, f)
    
    def set_defaults(self, command: str, **kwargs) -> None:
        """Set defaults for a specific command.
        
        Args:
            command: The command name (e.g., 'generate', 'embed', 'rerank')
            **kwargs: The default parameters to set
        """
        # Remove None values and convert special types
        processed_kwargs = {}
        for key, value in kwargs.items():
            if value is not None:
                # Handle special conversions
                if key == "choices" and isinstance(value, list):
                    # Convert list back to comma-separated string for storage
                    value = ",".join(value)
                elif key == "schema" and isinstance(value, dict):
                    # Convert dict to JSON string for storage
                    value = json.dumps(value)
                processed_kwargs[key] = value
        
        if processed_kwargs:
            # Update defaults for this command
            self._defaults[command] = processed_kwargs
        else:
            # If no args provided, reset to empty (original defaults)
            self._defaults.pop(command, None)
        
        self._save_defaults()
    
    def get_defaults(self, command: str) -> Dict[str, Any]:
        """Get defaults for a specific command.
        
        Args:
            command: The command name
            
        Returns:
            Dictionary of default parameters for the command
        """
        return self._defaults.get(command, {})
    
    def get_all_defaults(self) -> Dict[str, Dict[str, Any]]:
        """Get all saved defaults.
        
        Returns:
            Dictionary mapping command names to their defaults
        """
        return self._defaults.copy()
    
    def reset_defaults(self, command: Optional[str] = None) -> None:
        """Reset defaults for a command or all commands.
        
        Args:
            command: If specified, reset only this command's defaults.
                    If None, reset all defaults.
        """
        if command:
            self._defaults.pop(command, None)
        else:
            self._defaults = {}
        
        self._save_defaults()
    
    def merge_with_cli_args(self, command: str, cli_args: Dict[str, Any]) -> Dict[str, Any]:
        """Merge saved defaults with CLI arguments.
        
        AIDEV-NOTE: Precedence order (highest to lowest):
        1. CLI arguments (if explicitly provided)
        2. Environment variables (handled by caller)
        3. Saved defaults
        4. Original command defaults (handled by Click)
        
        Args:
            command: The command name
            cli_args: Arguments provided via CLI
            
        Returns:
            Merged arguments with proper precedence
        """
        # Get saved defaults for this command
        defaults = self.get_defaults(command)
        
        # Start with saved defaults
        merged = defaults.copy()
        
        # Override with CLI args (only if explicitly provided)
        for key, value in cli_args.items():
            # AIDEV-NOTE: We need to distinguish between explicitly provided args
            # and default values. This is handled by the caller checking if the
            # value differs from Click's default.
            if value is not None:
                merged[key] = value
        
        return merged


# Global instance
_defaults_manager = None


def get_defaults_manager() -> DefaultsManager:
    """Get the global defaults manager instance."""
    global _defaults_manager
    if _defaults_manager is None:
        _defaults_manager = DefaultsManager()
    return _defaults_manager