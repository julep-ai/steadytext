"""
Configuration management for SteadyText CLI defaults.

This module handles persistent storage and retrieval of default parameters
for CLI commands, implementing the precedence order:
1. User-specified command-line arguments (highest)
2. Environment variables
3. set-default values (persisted)
4. Original command defaults (lowest)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable, cast
import functools

# Handle TOML imports for different Python versions
if sys.version_info >= (3, 11):
    import tomllib
    import tomli_w as tomli_write
else:
    import tomli as tomllib
    import tomli_w as tomli_write

from .utils import logger


# Configuration file location following XDG Base Directory Specification
def get_config_dir() -> Path:
    """Get the configuration directory for SteadyText."""
    if os.name == "nt":  # Windows
        config_dir = (
            Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
            / "steadytext"
        )
    else:  # Unix-like systems
        config_dir = (
            Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            / "steadytext"
        )

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_file() -> Path:
    """Get the path to the defaults configuration file."""
    return get_config_dir() / "defaults.toml"


class ConfigManager:
    """Manages configuration defaults for CLI commands."""

    def __init__(self):
        self.config_file = get_config_file()
        self._config_cache: Optional[Dict[str, Any]] = None

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file, with caching."""
        if self._config_cache is not None:
            return self._config_cache

        if not self.config_file.exists():
            self._config_cache = {}
            return self._config_cache

        try:
            with open(self.config_file, "rb") as f:
                self._config_cache = tomllib.load(f)
            logger.debug(f"Loaded configuration from {self.config_file}")
        except Exception as e:
            logger.warning(f"Failed to load configuration from {self.config_file}: {e}")
            self._config_cache = {}

        return self._config_cache

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file atomically."""
        try:
            # Ensure config directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first for atomic operation
            temp_file = self.config_file.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                tomli_write.dump(config, f)

            # Atomic move
            temp_file.replace(self.config_file)

            # Update cache
            self._config_cache = config
            logger.debug(f"Saved configuration to {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to save configuration to {self.config_file}: {e}")
            # Clean up temp file if it exists
            temp_file = self.config_file.with_suffix(".tmp")
            if temp_file.exists():
                temp_file.unlink()
            raise

    def get_command_defaults(self, command: str) -> Dict[str, Any]:
        """Get stored defaults for a specific command."""
        config = self._load_config()
        return config.get(command, {})

    def set_command_defaults(self, command: str, defaults: Dict[str, Any]) -> None:
        """Set defaults for a specific command."""
        config = self._load_config()

        if defaults:
            # Filter out None values and empty strings
            filtered_defaults = {
                k: v for k, v in defaults.items() if v is not None and v != ""
            }
            if filtered_defaults:
                config[command] = filtered_defaults
            elif command in config:
                # If no valid defaults and command exists, remove it
                del config[command]
        else:
            # Empty defaults means reset - remove the command section
            if command in config:
                del config[command]

        self._save_config(config)

    def reset_command_defaults(self, command: str) -> None:
        """Reset defaults for a specific command."""
        self.set_command_defaults(command, {})

    def reset_all_defaults(self) -> None:
        """Reset all stored defaults."""
        self._save_config({})

    def get_all_defaults(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored defaults."""
        return self._load_config()

    def merge_defaults(
        self,
        command: str,
        cli_params: Dict[str, Any],
        env_vars: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Merge defaults with CLI parameters and environment variables.

        Precedence order (highest to lowest):
        1. CLI parameters (non-None values)
        2. Environment variables
        3. Stored defaults
        4. Built-in defaults (handled by Click)

        Args:
            command: Command name
            cli_params: Parameters from CLI (may contain None values)
            env_vars: Environment variables (optional)

        Returns:
            Merged parameters dictionary
        """
        # Start with stored defaults
        merged = self.get_command_defaults(command).copy()

        # Apply environment variables if provided
        if env_vars:
            for key, value in env_vars.items():
                if value is not None:
                    merged[key] = value

        # Apply CLI parameters (only non-None values)
        for key, value in cli_params.items():
            if value is not None:
                merged[key] = value

        return merged


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return cast(ConfigManager, _config_manager)


# Convenience functions
def get_command_defaults(command: str) -> Dict[str, Any]:
    """Get stored defaults for a command."""
    return get_config_manager().get_command_defaults(command)


def set_command_defaults(command: str, defaults: Dict[str, Any]) -> None:
    """Set defaults for a command."""
    get_config_manager().set_command_defaults(command, defaults)


def reset_command_defaults(command: str) -> None:
    """Reset defaults for a command."""
    get_config_manager().reset_command_defaults(command)


def merge_defaults(
    command: str, cli_params: Dict[str, Any], env_vars: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Merge defaults with CLI parameters and environment variables."""
    return get_config_manager().merge_defaults(command, cli_params, env_vars)


def with_defaults(command_name: str):
    """
    Decorator to automatically apply stored defaults to a Click command.

    This decorator modifies the command function to load stored defaults
    and merge them with CLI arguments before the function is called.

    Args:
        command_name: Name of the command for configuration lookup

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get stored defaults for this command
            config_manager = get_config_manager()
            stored_defaults = config_manager.get_command_defaults(command_name)

            # Apply stored defaults for parameters that weren't explicitly set
            for key, default_value in stored_defaults.items():
                if key in kwargs and kwargs[key] is None:
                    kwargs[key] = default_value

            return func(*args, **kwargs)

        return wrapper

    return decorator
