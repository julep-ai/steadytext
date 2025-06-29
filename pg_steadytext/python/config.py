"""
config.py - Configuration management for pg_steadytext

AIDEV-NOTE: This module provides configuration management utilities
for the PostgreSQL extension, reading from the steadytext_config table.
"""

import json
import logging
from typing import Any, Optional, Dict, cast

# Configure logging
logger = logging.getLogger(__name__)

# AIDEV-NOTE: Default configuration values
DEFAULTS = {
    "daemon_host": "localhost",
    "daemon_port": 5555,
    "cache_enabled": True,
    "max_cache_entries": 1000,
    "max_cache_size_mb": 500,
    "default_max_tokens": 512,
    "daemon_auto_start": True,
    "model_name": "qwen3-1.7b",
    "embedding_model": "qwen3-embedding",
    # thinking_mode removed - not supported by SteadyText
    "request_timeout": 30,  # seconds
    "batch_timeout": 120,  # seconds for batch operations
}

# PostgreSQL interaction
try:
    import plpy  # type: ignore

    IN_POSTGRES = True
except ImportError:
    IN_POSTGRES = False

    # Mock for testing
    class MockPlpy:
        def execute(self, query, args=None):
            return []

        def prepare(self, query, types=None):
            return lambda *args: []

    plpy = MockPlpy()


class ConfigManager:
    """
    Manages configuration for pg_steadytext.

    AIDEV-NOTE: This class provides a centralized way to access configuration
    values stored in the steadytext_config table, with fallback to defaults.
    """

    def __init__(self, table_name: str = "steadytext_config"):
        """
        Initialize the configuration manager.

        Args:
            table_name: Name of the configuration table
        """
        self.table_name = table_name
        self._cache = {}  # In-memory cache for config values

        if IN_POSTGRES:
            self._prepare_statements()

    def _prepare_statements(self):
        """Prepare PostgreSQL statements for configuration access."""
        self.get_plan = plpy.prepare(
            f"""
            SELECT value FROM {self.table_name} WHERE key = $1
        """,
            ["text"],
        )

        self.set_plan = plpy.prepare(
            f"""
            INSERT INTO {self.table_name} (key, value, description)
            VALUES ($1, $2, $3)
            ON CONFLICT (key) DO UPDATE
            SET value = $2,
                updated_at = NOW(),
                updated_by = current_user
        """,
            ["text", "jsonb", "text"],
        )

        self.get_all_plan = plpy.prepare(f"""
            SELECT key, value FROM {self.table_name}
        """)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        AIDEV-NOTE: Retrieves from PostgreSQL with fallback to defaults.
        Values are cached in memory for the session.

        Args:
            key: Configuration key
            default: Default value if not found (overrides DEFAULTS)

        Returns:
            Configuration value
        """
        # Check cache first
        if key in self._cache:
            return self._cache[key]

        # Use provided default or global default
        if default is None:
            default = DEFAULTS.get(key)

        if not IN_POSTGRES:
            return default

        try:
            result = plpy.execute(self.get_plan, [key])

            if result and len(result) > 0:
                # Parse JSON value
                value = json.loads(result[0]["value"])
                self._cache[key] = value
                return value

        except Exception as e:
            logger.warning(f"Error getting config '{key}': {e}")

        # Cache and return default
        self._cache[key] = default
        return default

    def set(self, key: str, value: Any, description: Optional[str] = None) -> bool:
        """
        Set a configuration value.

        AIDEV-NOTE: Stores in PostgreSQL and updates cache.

        Args:
            key: Configuration key
            value: Value to store (must be JSON-serializable)
            description: Optional description of the setting

        Returns:
            True if successful, False otherwise
        """
        if not IN_POSTGRES:
            self._cache[key] = value
            return True

        try:
            # Convert to JSON
            json_value = json.dumps(value)

            # Store in database
            plpy.execute(self.set_plan, [key, json_value, description or ""])

            # Update cache
            self._cache[key] = value

            return True

        except Exception as e:
            logger.error(f"Error setting config '{key}': {e}")
            return False

    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.

        AIDEV-NOTE: Returns merged configuration from database and defaults.

        Returns:
            Dictionary of all configuration values
        """
        config = DEFAULTS.copy()

        if IN_POSTGRES:
            try:
                result = plpy.execute(self.get_all_plan)

                for row in result:
                    try:
                        config[row["key"]] = json.loads(row["value"])
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON for config key '{row['key']}'")

            except Exception as e:
                logger.error(f"Error getting all config: {e}")

        # Override with cached values
        config.update(self._cache)

        return config

    def refresh_cache(self):
        """
        Refresh the in-memory configuration cache.

        AIDEV-NOTE: Call this if configuration might have been changed
        by another session.
        """
        self._cache.clear()

    # Convenience properties for common settings
    @property
    def daemon_host(self) -> str:
        """Get daemon host address."""
        return self.get("daemon_host")

    @property
    def daemon_port(self) -> int:
        """Get daemon port number."""
        return self.get("daemon_port")

    @property
    def daemon_endpoint(self) -> str:
        """Get full daemon endpoint URL."""
        return f"tcp://{self.daemon_host}:{self.daemon_port}"

    @property
    def cache_enabled(self) -> bool:
        """Check if caching is enabled."""
        return self.get("cache_enabled")

    @property
    def default_max_tokens(self) -> int:
        """Get default max tokens for generation."""
        return self.get("default_max_tokens")

    @property
    def model_name(self) -> str:
        """Get default model name."""
        return self.get("model_name")

    @property
    def daemon_auto_start(self) -> bool:
        """Check if daemon should auto-start."""
        return self.get("daemon_auto_start")


# AIDEV-NOTE: Global configuration instance
_config_instance: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """
    Get the global configuration manager instance.

    AIDEV-NOTE: Uses singleton pattern to avoid multiple database queries.

    Returns:
        ConfigManager instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    assert _config_instance is not None
    return cast(ConfigManager, _config_instance)


def get_setting(key: str, default: Any = None) -> Any:
    """
    Convenience function to get a configuration value.

    Args:
        key: Configuration key
        default: Default value if not found

    Returns:
        Configuration value
    """
    return get_config().get(key, default)


def set_setting(key: str, value: Any, description: Optional[str] = None) -> bool:
    """
    Convenience function to set a configuration value.

    Args:
        key: Configuration key
        value: Value to store
        description: Optional description

    Returns:
        True if successful
    """
    return get_config().set(key, value, description)


# AIDEV-NOTE: Export key items
__all__ = ["ConfigManager", "get_config", "get_setting", "set_setting", "DEFAULTS"]
