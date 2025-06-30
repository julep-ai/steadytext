# AIDEV-NOTE: SQLite cache backend refactored to use the new CacheBackend interface
"""SQLite cache backend implementation."""

from pathlib import Path
from typing import Any, Dict, Optional

from ..sqlite_cache_backend import SQLiteDiskBackedFrecencyCache
from .base import CacheBackend


class SQLiteCacheBackend(CacheBackend):
    """SQLite-based cache backend with frecency eviction.

    This is a wrapper around the existing SQLiteDiskBackedFrecencyCache
    that implements the new CacheBackend interface.
    """

    def __init__(
        self,
        capacity: int = 128,
        cache_name: str = "sqlite_cache",
        max_size_mb: float = 100.0,
        cache_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize SQLite backend."""
        super().__init__(capacity, cache_name, max_size_mb, cache_dir, **kwargs)

        # AIDEV-NOTE: Delegate to the existing SQLite implementation
        self._backend = SQLiteDiskBackedFrecencyCache(
            capacity=capacity,
            cache_name=cache_name,
            max_size_mb=max_size_mb,
            cache_dir=cache_dir,
        )

    def get(self, key: Any) -> Optional[Any]:
        """Get value from SQLite cache."""
        return self._backend.get(key)

    def set(self, key: Any, value: Any) -> None:
        """Set value in SQLite cache."""
        self._backend.set(key, value)

    def clear(self) -> None:
        """Clear all entries from SQLite cache."""
        self._backend.clear()

    def sync(self) -> None:
        """Sync SQLite cache to disk."""
        self._backend.sync()

    def get_stats(self) -> Dict[str, Any]:
        """Get SQLite cache statistics."""
        stats = self._backend.get_stats()
        stats["backend"] = "sqlite"
        return stats

    def __len__(self) -> int:
        """Return number of entries in SQLite cache."""
        return len(self._backend)

    def delete(self, key: Any) -> None:
        """Delete a single entry from the cache."""
        self._backend.delete(key)

    def exists(self, key: Any) -> bool:
        """Check if a key exists in the cache."""
        return self._backend.exists(key)

    def batch_get(self, keys: list) -> Dict[Any, Any]:
        """Get multiple values from the cache."""
        return self._backend.batch_get(keys)

    def batch_set(self, items: Dict[Any, Any]) -> None:
        """Set multiple values in the cache."""
        self._backend.batch_set(items)

    def batch_delete(self, keys: list) -> None:
        """Delete multiple entries from the cache."""
        self._backend.batch_delete(keys)

    def close(self) -> None:
        """Clean up SQLite connections."""
        # The existing implementation handles cleanup in __del__
        # but we can trigger it explicitly if needed
        if hasattr(self._backend, "_local") and hasattr(
            self._backend._local, "connection"
        ):
            connection = self._backend._local.connection
            if connection:
                try:
                    connection.close()
                except Exception:
                    pass
