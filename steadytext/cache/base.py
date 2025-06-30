# AIDEV-NOTE: Abstract base class defining the interface for all cache backends
"""Abstract base class for cache backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class CacheBackend(ABC):
    """Abstract base class defining the interface for cache backends.

    All cache backends must implement these methods to ensure compatibility
    with the SteadyText caching system.
    """

    def __init__(
        self,
        capacity: int = 128,
        cache_name: str = "cache",
        max_size_mb: float = 100.0,
        cache_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize cache backend.

        Args:
            capacity: Maximum number of entries (may be advisory for some backends)
            cache_name: Name for the cache (used for identification)
            max_size_mb: Maximum cache size in megabytes
            cache_dir: Directory for cache files (if applicable)
            **kwargs: Backend-specific configuration options
        """
        self.capacity = capacity
        self.cache_name = cache_name
        self.max_size_mb = max_size_mb
        self.cache_dir = cache_dir
        self._config = kwargs

    @abstractmethod
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        pass

    @abstractmethod
    def set(self, key: Any, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        pass

    @abstractmethod
    def delete(self, key: Any) -> None:
        """Delete an entry from the cache.

        Args:
            key: The key to delete.
        """
        pass

    def exists(self, key: Any) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: The key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        return self.get(key) is not None

    def batch_get(self, keys: List[Any]) -> Dict[Any, Any]:
        """Get multiple values from the cache.

        Args:
            keys: A list of keys to retrieve.

        Returns:
            A dictionary mapping keys to their found values.
        """
        results = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                results[key] = value
        return results

    def batch_set(self, items: Dict[Any, Any]) -> None:
        """Set multiple values in the cache.

        Args:
            items: A dictionary of key-value pairs to set.
        """
        for key, value in items.items():
            self.set(key, value)

    def batch_delete(self, keys: List[Any]) -> None:
        """Delete multiple entries from the cache.

        Args:
            keys: A list of keys to delete.
        """
        for key in keys:
            self.delete(key)

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def sync(self) -> None:
        """Sync cache to persistent storage (if applicable)."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics (backend-specific)
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return number of entries in cache.

        Returns:
            Number of cached entries
        """
        pass

    def close(self) -> None:
        """Clean up resources (optional).

        Backends can override this to perform cleanup operations.
        """
        pass
