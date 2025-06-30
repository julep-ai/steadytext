# AIDEV-NOTE: Simple in-memory cache backend for testing and lightweight usage
"""In-memory cache backend implementation."""

from pathlib import Path
from typing import Any, Dict, Optional

from .base import CacheBackend


class MemoryBackend(CacheBackend):
    """Simple in-memory cache backend.

    This backend stores all data in memory and does not persist to disk.
    Useful for testing and scenarios where persistence is not required.
    """

    def __init__(
        self,
        capacity: int = 128,
        cache_name: str = "memory_cache",
        max_size_mb: float = 100.0,
        cache_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize memory backend."""
        super().__init__(capacity, cache_name, max_size_mb, cache_dir, **kwargs)
        self._data: Dict[Any, Any] = {}

    def get(self, key: Any) -> Optional[Any]:
        """Get value from memory cache."""
        return self._data.get(key)

    def set(self, key: Any, value: Any) -> None:
        """Set value in memory cache."""
        # Simple capacity enforcement
        if len(self._data) >= self.capacity and key not in self._data:
            # Remove oldest entry (simple FIFO for now)
            if self._data:
                oldest_key = next(iter(self._data))
                del self._data[oldest_key]

        self._data[key] = value

    def clear(self) -> None:
        """Clear all entries from memory."""
        self._data.clear()

    def sync(self) -> None:
        """No-op for memory backend."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics."""
        return {
            "entries": len(self._data),
            "backend": "memory",
            "capacity": self.capacity,
            "max_size_mb": self.max_size_mb,
        }

    def __len__(self) -> int:
        """Return number of entries in memory."""
        return len(self._data)

    def delete(self, key: Any) -> None:
        """Delete a single entry from the cache."""
        if key in self._data:
            del self._data[key]

    def exists(self, key: Any) -> bool:
        """Check if a key exists in the cache."""
        return key in self._data
