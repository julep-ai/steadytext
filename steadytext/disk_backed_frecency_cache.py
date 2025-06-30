# AIDEV-NOTE: A disk-backed frecency cache that uses pluggable cache backends through the factory pattern
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .cache import CacheBackend, get_cache_backend


class DiskBackedFrecencyCache:
    """Disk-backed frecency cache with configurable backends.

    Supports multiple backends through a pluggable architecture:
    - SQLite: Thread-safe and process-safe with automatic migration
    - D1: Cloudflare's distributed SQLite (requires proxy Worker)
    - Memory: Simple in-memory cache for testing

    Maintains the same API as the original implementation.
    """

    _backend: CacheBackend

    def __init__(
        self,
        capacity: int = 128,
        cache_name: str = "frecency_cache",
        max_size_mb: float = 100.0,
        cache_dir: Optional[Path] = None,
        backend_type: Optional[str] = None,
        **backend_kwargs: Any,
    ) -> None:
        """Initialize disk-backed frecency cache.

        Args:
            capacity: Maximum number of entries (for compatibility)
            cache_name: Name for the cache file (without extension)
            max_size_mb: Maximum cache file size in megabytes
            cache_dir: Directory for cache file (defaults to steadytext cache dir)
            backend_type: Type of backend to use (sqlite, d1, memory)
            **backend_kwargs: Additional backend-specific configuration
        """
        # AIDEV-NOTE: Use the factory to create the appropriate backend
        self._backend = get_cache_backend(
            backend_type=backend_type,
            capacity=capacity,
            cache_name=cache_name,
            max_size_mb=max_size_mb,
            cache_dir=cache_dir,
            **backend_kwargs,
        )

        # Store parameters for compatibility
        self.capacity = capacity
        self.cache_name = cache_name
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.cache_dir = cache_dir

    def get(self, key: Any) -> Any | None:
        """Get value from cache, updating frecency metadata."""
        return self._backend.get(key)

    def set(self, key: Any, value: Any) -> None:
        """Set value in cache and persist to disk."""
        self._backend.set(key, value)

    def clear(self) -> None:
        """Clear cache and remove disk file."""
        self._backend.clear()

    def sync(self) -> None:
        """Explicitly sync cache to disk."""
        self._backend.sync()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring and debugging."""
        return self._backend.get_stats()

    def __len__(self) -> int:
        """Return number of entries in cache."""
        return len(self._backend)
