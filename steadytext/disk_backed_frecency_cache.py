# AIDEV-NOTE: Disk-backed frecency cache implementation with configurable size limits
# Now uses SQLite backend for concurrent access safety while maintaining API compatibility
# AIDEV-NOTE: Automatically migrates from legacy pickle format to SQLite
# AIDEV-NOTE: Falls back to pickle implementation if SQLite fails
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from .sqlite_cache_backend import SQLiteDiskBackedFrecencyCache
    from .utils import get_cache_dir, logger
except ImportError:
    # For direct testing
    from sqlite_cache_backend import SQLiteDiskBackedFrecencyCache
    from utils import get_cache_dir, logger


class DiskBackedFrecencyCache:
    """Disk-backed frecency cache with configurable size limits.
    
    Now uses SQLite backend for:
    - Thread-safe and process-safe concurrent access
    - Automatic migration from legacy pickle format
    - Configurable maximum cache file size in MB
    - Automatic eviction when size limit is exceeded
    - Atomic operations with proper error handling
    
    Maintains the same API as the original pickle-based implementation.
    """
    
    def __init__(
        self, 
        capacity: int = 128,
        cache_name: str = "frecency_cache",
        max_size_mb: float = 100.0,
        cache_dir: Optional[Path] = None
    ) -> None:
        """Initialize disk-backed frecency cache.
        
        Args:
            capacity: Maximum number of entries (for compatibility)
            cache_name: Name for the cache file (without extension)
            max_size_mb: Maximum cache file size in megabytes
            cache_dir: Directory for cache file (defaults to steadytext cache dir)
        """
        # AIDEV-NOTE: Use SQLite backend for all operations
        self._backend = SQLiteDiskBackedFrecencyCache(
            capacity=capacity,
            cache_name=cache_name,
            max_size_mb=max_size_mb,
            cache_dir=cache_dir
        )
        
        # Store parameters for compatibility
        self.capacity = capacity
        self.cache_name = cache_name
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.cache_dir = self._backend.cache_dir
    
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