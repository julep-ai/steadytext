# AIDEV-NOTE: Centralized cache management for SteadyText
# Provides singleton cache instances shared between daemon and direct access
# Ensures consistent caching behavior across all components

import os as _os
from typing import Optional
from pathlib import Path

from .disk_backed_frecency_cache import DiskBackedFrecencyCache
from .utils import get_cache_dir


class CacheManager:
    """Centralized cache manager for SteadyText.

    AIDEV-NOTE: Singleton pattern ensures all components use the same cache instances.
    Provides thread-safe and process-safe cache access through SQLite backend.
    """

    _instance: Optional["CacheManager"] = None
    _generation_cache: Optional[DiskBackedFrecencyCache] = None
    _embedding_cache: Optional[DiskBackedFrecencyCache] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
        return cls._instance

    def get_generation_cache(self) -> DiskBackedFrecencyCache:
        """Get the shared generation cache instance."""
        if self._generation_cache is None:
            self._generation_cache = DiskBackedFrecencyCache(
                capacity=int(
                    _os.environ.get("STEADYTEXT_GENERATION_CACHE_CAPACITY", "256")
                ),
                cache_name="generation_cache",
                max_size_mb=float(
                    _os.environ.get("STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB", "50.0")
                ),
                cache_dir=Path(get_cache_dir()) / "caches",
            )
        return self._generation_cache

    def get_embedding_cache(self) -> DiskBackedFrecencyCache:
        """Get the shared embedding cache instance."""
        if self._embedding_cache is None:
            self._embedding_cache = DiskBackedFrecencyCache(
                capacity=int(
                    _os.environ.get("STEADYTEXT_EMBEDDING_CACHE_CAPACITY", "512")
                ),
                cache_name="embedding_cache",
                max_size_mb=float(
                    _os.environ.get("STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB", "100.0")
                ),
                cache_dir=Path(get_cache_dir()) / "caches",
            )
        return self._embedding_cache

    def clear_all_caches(self):
        """Clear all cache instances. Used for testing."""
        if self._generation_cache is not None:
            self._generation_cache.clear()
        if self._embedding_cache is not None:
            self._embedding_cache.clear()

    def get_cache_stats(self) -> dict:
        """Get statistics for all caches."""
        stats = {}
        if self._generation_cache is not None:
            stats["generation"] = {
                "size": len(self._generation_cache),
                "hits": getattr(self._generation_cache, "_hits", 0),
                "misses": getattr(self._generation_cache, "_misses", 0),
            }
        if self._embedding_cache is not None:
            stats["embedding"] = {
                "size": len(self._embedding_cache),
                "hits": getattr(self._embedding_cache, "_hits", 0),
                "misses": getattr(self._embedding_cache, "_misses", 0),
            }
        return stats


# AIDEV-NOTE: Global cache manager instance
_cache_manager = CacheManager()


def get_generation_cache() -> DiskBackedFrecencyCache:
    """Get the global generation cache instance."""
    return _cache_manager.get_generation_cache()


def get_embedding_cache() -> DiskBackedFrecencyCache:
    """Get the global embedding cache instance."""
    return _cache_manager.get_embedding_cache()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return _cache_manager
