# AIDEV-NOTE: Centralized cache management for SteadyText, providing singleton cache instances shared between the daemon and direct access. It has proper __len__ method support and improved error handling.

import os as _os
from typing import Optional, cast
from pathlib import Path

from .disk_backed_frecency_cache import DiskBackedFrecencyCache
from .utils import get_cache_dir


# AIDEV-NOTE: Dummy cache for testing/collection scenarios
class _DummyCache(DiskBackedFrecencyCache):
    """A no-op cache implementation that satisfies type checking."""

    def __init__(self):
        # Don't call super().__init__() to avoid any initialization
        self.capacity = 0
        self._backend = None
        self._memory_cache = {}
        self.cache_dir = None

    def get(self, key):
        return None

    def set(self, key, value):
        pass

    def clear(self):
        pass

    def sync(self):
        pass

    def __len__(self):
        return 0

    def get_stats(self):
        return {"size": 0, "capacity": 0}


class CacheManager:
    """Centralized cache manager for SteadyText.

    AIDEV-NOTE: This singleton ensures cache consistency by providing the same cache objects to both the daemon and direct API calls.
    """

    _instance: Optional["CacheManager"] = None
    _generation_cache: Optional[DiskBackedFrecencyCache] = None
    _embedding_cache: Optional[DiskBackedFrecencyCache] = None
    _reranking_cache: Optional[DiskBackedFrecencyCache] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
        return cls._instance

    def get_generation_cache(self) -> DiskBackedFrecencyCache:
        """Get the shared generation cache instance."""
        # AIDEV-NOTE: Check if cache initialization should be skipped
        if _os.environ.get("STEADYTEXT_SKIP_CACHE_INIT") == "1":
            # Return a dummy cache that does nothing
            return _DummyCache()

        if self._generation_cache is None:
            # AIDEV-NOTE: Support configurable cache backend
            backend_type = _os.environ.get("STEADYTEXT_CACHE_BACKEND")

            # Collect backend-specific configuration
            backend_kwargs = {}
            if backend_type == "d1":
                backend_kwargs["api_url"] = _os.environ.get("STEADYTEXT_D1_API_URL")
                backend_kwargs["api_key"] = _os.environ.get("STEADYTEXT_D1_API_KEY")

            self._generation_cache = DiskBackedFrecencyCache(
                capacity=int(
                    _os.environ.get("STEADYTEXT_GENERATION_CACHE_CAPACITY", "256")
                ),
                cache_name="generation_cache",
                max_size_mb=float(
                    _os.environ.get("STEADYTEXT_GENERATION_CACHE_MAX_SIZE_MB", "50.0")
                ),
                cache_dir=Path(get_cache_dir()) / "caches",
                backend_type=backend_type,
                **backend_kwargs,
            )
        return self._generation_cache

    def get_embedding_cache(self) -> DiskBackedFrecencyCache:
        """Get the shared embedding cache instance."""
        # AIDEV-NOTE: Check if cache initialization should be skipped
        if _os.environ.get("STEADYTEXT_SKIP_CACHE_INIT") == "1":
            # Return a dummy cache that does nothing
            return _DummyCache()

        if self._embedding_cache is None:
            # AIDEV-NOTE: Support configurable cache backend
            backend_type = _os.environ.get("STEADYTEXT_CACHE_BACKEND")

            # Collect backend-specific configuration
            backend_kwargs = {}
            if backend_type == "d1":
                backend_kwargs["api_url"] = _os.environ.get("STEADYTEXT_D1_API_URL")
                backend_kwargs["api_key"] = _os.environ.get("STEADYTEXT_D1_API_KEY")

            self._embedding_cache = DiskBackedFrecencyCache(
                capacity=int(
                    _os.environ.get("STEADYTEXT_EMBEDDING_CACHE_CAPACITY", "512")
                ),
                cache_name="embedding_cache",
                max_size_mb=float(
                    _os.environ.get("STEADYTEXT_EMBEDDING_CACHE_MAX_SIZE_MB", "100.0")
                ),
                cache_dir=Path(get_cache_dir()) / "caches",
                backend_type=backend_type,
                **backend_kwargs,
            )
        return self._embedding_cache

    def get_reranking_cache(self) -> DiskBackedFrecencyCache:
        """Get the shared reranking cache instance."""
        # AIDEV-NOTE: Check if cache initialization should be skipped
        if _os.environ.get("STEADYTEXT_SKIP_CACHE_INIT") == "1":
            # Return a dummy cache that does nothing
            return _DummyCache()

        if self._reranking_cache is None:
            # AIDEV-NOTE: Support configurable cache backend
            backend_type = _os.environ.get("STEADYTEXT_CACHE_BACKEND")

            # Collect backend-specific configuration
            backend_kwargs = {}
            if backend_type == "d1":
                backend_kwargs["api_url"] = _os.environ.get("STEADYTEXT_D1_API_URL")
                backend_kwargs["api_key"] = _os.environ.get("STEADYTEXT_D1_API_KEY")

            self._reranking_cache = DiskBackedFrecencyCache(
                capacity=int(
                    _os.environ.get("STEADYTEXT_RERANKING_CACHE_CAPACITY", "256")
                ),
                cache_name="reranking_cache",
                max_size_mb=float(
                    _os.environ.get("STEADYTEXT_RERANKING_CACHE_MAX_SIZE_MB", "25.0")
                ),
                cache_dir=Path(get_cache_dir()) / "caches",
                backend_type=backend_type,
                **backend_kwargs,
            )
        return self._reranking_cache

    def clear_all_caches(self):
        """Clear all cache instances. Used for testing."""
        if self._generation_cache is not None:
            self._generation_cache.clear()
        if self._embedding_cache is not None:
            self._embedding_cache.clear()
        if self._reranking_cache is not None:
            self._reranking_cache.clear()

    def get_cache_stats(self) -> dict:
        """Get statistics for all caches."""
        stats = {}

        # AIDEV-NOTE: Use cache.get_stats() method for comprehensive statistics
        # instead of direct len() calls which may not be implemented
        try:
            if self._generation_cache is not None:
                cache_stats = self._generation_cache.get_stats()
                stats["generation"] = {
                    "size": len(self._generation_cache),
                    "capacity": self._generation_cache.capacity,
                    **cache_stats,
                }
        except Exception as e:
            # Fallback if len() fails
            stats["generation"] = {"error": str(e)}

        try:
            if self._embedding_cache is not None:
                cache_stats = self._embedding_cache.get_stats()
                stats["embedding"] = {
                    "size": len(self._embedding_cache),
                    "capacity": self._embedding_cache.capacity,
                    **cache_stats,
                }
        except Exception as e:
            # Fallback if len() fails
            stats["embedding"] = {"error": str(e)}

        try:
            if self._reranking_cache is not None:
                cache_stats = self._reranking_cache.get_stats()
                stats["reranking"] = {
                    "size": len(self._reranking_cache),
                    "capacity": self._reranking_cache.capacity,
                    **cache_stats,
                }
        except Exception as e:
            # Fallback if len() fails
            stats["reranking"] = {"error": str(e)}

        return stats


# AIDEV-NOTE: The cache manager singleton is initialized lazily on first use.

# AIDEV-NOTE: Module-level cache manager instance, initialized lazily
_cache_manager: Optional[CacheManager] = None


def get_generation_cache() -> DiskBackedFrecencyCache:
    """Get the global generation cache instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager.get_generation_cache()


def get_embedding_cache() -> DiskBackedFrecencyCache:
    """Get the global embedding cache instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager.get_embedding_cache()


def get_reranking_cache() -> DiskBackedFrecencyCache:
    """Get the global reranking cache instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager.get_reranking_cache()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    # AIDEV-NOTE: Cast since we know it's not None after initialization
    return cast(CacheManager, _cache_manager)
