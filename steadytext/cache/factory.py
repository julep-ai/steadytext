# AIDEV-NOTE: Factory for creating cache backends based on configuration
"""Factory for creating cache backends."""

import os
from pathlib import Path
from typing import Any, Optional

from .base import CacheBackend


def get_cache_backend(
    backend_type: Optional[str] = None,
    capacity: int = 128,
    cache_name: str = "cache",
    max_size_mb: float = 100.0,
    cache_dir: Optional[Path] = None,
    **kwargs: Any,
) -> CacheBackend:
    """Create a cache backend based on configuration.

    Args:
        backend_type: Type of backend ("sqlite", "d1", "memory", or None for auto)
        capacity: Maximum number of entries
        cache_name: Name for the cache
        max_size_mb: Maximum cache size in megabytes
        cache_dir: Directory for cache files
        **kwargs: Backend-specific configuration

    Returns:
        Configured cache backend instance

    Raises:
        ValueError: If backend type is not supported
    """
    # Determine backend type from environment or parameter
    if backend_type is None:
        backend_type = os.environ.get("STEADYTEXT_CACHE_BACKEND", "sqlite").lower()

    # AIDEV-NOTE: Check if we should skip cache initialization (for pytest)
    # Only use memory backend with capacity 0 if explicitly requested
    if os.environ.get("STEADYTEXT_SKIP_CACHE_INIT") == "1" and backend_type == "sqlite":
        from .memory_backend import MemoryBackend

        return MemoryBackend(capacity=0, cache_name=cache_name)

    # Create the appropriate backend
    if backend_type == "sqlite":
        from .sqlite_backend import SQLiteCacheBackend

        return SQLiteCacheBackend(
            capacity=capacity,
            cache_name=cache_name,
            max_size_mb=max_size_mb,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend_type == "d1":
        # AIDEV-NOTE: D1 backend requires API URL and key configuration
        from .d1_backend import D1CacheBackend

        api_url = kwargs.pop("api_url", os.environ.get("STEADYTEXT_D1_API_URL"))
        api_key = kwargs.pop("api_key", os.environ.get("STEADYTEXT_D1_API_KEY"))

        if not api_url or not api_key:
            raise ValueError(
                "D1 backend requires STEADYTEXT_D1_API_URL and STEADYTEXT_D1_API_KEY "
                "environment variables or api_url/api_key parameters"
            )

        return D1CacheBackend(
            capacity=capacity,
            cache_name=cache_name,
            max_size_mb=max_size_mb,
            api_url=api_url,
            api_key=api_key,
            **kwargs,
        )
    elif backend_type == "memory":
        from .memory_backend import MemoryBackend

        return MemoryBackend(
            capacity=capacity, cache_name=cache_name, max_size_mb=max_size_mb, **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported cache backend: {backend_type}. "
            "Supported backends: sqlite, d1, memory"
        )
