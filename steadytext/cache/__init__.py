# AIDEV-NOTE: Cache backend module for pluggable cache implementations
"""Cache backend module providing pluggable cache implementations."""

from .base import CacheBackend
from .factory import get_cache_backend

__all__ = ["CacheBackend", "get_cache_backend"]
