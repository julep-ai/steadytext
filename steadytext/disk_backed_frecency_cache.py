# AIDEV-NOTE: Disk-backed frecency cache implementation with configurable size limits
# Extends the in-memory FrecencyCache with persistent storage and automatic eviction
# AIDEV-NOTE: Uses pickle for serialization to handle arbitrary Python objects
# AIDEV-TODO: Consider adding compression support for large cache files
from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from .frecency_cache import FrecencyCache
    from .utils import get_cache_dir, logger
except ImportError:
    # For direct testing
    from frecency_cache import FrecencyCache
    from utils import get_cache_dir, logger


class DiskBackedFrecencyCache(FrecencyCache):
    """Disk-backed frecency cache with configurable size limits.
    
    Extends the in-memory FrecencyCache with:
    - Persistent storage to disk
    - Configurable maximum cache file size in MB
    - Automatic eviction when size limit is exceeded
    - Thread-safe operations
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
            capacity: Maximum number of entries in memory
            cache_name: Name for the cache file (without extension)
            max_size_mb: Maximum cache file size in megabytes
            cache_dir: Directory for cache file (defaults to steadytext cache dir)
        """
        super().__init__(capacity)
        self.cache_name = cache_name
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        
        # AIDEV-NOTE: Use the existing cache directory structure
        if cache_dir is None:
            self.cache_dir = get_cache_dir().parent / "caches"
        else:
            self.cache_dir = cache_dir
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"{cache_name}.pkl"
        
        # Load existing cache from disk if available
        self._load_from_disk()
        
    def _load_from_disk(self) -> None:
        """Load cache data from disk if file exists."""
        if not self.cache_file.exists():
            logger.debug(f"No existing cache file at {self.cache_file}")
            return
            
        try:
            with self.cache_file.open("rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict) and "data" in data and "meta" in data:
                    self._data = data["data"]
                    self._meta = data["meta"]
                    self._counter = data.get("counter", 0)
                    logger.info(f"Loaded {len(self._data)} entries from disk cache")
                else:
                    logger.warning(f"Invalid cache format in {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")
            # AIDEV-NOTE: On corruption, start fresh rather than crash
            self._data.clear()
            self._meta.clear()
            self._counter = 0
    
    def _save_to_disk(self) -> None:
        """Save current cache state to disk."""
        try:
            # AIDEV-NOTE: Check file size before saving
            if self._should_evict_for_size():
                self._evict_until_size_ok()
                
            data = {
                "data": self._data,
                "meta": self._meta,
                "counter": self._counter
            }
            
            # Write to temporary file first for atomicity
            temp_file = self.cache_file.with_suffix(".tmp")
            with temp_file.open("wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic rename
            temp_file.replace(self.cache_file)
            
        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")
            # AIDEV-NOTE: Continue operation even if disk save fails
    
    def _get_cache_size(self) -> int:
        """Get current cache file size in bytes."""
        if self.cache_file.exists():
            return self.cache_file.stat().st_size
        return 0
    
    def _should_evict_for_size(self) -> bool:
        """Check if cache file exceeds size limit."""
        # AIDEV-NOTE: Estimate size by serializing current data
        try:
            data = {
                "data": self._data,
                "meta": self._meta,
                "counter": self._counter
            }
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            return len(serialized) > self.max_size_bytes
        except:
            # If we can't serialize, assume we need to evict
            return True
    
    def _evict_until_size_ok(self) -> None:
        """Evict entries until cache size is under limit."""
        # AIDEV-NOTE: Evict 20% of entries when over limit
        target_entries = int(len(self._data) * 0.8)
        
        while len(self._data) > target_entries:
            # Find victim using frecency algorithm
            if not self._meta:
                break
            victim = min(self._meta.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
            self._data.pop(victim, None)
            self._meta.pop(victim, None)
            
        logger.info(f"Evicted entries to reduce cache size. Remaining: {len(self._data)}")
    
    def get(self, key: Any) -> Any | None:
        """Get value from cache, updating frecency metadata."""
        # AIDEV-NOTE: Parent class handles locking and metadata update
        result = super().get(key)
        # No need to save on every get - too expensive
        return result
    
    def set(self, key: Any, value: Any) -> None:
        """Set value in cache and persist to disk."""
        # AIDEV-NOTE: Don't acquire lock here - parent's set() already handles locking
        super().set(key, value)
        # Save to disk after set
        with self._lock:
            self._save_to_disk()
    
    def clear(self) -> None:
        """Clear cache and remove disk file."""
        super().clear()
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.info(f"Removed cache file: {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to remove cache file: {e}")
    
    def sync(self) -> None:
        """Explicitly sync cache to disk."""
        with self._lock:
            self._save_to_disk()
    
    def __del__(self):
        """Ensure cache is saved on object destruction."""
        # AIDEV-NOTE: Best effort save on cleanup
        try:
            self._save_to_disk()
        except:
            pass