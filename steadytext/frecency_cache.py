# AIDEV-NOTE: Simple in-memory frecency cache for deterministic functions
# Combines frequency and recency to evict the least useful entry when capacity is exceeded
from __future__ import annotations

import threading
from typing import Any, Dict, Tuple


class FrecencyCache:
    """Rolling in-memory cache with simple frecency eviction."""

    def __init__(self, capacity: int = 128) -> None:
        self.capacity = capacity
        self._data: Dict[Any, Any] = {}
        self._meta: Dict[Any, Tuple[int, int]] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        with self._lock:
            if key in self._data:
                freq, _ = self._meta[key]
                self._counter += 1
                self._meta[key] = (freq + 1, self._counter)
                return self._data[key]
            return None

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            if key in self._data:
                freq, _ = self._meta[key]
                self._counter += 1
                self._meta[key] = (freq + 1, self._counter)
                self._data[key] = value
                return

            if len(self._data) >= self.capacity:
                victim = min(self._meta.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
                self._data.pop(victim, None)
                self._meta.pop(victim, None)

            self._counter += 1
            self._data[key] = value
            self._meta[key] = (1, self._counter)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._meta.clear()
            self._counter = 0
