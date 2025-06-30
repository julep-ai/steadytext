# AIDEV-NOTE: Tests for the pluggable cache backend system
"""Tests for cache backend system and factory."""

import os
import pytest
from unittest.mock import patch, MagicMock

from steadytext.cache import get_cache_backend
from steadytext.cache.memory_backend import MemoryBackend
from steadytext.cache.sqlite_backend import SQLiteCacheBackend


class TestCacheFactory:
    """Test cache backend factory."""

    def test_default_backend_is_sqlite(self):
        """Test that SQLite is the default backend."""
        # Clear relevant environment variables to test default behavior
        env_copy = os.environ.copy()
        env_copy.pop("STEADYTEXT_SKIP_CACHE_INIT", None)
        env_copy.pop("STEADYTEXT_CACHE_BACKEND", None)
        with patch.dict(os.environ, env_copy, clear=True):
            backend = get_cache_backend()
            assert isinstance(backend, SQLiteCacheBackend)

    def test_memory_backend_selection(self):
        """Test selecting memory backend."""
        backend = get_cache_backend(backend_type="memory")
        assert isinstance(backend, MemoryBackend)

    def test_memory_backend_via_env(self):
        """Test selecting memory backend via environment."""
        with patch.dict(os.environ, {"STEADYTEXT_CACHE_BACKEND": "memory"}):
            backend = get_cache_backend()
            assert isinstance(backend, MemoryBackend)

    def test_sqlite_backend_selection(self):
        """Test explicit SQLite backend selection."""
        # Temporarily unset STEADYTEXT_SKIP_CACHE_INIT for this test
        env_copy = os.environ.copy()
        env_copy.pop("STEADYTEXT_SKIP_CACHE_INIT", None)
        with patch.dict(os.environ, env_copy, clear=True):
            backend = get_cache_backend(backend_type="sqlite")
            assert isinstance(backend, SQLiteCacheBackend)

    def test_d1_backend_selection(self):
        """Test D1 backend selection with required config."""
        pytest.importorskip("httpx")

        with patch("steadytext.cache.d1_backend.D1CacheBackend") as mock_d1_class:
            mock_d1_instance = MagicMock()
            mock_d1_class.return_value = mock_d1_instance

            backend = get_cache_backend(
                backend_type="d1",
                api_url="https://test.workers.dev",
                api_key="test_key",
            )

            assert backend == mock_d1_instance
            mock_d1_class.assert_called_once()

    def test_d1_backend_missing_config(self):
        """Test D1 backend fails without required config."""
        with pytest.raises(ValueError, match="D1 backend requires"):
            get_cache_backend(backend_type="d1")

    def test_d1_backend_from_env(self):
        """Test D1 backend configuration from environment."""
        pytest.importorskip("httpx")

        with patch("steadytext.cache.d1_backend.D1CacheBackend") as mock_d1_class:
            mock_d1_instance = MagicMock()
            mock_d1_class.return_value = mock_d1_instance

            with patch.dict(
                os.environ,
                {
                    "STEADYTEXT_CACHE_BACKEND": "d1",
                    "STEADYTEXT_D1_API_URL": "https://env.workers.dev",
                    "STEADYTEXT_D1_API_KEY": "env_key",
                },
            ):
                backend = get_cache_backend()

                assert backend == mock_d1_instance
                call_kwargs = mock_d1_class.call_args[1]
                assert call_kwargs["api_url"] == "https://env.workers.dev"
                assert call_kwargs["api_key"] == "env_key"

    def test_invalid_backend_type(self):
        """Test error on invalid backend type."""
        with pytest.raises(ValueError, match="Unsupported cache backend"):
            get_cache_backend(backend_type="invalid")

    def test_skip_cache_init_env(self):
        """Test cache initialization skip via environment."""
        with patch.dict(os.environ, {"STEADYTEXT_SKIP_CACHE_INIT": "1"}):
            backend = get_cache_backend()
            assert isinstance(backend, MemoryBackend)
            assert backend.capacity == 0


class TestMemoryBackend:
    """Test memory cache backend."""

    def test_basic_operations(self):
        """Test basic get/set operations."""
        cache = MemoryBackend(capacity=3)

        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Test missing key
        assert cache.get("missing") is None

        # Test overwrite
        cache.set("key1", "new_value")
        assert cache.get("key1") == "new_value"

    def test_capacity_limit(self):
        """Test capacity enforcement."""
        cache = MemoryBackend(capacity=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache) == 2

        # Adding third item should evict first
        cache.set("key3", "value3")
        assert len(cache) == 2
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_clear(self):
        """Test clearing the cache."""
        cache = MemoryBackend()

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_sync_noop(self):
        """Test that sync is a no-op."""
        cache = MemoryBackend()
        cache.sync()  # Should not raise

    def test_stats(self):
        """Test getting statistics."""
        cache = MemoryBackend(capacity=10, max_size_mb=5.0)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        stats = cache.get_stats()
        assert stats["entries"] == 2
        assert stats["backend"] == "memory"
        assert stats["capacity"] == 10
        assert stats["max_size_mb"] == 5.0


class TestCacheBackendInterface:
    """Test that all backends implement the interface correctly."""

    @pytest.fixture(params=["memory", "sqlite"])
    def backend(self, request, tmp_path):
        """Create backend instances for testing."""
        if request.param == "memory":
            return MemoryBackend()
        elif request.param == "sqlite":
            return SQLiteCacheBackend(cache_dir=tmp_path)

    def test_interface_methods(self, backend):
        """Test that all interface methods are implemented."""
        # Test all required methods exist and work
        backend.set("test_key", "test_value")
        assert backend.get("test_key") == "test_value"

        backend.clear()
        assert backend.get("test_key") is None

        backend.sync()  # Should not raise

        stats = backend.get_stats()
        assert isinstance(stats, dict)
        assert "backend" in stats

        length = len(backend)
        assert isinstance(length, int)
        assert length >= 0

        backend.close()  # Should not raise

    def test_none_values(self, backend):
        """Test handling of None values."""
        backend.set("none_key", None)
        assert backend.get("none_key") is None

        # But we should distinguish between missing and None
        assert backend.get("missing_key") is None

    def test_complex_values(self, backend):
        """Test storing complex data types."""
        test_data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (1, 2, 3),
            "none": None,
        }

        backend.set("complex", test_data)
        retrieved = backend.get("complex")

        assert retrieved == test_data

    def test_key_types(self, backend):
        """Test different key types."""
        # String keys
        backend.set("string_key", "value1")
        assert backend.get("string_key") == "value1"

        # Numeric keys
        backend.set(123, "value2")
        assert backend.get(123) == "value2"

        # Tuple keys
        backend.set(("tuple", "key"), "value3")
        assert backend.get(("tuple", "key")) == "value3"


class TestDiskBackedFrecencyCacheWithBackends:
    """Test DiskBackedFrecencyCache with different backends."""

    def test_with_memory_backend(self):
        """Test DiskBackedFrecencyCache with memory backend."""
        from steadytext.disk_backed_frecency_cache import DiskBackedFrecencyCache

        cache = DiskBackedFrecencyCache(backend_type="memory")

        cache.set("test", "value")
        assert cache.get("test") == "value"

        stats = cache.get_stats()
        assert stats["backend"] == "memory"

    def test_with_sqlite_backend(self, tmp_path):
        """Test DiskBackedFrecencyCache with SQLite backend."""
        from steadytext.disk_backed_frecency_cache import DiskBackedFrecencyCache

        # Temporarily unset STEADYTEXT_SKIP_CACHE_INIT for this test
        env_copy = os.environ.copy()
        env_copy.pop("STEADYTEXT_SKIP_CACHE_INIT", None)
        with patch.dict(os.environ, env_copy, clear=True):
            cache = DiskBackedFrecencyCache(backend_type="sqlite", cache_dir=tmp_path)

            cache.set("test", "value")
            assert cache.get("test") == "value"

            stats = cache.get_stats()
            assert stats["backend"] == "sqlite"

    def test_with_d1_backend(self):
        """Test DiskBackedFrecencyCache with D1 backend."""
        pytest.importorskip("httpx")

        with patch("steadytext.cache.d1_backend.D1CacheBackend") as mock_d1_class:
            from steadytext.disk_backed_frecency_cache import (
                DiskBackedFrecencyCache,
            )

            mock_backend = MagicMock()
            mock_backend.get_stats.return_value = {"backend": "d1"}
            mock_d1_class.return_value = mock_backend

            cache = DiskBackedFrecencyCache(
                backend_type="d1",
                api_url="https://test.workers.dev",
                api_key="test_key",
            )

            cache.set("test", "value")
            mock_backend.set.assert_called_with("test", "value")

            cache.get("test")
            mock_backend.get.assert_called_with("test")


class TestCacheManagerWithBackends:
    """Test CacheManager with different backends."""

    def test_cache_manager_default_backend(self):
        """Test CacheManager uses default backend."""
        from steadytext.cache_manager import get_generation_cache

        # Clear relevant environment variables to test default behavior
        env_copy = os.environ.copy()
        env_copy.pop("STEADYTEXT_SKIP_CACHE_INIT", None)
        env_copy.pop("STEADYTEXT_CACHE_BACKEND", None)
        with patch.dict(os.environ, env_copy, clear=True):
            cache = get_generation_cache()
            assert hasattr(cache, "_backend")

    def test_cache_manager_memory_backend(self):
        """Test CacheManager with memory backend."""
        from steadytext.cache_manager import CacheManager

        with patch.dict(os.environ, {"STEADYTEXT_CACHE_BACKEND": "memory"}):
            manager = CacheManager()
            # Reset cached instances
            manager._generation_cache = None
            manager._embedding_cache = None

            gen_cache = manager.get_generation_cache()
            assert hasattr(gen_cache, "_backend")

            embed_cache = manager.get_embedding_cache()
            assert hasattr(embed_cache, "_backend")

    def test_cache_manager_d1_backend(self):
        """Test CacheManager with D1 backend."""
        pytest.importorskip("httpx")

        with patch("steadytext.cache.d1_backend.D1CacheBackend") as mock_d1_class:
            from steadytext.cache_manager import CacheManager

            mock_backend = MagicMock()
            mock_d1_class.return_value = mock_backend

            env_copy = os.environ.copy()
            env_copy.pop("STEADYTEXT_SKIP_CACHE_INIT", None)
            env_copy["STEADYTEXT_CACHE_BACKEND"] = "d1"
            env_copy["STEADYTEXT_D1_API_URL"] = "https://test.workers.dev"
            env_copy["STEADYTEXT_D1_API_KEY"] = "test_key"
            with patch.dict(os.environ, env_copy, clear=True):
                manager = CacheManager()
                # Reset cached instances
                manager._generation_cache = None
                manager._embedding_cache = None

                gen_cache = manager.get_generation_cache()
                assert hasattr(gen_cache, "_backend")

                # D1 backend should have been created with correct params
                assert mock_d1_class.call_count >= 1
                call_kwargs = mock_d1_class.call_args[1]
                assert call_kwargs["api_url"] == "https://test.workers.dev"
                assert call_kwargs["api_key"] == "test_key"
