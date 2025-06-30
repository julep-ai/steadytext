# AIDEV-NOTE: Tests for the D1 cache backend
"""Tests for Cloudflare D1 cache backend."""

import os
import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from steadytext.cache.d1_backend import D1CacheBackend


# Skip tests if httpx is not available
httpx = pytest.importorskip("httpx")


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(self, json_data=None, status_code=200, headers=None):
        self.json_data = json_data or {}
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("Mock error", request=Mock(), response=self)


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for testing."""
    with patch("steadytext.cache.d1_backend.httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Default successful responses
        mock_client.post.return_value = MockResponse({"success": True})
        mock_client.get.return_value = MockResponse({"found": False})

        yield mock_client


@pytest.fixture
def d1_backend(mock_httpx_client):
    """Create D1 backend with mocked HTTP client."""
    backend = D1CacheBackend(
        capacity=10,
        cache_name="test_cache",
        max_size_mb=1.0,
        api_url="https://test.workers.dev",
        api_key="test_key",
        timeout=5.0,
        max_retries=2,
    )
    return backend


class TestD1CacheBackend:
    """Test D1 cache backend functionality."""

    def test_initialization(self, mock_httpx_client):
        """Test D1 backend initialization."""
        backend = D1CacheBackend(api_url="https://test.workers.dev", api_key="test_key")

        assert backend.api_url == "https://test.workers.dev"
        assert backend.api_key == "test_key"
        assert backend.cache_name == "d1_cache"

        # Check that initialization endpoint was called
        mock_httpx_client.post.assert_called_with(
            "/api/init", json={"cache_name": "d1_cache", "max_size_mb": 100.0}
        )

    def test_get_not_found(self, d1_backend, mock_httpx_client):
        """Test getting a non-existent key."""
        mock_httpx_client.post.return_value = MockResponse({"found": False})

        result = d1_backend.get("missing_key")
        assert result is None

        mock_httpx_client.post.assert_called_with(
            "/api/get", json={"cache_name": "test_cache", "key": "missing_key"}
        )

    def test_get_found(self, d1_backend, mock_httpx_client):
        """Test getting an existing key."""
        # Mock the serialization format
        import pickle
        import base64

        test_value = {"data": "test"}
        serialized = base64.b64encode(
            pickle.dumps(test_value, protocol=pickle.HIGHEST_PROTOCOL)
        ).decode("ascii")

        mock_httpx_client.post.return_value = MockResponse(
            {"found": True, "value": serialized}
        )

        result = d1_backend.get("test_key")
        assert result == test_value

    def test_set(self, d1_backend, mock_httpx_client):
        """Test setting a value."""
        mock_httpx_client.post.return_value = MockResponse({"success": True})

        d1_backend.set("test_key", {"data": "test"})

        # Check that set endpoint was called
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "/api/set"

        json_data = call_args[1]["json"]
        assert json_data["cache_name"] == "test_cache"
        assert json_data["key"] == "test_key"
        assert "value" in json_data
        assert "size" in json_data

    def test_clear(self, d1_backend, mock_httpx_client):
        """Test clearing the cache."""
        mock_httpx_client.post.return_value = MockResponse({"success": True})

        d1_backend.clear()

        mock_httpx_client.post.assert_called_with(
            "/api/clear", json={"cache_name": "test_cache"}
        )

    def test_get_stats(self, d1_backend, mock_httpx_client):
        """Test getting cache statistics."""
        stats_response = {
            "entry_count": 5,
            "total_size": 1024,
            "avg_frequency": 2.5,
            "max_frequency": 10,
        }
        mock_httpx_client.post.return_value = MockResponse(stats_response)

        stats = d1_backend.get_stats()

        assert stats["entry_count"] == 5
        assert stats["backend"] == "d1"
        assert stats["api_url"] == "https://test.workers.dev"

    def test_retry_on_timeout(self, d1_backend, mock_httpx_client):
        """Test retry logic on timeout errors."""
        # First call times out, second succeeds
        mock_httpx_client.post.side_effect = [
            httpx.TimeoutException("Timeout"),
            MockResponse({"found": False}),
        ]

        with patch("time.sleep"):  # Skip actual sleep in tests
            result = d1_backend.get("test_key")

        assert result is None
        # 3 calls: 1 for initialization + 2 for the retried request
        assert mock_httpx_client.post.call_count == 3

    def test_retry_on_rate_limit(self, d1_backend, mock_httpx_client):
        """Test retry logic on rate limiting."""
        # First call rate limited, second succeeds
        mock_httpx_client.post.side_effect = [
            MockResponse(status_code=429, headers={"Retry-After": "0.1"}),
            MockResponse({"found": False}),
        ]

        with patch("time.sleep"):  # Skip actual sleep in tests
            result = d1_backend.get("test_key")

        assert result is None
        # 3 calls: 1 for initialization + 2 for the retried request
        assert mock_httpx_client.post.call_count == 3

    def test_batch_get(self, d1_backend, mock_httpx_client):
        """Test batch get operation."""
        import pickle
        import base64

        # Mock response for batch get
        batch_results = {
            "key1": {
                "found": True,
                "value": base64.b64encode(
                    pickle.dumps("value1", protocol=pickle.HIGHEST_PROTOCOL)
                ).decode("ascii"),
            },
            "key2": {"found": False},
            "key3": {
                "found": True,
                "value": base64.b64encode(
                    pickle.dumps("value3", protocol=pickle.HIGHEST_PROTOCOL)
                ).decode("ascii"),
            },
        }

        mock_httpx_client.post.return_value = MockResponse({"results": batch_results})

        keys = ["key1", "key2", "key3"]
        results = d1_backend.batch_get(keys)

        assert results["key1"] == "value1"
        assert "key2" not in results
        assert results["key3"] == "value3"

    def test_batch_set(self, d1_backend, mock_httpx_client):
        """Test batch set operation."""
        mock_httpx_client.post.return_value = MockResponse({"success": True})

        items = {"key1": "value1", "key2": {"data": "value2"}, "key3": [1, 2, 3]}

        d1_backend.batch_set(items)

        # Check that batch set endpoint was called
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "/api/batch/set"

        json_data = call_args[1]["json"]
        assert json_data["cache_name"] == "test_cache"
        assert len(json_data["items"]) == 3

    def test_batch_size_limit(self, d1_backend, mock_httpx_client):
        """Test that large batches are split correctly."""
        d1_backend.batch_size = 2  # Small batch size for testing

        mock_httpx_client.post.return_value = MockResponse({"success": True})

        # Create more items than batch size
        items = {f"key{i}": f"value{i}" for i in range(5)}

        d1_backend.batch_set(items)

        # Should be called 3 times (2 + 2 + 1)
        assert mock_httpx_client.post.call_count >= 3

    def test_error_handling(self, d1_backend, mock_httpx_client):
        """Test graceful error handling."""
        # Simulate server error
        mock_httpx_client.post.side_effect = Exception("Server error")

        # Should not raise, but return None
        result = d1_backend.get("test_key")
        assert result is None

        # Should not raise for set either
        d1_backend.set("test_key", "value")  # No exception

        # Stats should return error info
        stats = d1_backend.get_stats()
        assert "error" in stats

    def test_sync_is_noop(self, d1_backend):
        """Test that sync is a no-op for D1."""
        d1_backend.sync()  # Should not raise

    def test_len(self, d1_backend, mock_httpx_client):
        """Test __len__ method."""
        mock_httpx_client.post.return_value = MockResponse(
            {"entry_count": 42, "backend": "d1"}
        )

        assert len(d1_backend) == 42

    def test_close(self, d1_backend, mock_httpx_client):
        """Test cleanup on close."""
        # Just verify the close method doesn't raise an exception
        d1_backend.close()
        # The actual client close testing is complex due to mocking setup


@pytest.mark.skipif(
    not os.environ.get("STEADYTEXT_D1_API_URL"),
    reason="D1 integration tests require STEADYTEXT_D1_API_URL",
)
class TestD1CacheBackendIntegration:
    """Integration tests with real D1 Worker (if configured)."""

    @pytest.fixture
    def real_d1_backend(self):
        """Create real D1 backend for integration testing."""
        return D1CacheBackend(
            api_url=os.environ["STEADYTEXT_D1_API_URL"],
            api_key=os.environ["STEADYTEXT_D1_API_KEY"],
            cache_name="test_integration",
        )

    def test_real_set_get(self, real_d1_backend):
        """Test real set and get operations."""
        test_key = f"test_key_{int(time.time())}"
        test_value = {"data": "integration test", "timestamp": time.time()}

        # Set value
        real_d1_backend.set(test_key, test_value)

        # Get value
        result = real_d1_backend.get(test_key)
        assert result == test_value

        # Clean up
        real_d1_backend.clear()

    def test_real_batch_operations(self, real_d1_backend):
        """Test real batch operations."""
        timestamp = int(time.time())
        items = {
            f"batch_key_{timestamp}_1": "value1",
            f"batch_key_{timestamp}_2": {"nested": "value2"},
            f"batch_key_{timestamp}_3": [1, 2, 3],
        }

        # Batch set
        real_d1_backend.batch_set(items)

        # Batch get
        keys = list(items.keys())
        results = real_d1_backend.batch_get(keys)

        assert len(results) == 3
        for key, expected_value in items.items():
            assert results[key] == expected_value

        # Clean up
        real_d1_backend.clear()
