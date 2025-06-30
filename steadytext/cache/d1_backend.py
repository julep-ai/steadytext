# AIDEV-NOTE: D1 cache backend that communicates with a Cloudflare Worker proxy
"""Cloudflare D1 cache backend implementation."""

import os
import pickle
import time
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

from ..utils import logger
from .base import CacheBackend

# AIDEV-NOTE: Centralize API endpoint paths for maintainability
API_INIT_ENDPOINT = "/api/init"
API_GET_ENDPOINT = "/api/get"
API_SET_ENDPOINT = "/api/set"
API_DELETE_ENDPOINT = "/api/delete"
API_EXISTS_ENDPOINT = "/api/exists"
API_CLEAR_ENDPOINT = "/api/clear"
API_STATS_ENDPOINT = "/api/stats"
API_BATCH_GET_ENDPOINT = "/api/batch/get"
API_BATCH_SET_ENDPOINT = "/api/batch/set"
API_BATCH_DELETE_ENDPOINT = "/api/batch/delete"


class D1CacheBackend(CacheBackend):
    """D1-based cache backend using a proxy Worker API.

    Since D1 databases can only be accessed from within Cloudflare Workers,
    this backend communicates with a proxy Worker that handles the actual
    D1 database operations.
    """

    def __init__(
        self,
        capacity: int = 128,
        cache_name: str = "d1_cache",
        max_size_mb: float = 100.0,
        api_url: str = "",
        api_key: str = "",
        batch_size: int = 50,
        timeout: float = 30.0,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize D1 backend.

        Args:
            capacity: Maximum number of entries
            cache_name: Name for the cache (used as table prefix in D1)
            max_size_mb: Maximum cache size in megabytes
            api_url: URL of the D1 proxy Worker
            api_key: Bearer token for Worker authentication
            batch_size: Maximum number of operations to batch
            timeout: HTTP request timeout in seconds
            max_retries: Maximum number of retry attempts
            **kwargs: Additional configuration options
        """
        super().__init__(capacity, cache_name, max_size_mb, **kwargs)

        if not httpx:
            raise ImportError(
                "httpx is required for D1 backend. Install with: pip install httpx"
            )

        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = int(
            os.environ.get("STEADYTEXT_D1_BATCH_SIZE", str(batch_size))
        )
        self.timeout = timeout
        self.max_retries = max_retries

        # Create HTTP client with connection pooling
        self._client = httpx.Client(
            base_url=self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )

        # Initialize the cache table in D1
        self._initialize_table()

    def _initialize_table(self) -> None:
        """Initialize the cache table in D1."""
        try:
            response = self._client.post(
                API_INIT_ENDPOINT,
                json={"cache_name": self.cache_name, "max_size_mb": self.max_size_mb},
            )
            response.raise_for_status()
            logger.info(f"Initialized D1 cache table for {self.cache_name}")
        except Exception as e:
            logger.error(f"Failed to initialize D1 cache table: {e}")
            # Continue anyway - the table might already exist

    def _serialize_value(self, value: Any) -> str:
        """Serialize value to string for storage in D1."""
        # Use pickle and base64 for reliable serialization
        import base64

        pickled = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return base64.b64encode(pickled).decode("ascii")

    def _deserialize_value(self, data: str) -> Any:
        """Deserialize value from D1 storage."""
        import base64

        pickled = base64.b64decode(data.encode("ascii"))
        return pickle.loads(pickled)

    def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make HTTP request with retry logic."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                if method == "GET":
                    response = self._client.get(endpoint, **kwargs)
                elif method == "POST":
                    response = self._client.post(endpoint, json=json_data, **kwargs)
                elif method == "DELETE":
                    # AIDEV-NOTE: DELETE requests might need a body for batch operations
                    response = self._client.request(
                        "DELETE", endpoint, json=json_data, **kwargs
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                return response

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 0.5 * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                    continue

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    if attempt < self.max_retries - 1:
                        # Try to get retry-after header
                        retry_after = e.response.headers.get("Retry-After", "1")
                        wait_time = float(retry_after)
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        time.sleep(wait_time)
                        continue
                raise

        raise Exception(
            f"Request failed after {self.max_retries} attempts: {last_error}"
        )

    def get(self, key: Any) -> Optional[Any]:
        """Get value from D1 cache."""
        try:
            response = self._request_with_retry(
                "POST",
                API_GET_ENDPOINT,
                {"cache_name": self.cache_name, "key": str(key)},
            )

            data = response.json()
            if data.get("found") and "value" in data:
                return self._deserialize_value(data["value"])
            return None

        except Exception as e:
            logger.error(f"Failed to get key {key} from D1: {e}")
            return None

    def set(self, key: Any, value: Any) -> None:
        """Set value in D1 cache."""
        try:
            serialized_value = self._serialize_value(value)

            self._request_with_retry(
                "POST",
                API_SET_ENDPOINT,
                {
                    "cache_name": self.cache_name,
                    "key": str(key),
                    "value": serialized_value,
                    "size": len(serialized_value),
                },
            )

        except Exception as e:
            logger.error(f"Failed to set key {key} in D1: {e}")

    def delete(self, key: Any) -> None:
        """Delete a single entry from the D1 cache."""
        try:
            self._request_with_retry(
                "POST",
                API_DELETE_ENDPOINT,
                {"cache_name": self.cache_name, "key": str(key)},
            )
        except Exception as e:
            logger.error(f"Failed to delete key {key} from D1: {e}")

    def exists(self, key: Any) -> bool:
        """Check if a key exists in the D1 cache."""
        try:
            response = self._request_with_retry(
                "POST",
                API_EXISTS_ENDPOINT,
                {"cache_name": self.cache_name, "key": str(key)},
            )
            return response.json().get("exists", False)
        except Exception as e:
            logger.error(f"Failed to check existence of key {key} in D1: {e}")
            return False

    def clear(self) -> None:
        """Clear all entries from D1 cache."""
        try:
            self._request_with_retry(
                "POST", API_CLEAR_ENDPOINT, {"cache_name": self.cache_name}
            )
            logger.info(f"Cleared D1 cache {self.cache_name}")

        except Exception as e:
            logger.error(f"Failed to clear D1 cache: {e}")

    def sync(self) -> None:
        """Sync is a no-op for D1 backend (always persistent)."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get D1 cache statistics."""
        try:
            response = self._request_with_retry(
                "POST", API_STATS_ENDPOINT, {"cache_name": self.cache_name}
            )

            stats = response.json()
            stats["backend"] = "d1"
            stats["api_url"] = self.api_url
            return stats

        except Exception as e:
            logger.error(f"Failed to get D1 cache stats: {e}")
            return {
                "backend": "d1",
                "error": str(e),
                "api_url": self.api_url,
            }

    def __len__(self) -> int:
        """Return number of entries in D1 cache."""
        stats = self.get_stats()
        return stats.get("entry_count", 0)

    def batch_get(self, keys: List[Any]) -> Dict[Any, Any]:
        """Get multiple values from D1 cache in a single request."""
        if not keys:
            return {}
        try:
            # AIDEV-FIX: Create a map from stringified key back to original key
            key_map = {str(k): k for k in keys}
            string_keys = list(key_map.keys())
            results = {}

            for i in range(0, len(string_keys), self.batch_size):
                batch_str_keys = string_keys[i : i + self.batch_size]

                response = self._request_with_retry(
                    "POST",
                    API_BATCH_GET_ENDPOINT,
                    {"cache_name": self.cache_name, "keys": batch_str_keys},
                )

                batch_results = response.json().get("results", {})
                for str_key, data in batch_results.items():
                    if data.get("found") and "value" in data:
                        original_key = key_map[str_key]
                        results[original_key] = self._deserialize_value(data["value"])

            return results

        except Exception as e:
            logger.error(f"Failed to batch get from D1: {e}")
            return {}

    def batch_set(self, items: Dict[Any, Any]) -> None:
        """Set multiple values in D1 cache in a single request."""
        if not items:
            return
        try:
            items_list = list(items.items())

            for i in range(0, len(items_list), self.batch_size):
                batch_items = items_list[i : i + self.batch_size]

                batch_data = []
                for key, value in batch_items:
                    serialized_value = self._serialize_value(value)
                    batch_data.append(
                        {
                            "key": str(key),
                            "value": serialized_value,
                            "size": len(serialized_value),
                        }
                    )

                self._request_with_retry(
                    "POST",
                    API_BATCH_SET_ENDPOINT,
                    {"cache_name": self.cache_name, "items": batch_data},
                )

        except Exception as e:
            logger.error(f"Failed to batch set in D1: {e}")

    def batch_delete(self, keys: List[Any]) -> None:
        """Delete multiple entries from the D1 cache."""
        if not keys:
            return
        try:
            string_keys = [str(k) for k in keys]

            for i in range(0, len(string_keys), self.batch_size):
                batch_str_keys = string_keys[i : i + self.batch_size]

                self._request_with_retry(
                    "POST",  # Using POST to have a body in the request
                    API_BATCH_DELETE_ENDPOINT,
                    {"cache_name": self.cache_name, "keys": batch_str_keys},
                )

        except Exception as e:
            logger.error(f"Failed to batch delete from D1: {e}")

    def close(self) -> None:
        """Clean up HTTP client."""
        try:
            if not self._client.is_closed:
                self._client.close()
        except Exception:
            pass
