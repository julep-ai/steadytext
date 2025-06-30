"""
Test daemon cache integration with centralized cache system.

AIDEV-NOTE: These tests ensure cache consistency between the daemon and direct access modes.
"""

import os
import time
import pytest
import threading
import socket
from contextlib import contextmanager
import numpy as np

# Import our modules
from steadytext import generate, generate_iter
from steadytext.cache_manager import (
    get_cache_manager,
    get_generation_cache,
    get_embedding_cache,
)
from steadytext.daemon.server import DaemonServer
from steadytext.daemon.client import DaemonClient

# AIDEV-NOTE: Skip daemon tests only if explicitly disabled
# Allow model downloads for daemon tests since they require real models
pytestmark = pytest.mark.skipif(
    os.environ.get("STEADYTEXT_DISABLE_DAEMON_TESTS", "0") == "1"
    or os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD", "0") == "1",
    reason="Daemon tests explicitly disabled or model loading is disabled",
)


@pytest.fixture(autouse=True)
def enable_cache_for_tests(monkeypatch):
    """Temporarily enable cache initialization for these tests."""
    monkeypatch.delenv("STEADYTEXT_SKIP_CACHE_INIT", raising=False)


def find_free_port():
    """Find a free port for daemon testing."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@contextmanager
def daemon_server_context(preload_models=False):
    """Context manager for daemon server lifecycle in tests."""
    port = find_free_port()
    server = DaemonServer(host="localhost", port=port, preload_models=preload_models)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Give server time to start with exponential backoff and better verification
    max_wait = 60  # Increased to 60 seconds for model loading
    wait_time = 0.2  # Start with shorter wait
    total_wait = 0
    connected = False

    while total_wait < max_wait:
        try:
            # Use higher timeout for initial connection attempt
            client = DaemonClient(host="localhost", port=port, timeout_ms=5000)
            if client.connect():
                # Verify daemon is actually responsive with a ping
                if client.ping():
                    client.disconnect()
                    connected = True
                    break
                else:
                    client.disconnect()
        except Exception as e:
            # Log connection attempts for debugging
            if total_wait > 10:  # Only log after 10 seconds to reduce noise
                print(f"Daemon connection attempt failed after {total_wait:.1f}s: {e}")

        time.sleep(wait_time)
        total_wait += wait_time
        wait_time = min(wait_time * 1.2, 1.0)  # Gentler exponential backoff, max 1s

    if not connected:
        server.running = False
        time.sleep(1)
        raise RuntimeError(f"Failed to connect to daemon after {max_wait} seconds")

    try:
        yield server, port
    finally:
        server.running = False
        # Give server more time to shut down properly
        time.sleep(2)


class TestDaemonCacheIntegration:
    """Test cache consistency between daemon and direct access modes."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup clean test environment before each test."""
        # Clear all caches before each test
        cache_manager = get_cache_manager()
        cache_manager.clear_all_caches()

        # Ensure daemon is disabled for direct access tests
        old_disable_daemon = os.environ.get("STEADYTEXT_DISABLE_DAEMON")
        os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"

        yield

        # Restore original daemon setting
        if old_disable_daemon is not None:
            os.environ["STEADYTEXT_DISABLE_DAEMON"] = old_disable_daemon
        elif "STEADYTEXT_DISABLE_DAEMON" in os.environ:
            del os.environ["STEADYTEXT_DISABLE_DAEMON"]

    def test_generation_cache_consistency_direct_vs_daemon(self):
        """Test that direct access and daemon produce identical cached results."""
        # AIDEV-NOTE: This test verifies cache consistency between access modes

        test_prompt = "What is the capital of France?"

        # Step 1: Generate via direct access (should cache the result)
        os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"
        direct_result = generate(test_prompt)
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
            assert direct_result is None
            return  # Cant test consistency if no model
        assert isinstance(direct_result, str)
        assert len(direct_result) > 0

        # Verify result is now cached
        cache_key = test_prompt
        cached_result = get_generation_cache().get(cache_key)
        assert cached_result is not None
        assert cached_result == direct_result

        # Step 2: Test via daemon using context manager
        with daemon_server_context(preload_models=False) as (server, port):
            # Step 3: Generate via daemon (should use cached result)
            if "STEADYTEXT_DISABLE_DAEMON" in os.environ:
                del os.environ["STEADYTEXT_DISABLE_DAEMON"]
            os.environ["STEADYTEXT_DAEMON_HOST"] = "localhost"
            os.environ["STEADYTEXT_DAEMON_PORT"] = str(port)

            # Use longer timeout for model loading and generation
            client = DaemonClient(host="localhost", port=port, timeout_ms=60000)
            assert client.connect(), "Failed to connect to test daemon"

            try:
                daemon_result = client.generate(test_prompt)
                assert isinstance(daemon_result, str)
                assert daemon_result == direct_result, (
                    "Daemon result doesn't match cached direct result"
                )

                # Step 4: Verify cache was used by daemon
                # The cached result should still be there and identical
                final_cached = get_generation_cache().get(cache_key)
                assert final_cached == direct_result

            finally:
                client.disconnect()

        # Cleanup environment
        os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"

    def test_streaming_cache_consistency(self):
        """Test that streaming generation uses cache when available."""
        # AIDEV-NOTE: Verify streaming respects cache and simulates streaming from cached results

        test_prompt = "Tell me about artificial intelligence"

        # Step 1: Generate normally to populate cache
        os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"
        direct_result = generate(test_prompt)
        if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
            assert direct_result is None
            return  # Cant test consistency if no model
        assert isinstance(direct_result, str)

        # Step 2: Test streaming from cache (direct mode)
        direct_stream_tokens = []
        for token in generate_iter(test_prompt):
            direct_stream_tokens.append(token)

        "".join(direct_stream_tokens)
        # Note: Direct streaming might not use cache, but daemon should

        # Step 3: Start daemon and test streaming cache usage
        with daemon_server_context(preload_models=False) as (server, port):
            if "STEADYTEXT_DISABLE_DAEMON" in os.environ:
                del os.environ["STEADYTEXT_DISABLE_DAEMON"]
            os.environ["STEADYTEXT_DAEMON_HOST"] = "localhost"
            os.environ["STEADYTEXT_DAEMON_PORT"] = str(port)

            # Use longer timeout for model loading and generation
            client = DaemonClient(host="localhost", port=port, timeout_ms=60000)
            assert client.connect(), "Failed to connect to daemon"

            try:
                # Stream via daemon - should use cached result
                daemon_stream_tokens = []
                for token in client.generate_iter(test_prompt):
                    daemon_stream_tokens.append(token)

                daemon_stream_result = "".join(daemon_stream_tokens)

                # The daemon streaming should produce the same result as the cached direct result
                # Note: streaming from cache splits by words and rejoins with spaces, so we normalize whitespace
                import re

                def normalize_ws(s):
                    return re.sub(r"\s+", " ", s.strip())

                assert normalize_ws(daemon_stream_result) == normalize_ws(direct_result)

            finally:
                client.disconnect()

        # Cleanup environment
        os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"

    import numpy as np

    def test_embedding_cache_consistency(self):
        """Test embedding cache consistency between daemon and direct modes."""
        # AIDEV-NOTE: This test is modified to not depend on model loading.
        # It manually populates the cache and verifies the daemon reads from it.

        test_text = "This is a test sentence for embedding consistency"
        cache_key = (test_text,)  # Embedder uses tuple cache keys

        # Step 1: Manually create a fake embedding and populate the cache
        fake_embedding = np.random.rand(1024).astype(np.float32)
        embedding_cache = get_embedding_cache()
        embedding_cache.set(cache_key, fake_embedding)

        # Verify it's in the cache
        cached_embedding = embedding_cache.get(cache_key)
        assert cached_embedding is not None
        assert np.array_equal(cached_embedding, fake_embedding)

        # Step 2: Start daemon and test embedding via daemon
        with daemon_server_context(preload_models=False) as (server, port):
            if "STEADYTEXT_DISABLE_DAEMON" in os.environ:
                del os.environ["STEADYTEXT_DISABLE_DAEMON"]
            os.environ["STEADYTEXT_DAEMON_HOST"] = "localhost"
            os.environ["STEADYTEXT_DAEMON_PORT"] = str(port)

            client = DaemonClient(host="localhost", port=port, timeout_ms=60000)
            assert client.connect(), "Failed to connect to daemon"

            try:
                # Ask daemon for the embedding
                daemon_embedding = client.embed(test_text)
                assert daemon_embedding is not None

                # The daemon should have retrieved the fake embedding from the cache
                assert np.allclose(daemon_embedding, fake_embedding, atol=1e-6)

            finally:
                client.disconnect()

        # Cleanup environment
        os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"

    def test_cache_miss_and_population_via_daemon(self):
        """Test that daemon properly populates cache on cache miss."""
        # AIDEV-NOTE: When cache is empty, daemon should generate and cache results

        test_prompt = "Generate a unique response for cache testing"

        # Ensure cache is empty
        cache_manager = get_cache_manager()
        cache_manager.clear_all_caches()

        cache_key = test_prompt
        assert get_generation_cache().get(cache_key) is None

        # Start daemon
        with daemon_server_context(preload_models=False) as (server, port):
            if "STEADYTEXT_DISABLE_DAEMON" in os.environ:
                del os.environ["STEADYTEXT_DISABLE_DAEMON"]
            os.environ["STEADYTEXT_DAEMON_HOST"] = "localhost"
            os.environ["STEADYTEXT_DAEMON_PORT"] = str(port)

            # Use longer timeout for model loading and generation
            client = DaemonClient(host="localhost", port=port, timeout_ms=60000)
            assert client.connect(), "Failed to connect to daemon"

            try:
                # Generate via daemon (cache miss)
                daemon_result = client.generate(test_prompt)
                if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
                    assert daemon_result is None
                    return
                assert isinstance(daemon_result, str)
                assert len(daemon_result) > 0

                # Verify result was cached by daemon
                cached_result = get_generation_cache().get(cache_key)
                assert cached_result is not None
                assert cached_result == daemon_result

                # Second call should hit cache
                daemon_result_2 = client.generate(test_prompt)
                assert daemon_result_2 == daemon_result

            finally:
                client.disconnect()

        # Cleanup environment
        os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"

    def test_custom_eos_string_cache_key(self):
        """Test that custom eos_string is properly handled in cache keys."""
        # AIDEV-NOTE: Cache keys must include eos_string to avoid collisions

        test_prompt = "Count to three"
        custom_eos = "[STOP]"

        # Clear cache
        get_cache_manager().clear_all_caches()

        # Start daemon
        with daemon_server_context(preload_models=False) as (server, port):
            if "STEADYTEXT_DISABLE_DAEMON" in os.environ:
                del os.environ["STEADYTEXT_DISABLE_DAEMON"]
            os.environ["STEADYTEXT_DAEMON_HOST"] = "localhost"
            os.environ["STEADYTEXT_DAEMON_PORT"] = str(port)

            # Use longer timeout for model loading and generation
            client = DaemonClient(host="localhost", port=port, timeout_ms=60000)
            assert client.connect(), "Failed to connect to daemon"

            try:
                # Generate with default eos_string
                result_default = client.generate(test_prompt)
                if os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1":
                    assert result_default is None
                    return

                # Generate with custom eos_string
                result_custom = client.generate(test_prompt, eos_string=custom_eos)

                # Both should be cached with different keys
                cache_key_default = test_prompt
                cache_key_custom = f"{test_prompt}::EOS::{custom_eos}"

                cached_default = get_generation_cache().get(cache_key_default)
                cached_custom = get_generation_cache().get(cache_key_custom)

                assert cached_default is not None
                assert cached_custom is not None
                assert cached_default == result_default
                assert cached_custom == result_custom

                # Results might be different due to different stop sequences
                # but they should be consistently cached

            finally:
                client.disconnect()

        # Cleanup environment
        os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"

    def test_no_cache_for_logprobs_requests(self):
        """Test that logprobs requests are not cached (by design)."""
        # AIDEV-NOTE: Logprobs requests should bypass cache to ensure fresh token probabilities

        test_prompt = "Simple test prompt"

        # Clear cache
        get_cache_manager().clear_all_caches()

        # Start daemon
        with daemon_server_context(preload_models=False) as (server, port):
            if "STEADYTEXT_DISABLE_DAEMON" in os.environ:
                del os.environ["STEADYTEXT_DISABLE_DAEMON"]
            os.environ["STEADYTEXT_DAEMON_HOST"] = "localhost"
            os.environ["STEADYTEXT_DAEMON_PORT"] = str(port)

            # Use longer timeout for logprobs tests as they may take longer
            client = DaemonClient(host="localhost", port=port, timeout_ms=60000)
            assert client.connect(), "Failed to connect to daemon"

            try:
                # Generate with logprobs
                result = client.generate(test_prompt, return_logprobs=True)
                assert isinstance(result, dict)
                assert "text" in result

                # Cache should remain empty for logprobs requests
                cache_key = test_prompt
                cached_result = get_generation_cache().get(cache_key)
                assert cached_result is None, "Logprobs requests should not be cached"

            finally:
                client.disconnect()

        # Cleanup environment
        os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
