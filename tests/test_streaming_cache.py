"""Test that streaming generation populates cache correctly."""

import os
import pytest
from steadytext import generate, generate_iter
from steadytext.daemon.client import DaemonClient
from steadytext.cache_manager import get_generation_cache, get_cache_manager
from tests.test_daemon_cache_integration import daemon_server_context


@pytest.fixture(autouse=True)
def enable_cache_for_tests(monkeypatch):
    """Temporarily enable cache initialization for these tests."""
    monkeypatch.delenv("STEADYTEXT_SKIP_CACHE_INIT", raising=False)


@pytest.mark.skipif(
    os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1",
    reason="Model loading is disabled, skipping streaming cache tests",
)
class TestStreamingCachePopulation:
    """Test that streaming generation populates cache in both modes."""

    def setup_method(self):
        """Clear caches before each test."""
        get_cache_manager().clear_all_caches()
        # Ensure daemon is disabled for direct tests
        os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"

    def teardown_method(self):
        """Cleanup after tests."""
        if "STEADYTEXT_DISABLE_DAEMON" in os.environ:
            del os.environ["STEADYTEXT_DISABLE_DAEMON"]

    def test_direct_streaming_populates_cache(self):
        """Test that direct mode streaming populates the cache."""
        # AIDEV-NOTE: This test verifies the new caching behavior in generate_iter

        test_prompt = "What is machine learning?"

        # Ensure cache is empty
        cache_key = test_prompt
        assert get_generation_cache().get(cache_key) is None

        # Stream generation (should populate cache)
        tokens = []
        for token in generate_iter(test_prompt):
            tokens.append(token)

        streamed_result = "".join(tokens)
        assert len(streamed_result) > 0

        # Verify cache was populated
        cached_result = get_generation_cache().get(cache_key)
        assert cached_result is not None
        assert cached_result == streamed_result

        # Verify subsequent streaming uses cache
        tokens2 = []
        for token in generate_iter(test_prompt):
            tokens2.append(token)

        streamed_result2 = "".join(tokens2)
        # Should be identical since it's from cache
        assert streamed_result2 == streamed_result

    def test_daemon_streaming_populates_cache(self):
        """Test that daemon mode streaming populates the cache."""
        # AIDEV-NOTE: This test verifies daemon streaming cache population

        test_prompt = "Explain quantum computing"

        # Ensure cache is empty
        cache_key = test_prompt
        assert get_generation_cache().get(cache_key) is None

        # Save original env values
        orig_disable_daemon = os.environ.get("STEADYTEXT_DISABLE_DAEMON")
        orig_daemon_host = os.environ.get("STEADYTEXT_DAEMON_HOST")
        orig_daemon_port = os.environ.get("STEADYTEXT_DAEMON_PORT")

        try:
            with daemon_server_context(preload_models=False) as (server, port):
                # Enable daemon mode
                if "STEADYTEXT_DISABLE_DAEMON" in os.environ:
                    del os.environ["STEADYTEXT_DISABLE_DAEMON"]
                os.environ["STEADYTEXT_DAEMON_HOST"] = "localhost"
                os.environ["STEADYTEXT_DAEMON_PORT"] = str(port)

                client = DaemonClient(host="localhost", port=port, timeout_ms=60000)
                assert client.connect()

                try:
                    # Stream via daemon (should populate cache)
                    tokens = []
                    for token in client.generate_iter(test_prompt):
                        tokens.append(token)

                    streamed_result = "".join(tokens)
                    assert len(streamed_result) > 0

                    # Verify cache was populated
                    cached_result = get_generation_cache().get(cache_key)
                    assert cached_result is not None
                    assert cached_result == streamed_result

                    # Stream again (should use cache)
                    tokens2 = []
                    for token in client.generate_iter(test_prompt):
                        tokens2.append(token)

                    streamed_result2 = "".join(tokens2)
                    # Should be identical
                    assert streamed_result2 == streamed_result

                finally:
                    client.disconnect()
        finally:
            # Restore original environment variables
            if orig_disable_daemon is not None:
                os.environ["STEADYTEXT_DISABLE_DAEMON"] = orig_disable_daemon
            elif "STEADYTEXT_DISABLE_DAEMON" not in os.environ:
                # Ensure it's set back to "1" for subsequent tests
                os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"

            if orig_daemon_host is not None:
                os.environ["STEADYTEXT_DAEMON_HOST"] = orig_daemon_host
            elif "STEADYTEXT_DAEMON_HOST" in os.environ:
                del os.environ["STEADYTEXT_DAEMON_HOST"]

            if orig_daemon_port is not None:
                os.environ["STEADYTEXT_DAEMON_PORT"] = orig_daemon_port
            elif "STEADYTEXT_DAEMON_PORT" in os.environ:
                del os.environ["STEADYTEXT_DAEMON_PORT"]

    def test_streaming_cache_with_custom_eos(self):
        """Test that streaming with custom EOS string uses separate cache entries."""
        test_prompt = "Tell me a story"

        # Stream with default EOS
        tokens1 = []
        for token in generate_iter(test_prompt):
            tokens1.append(token)
        result1 = "".join(tokens1)

        # Stream with custom EOS
        tokens2 = []
        for token in generate_iter(test_prompt, eos_string="END"):
            tokens2.append(token)
        result2 = "".join(tokens2)

        # Results might be different due to different EOS
        # But each should be cached separately
        cache_key1 = test_prompt
        cache_key2 = f"{test_prompt}::EOS::END"

        assert get_generation_cache().get(cache_key1) == result1
        assert get_generation_cache().get(cache_key2) == result2

    def test_streaming_no_cache_for_logprobs(self):
        """Test that streaming with logprobs doesn't populate cache."""
        test_prompt = "Explain gravity"

        # Ensure cache is empty
        cache_key = test_prompt
        assert get_generation_cache().get(cache_key) is None

        # Stream with logprobs
        tokens = []
        for token in generate_iter(test_prompt, include_logprobs=True):
            # Token should be a dict with logprobs
            assert isinstance(token, dict)
            assert "token" in token
            tokens.append(token.get("token", ""))

        result = "".join(tokens)
        assert len(result) > 0

        # Verify cache was NOT populated (logprobs requests don't cache)
        assert get_generation_cache().get(cache_key) is None

    def test_streaming_cache_consistency_with_non_streaming(self):
        """Test that streaming and non-streaming produce consistent cached results."""
        test_prompt = "What is artificial intelligence?"

        # Clear cache
        get_cache_manager().clear_all_caches()

        # Generate with non-streaming (populates cache)
        non_streaming_result = generate(test_prompt)

        # Clear cache again
        get_cache_manager().clear_all_caches()

        # Generate with streaming (populates cache)
        streaming_tokens = []
        for token in generate_iter(test_prompt):
            streaming_tokens.append(token)
        streaming_result = "".join(streaming_tokens)

        # Both should produce the same result (deterministic)
        # With v2.1.0+, both may return None/empty when model is not loaded
        if non_streaming_result is None:
            assert streaming_result == ""
        else:
            assert streaming_result == non_streaming_result
            # Cache should contain the same result
            cached_result = get_generation_cache().get(test_prompt)
            assert cached_result == non_streaming_result
