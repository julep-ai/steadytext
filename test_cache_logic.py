#!/usr/bin/env python3
"""
Test cache logic in daemon server methods directly.

This tests the cache integration logic without needing to run a full daemon.
"""

import os
import sys
from pathlib import Path

# Add steadytext to path
sys.path.insert(0, str(Path(__file__).parent))

from steadytext.cache_manager import get_generation_cache, get_cache_manager
from steadytext.daemon.server import DaemonServer


def test_daemon_cache_logic():
    """Test the cache logic in daemon server methods directly."""
    print("🧪 Testing daemon server cache logic...")
    
    # Clear cache
    cache_manager = get_cache_manager()
    cache_manager.clear_all_caches()
    
    # Create daemon server instance
    server = DaemonServer(preload_models=False)
    
    # Test generation cache logic
    test_prompt = "What is Python?"
    test_params = {
        "prompt": test_prompt,
        "return_logprobs": False,
        "eos_string": "[EOS]",
        "model": None,
        "model_repo": None,  
        "model_filename": None,
        "size": None
    }
    
    print("🔧 Testing cache miss (should generate new result)...")
    
    # First call should be a cache miss and generate content
    result1 = server._handle_generate(test_params)
    print(f"   First result: {result1[:50] if isinstance(result1, str) else str(result1)[:50]}...")
    
    # Verify result was cached
    cache = get_generation_cache()
    cached_result = cache.get(test_prompt)
    cache_populated = cached_result is not None
    print(f"   Cache populated: {'✅' if cache_populated else '❌'}")
    
    if cache_populated:
        print(f"   Cached result matches: {'✅' if cached_result == result1 else '❌'}")
    
    print("🔧 Testing cache hit (should return cached result)...")
    
    # Second call should be a cache hit
    result2 = server._handle_generate(test_params)
    cache_hit = result2 == result1
    print(f"   Second result matches first: {'✅' if cache_hit else '❌'}")
    
    # Test custom eos_string caching
    print("🔧 Testing custom EOS string cache logic...")
    
    custom_eos_params = dict(test_params)
    custom_eos_params["eos_string"] = "[STOP]"
    
    result3 = server._handle_generate(custom_eos_params)
    
    # Check that custom eos result uses different cache key
    custom_cache_key = f"{test_prompt}::EOS::[STOP]"
    cached_custom = cache.get(custom_cache_key)
    custom_cached = cached_custom is not None and cached_custom == result3
    print(f"   Custom EOS cached separately: {'✅' if custom_cached else '❌'}")
    
    # Test logprobs doesn't use cache
    print("🔧 Testing logprobs requests bypass cache...")
    
    logprobs_params = dict(test_params)
    logprobs_params["return_logprobs"] = True
    
    result4 = server._handle_generate(logprobs_params)
    # Logprobs should not be cached, so cache for original key should be unchanged
    final_cached = cache.get(test_prompt)
    logprobs_no_cache = final_cached == result1  # Original cache unchanged
    print(f"   Logprobs didn't modify cache: {'✅' if logprobs_no_cache else '❌'}")
    
    all_passed = cache_populated and cache_hit and custom_cached and logprobs_no_cache
    
    if all_passed:
        print("🎉 All daemon cache logic tests passed!")
    else:
        print("❌ Some daemon cache logic tests failed.")
    
    return all_passed


if __name__ == "__main__":
    success = test_daemon_cache_logic()
    sys.exit(0 if success else 1)