#!/usr/bin/env python3
"""
Direct test of daemon cache functionality.
"""

import os
import sys
from pathlib import Path

# Add steadytext to path
sys.path.insert(0, str(Path(__file__).parent))

# Ensure daemon is disabled for this test
os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"

from steadytext import generate
from steadytext.cache_manager import get_generation_cache, get_cache_manager
from steadytext.daemon.client import DaemonClient

def test_direct_vs_daemon_cache():
    """Test cache consistency between direct and daemon modes."""
    print("🧪 Testing cache functionality...")
    
    # Clear all caches
    print("📝 Clearing caches...")
    cache_manager = get_cache_manager()
    cache_manager.clear_all_caches()
    
    # Test prompt
    prompt = "What is Python?"
    
    # Step 1: Generate directly (daemon disabled)
    print("\n🔧 Testing direct generation...")
    result1 = generate(prompt)
    print(f"   Result: {result1[:50]}...")
    
    # Check cache
    cache = get_generation_cache()
    cached = cache.get(prompt)
    print(f"   Cached: {'✅' if cached == result1 else '❌ (cache miss)'}")
    
    # Step 2: Connect to daemon and test
    print("\n🔧 Testing daemon generation...")
    del os.environ["STEADYTEXT_DISABLE_DAEMON"]
    
    client = DaemonClient()
    if not client.connect():
        print("   ❌ Failed to connect to daemon")
        print("   Please start daemon with: st daemon start")
        return False
    
    try:
        # Generate via daemon (should use cache)
        result2 = client.generate(prompt)
        print(f"   Result: {result2[:50]}...")
        print(f"   Cache hit (same result): {'✅' if result2 == result1 else '❌'}")
        
        # Check cache stats
        stats = cache_manager.get_cache_stats()
        print(f"\n📊 Cache stats:")
        print(f"   Generation cache size: {stats['generation'].get('size', 'N/A')}")
        print(f"   Generation cache backend: {stats['generation'].get('backend', 'N/A')}")
        
        return result2 == result1
        
    finally:
        client.disconnect()

if __name__ == "__main__":
    success = test_direct_vs_daemon_cache()
    sys.exit(0 if success else 1)