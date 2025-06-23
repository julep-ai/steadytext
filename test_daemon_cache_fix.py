#!/usr/bin/env python3
"""
Demonstration script showing that daemon cache integration is working.

This script shows that:
1. Cache works in direct mode
2. Cache works in daemon mode  
3. Results are consistent between modes
4. Cache hits occur properly

Run with: python test_daemon_cache_fix.py
"""

import os
import sys
import tempfile
import time
import threading
from pathlib import Path

# Add steadytext to path
sys.path.insert(0, str(Path(__file__).parent))

from steadytext import generate
from steadytext.cache_manager import get_generation_cache, get_cache_manager
from steadytext.daemon.server import DaemonServer
from steadytext.daemon.client import DaemonClient


def test_cache_integration():
    """Test that cache works consistently between daemon and direct modes."""
    print("🧪 Testing SteadyText daemon cache integration...")
    
    # Clear all caches first
    print("📝 Clearing all caches...")
    cache_manager = get_cache_manager()
    cache_manager.clear_all_caches()
    
    test_prompt = "What is 2+2?"
    
    # Step 1: Test direct mode caching
    print("🔧 Testing direct mode (daemon disabled)...")
    os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"
    
    direct_result = generate(test_prompt)
    print(f"   Direct result: {direct_result[:50]}...")
    
    # Verify result was cached
    cache = get_generation_cache()
    cached_result = cache.get(test_prompt)
    cache_hit = cached_result is not None and cached_result == direct_result
    print(f"   Cache populated: {'✅' if cache_hit else '❌'}")
    
    # Step 2: Test daemon mode cache usage
    print("🔧 Testing daemon mode...")
    
    # Start daemon server
    server = DaemonServer(host="localhost", port=5555, preload_models=False)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    time.sleep(3)  # Give server time to start
    
    try:
        # Enable daemon mode
        if "STEADYTEXT_DISABLE_DAEMON" in os.environ:
            del os.environ["STEADYTEXT_DISABLE_DAEMON"]
        os.environ["STEADYTEXT_DAEMON_HOST"] = "localhost"
        os.environ["STEADYTEXT_DAEMON_PORT"] = "5555"
        
        # Connect to daemon
        client = DaemonClient(host="localhost", port=5555)
        connected = client.connect()
        
        if not connected:
            print("   ❌ Failed to connect to daemon")
            return False
            
        print("   ✅ Connected to daemon")
        
        # Generate via daemon (should use cache)
        daemon_result = client.generate(test_prompt)
        print(f"   Daemon result: {daemon_result[:50]}...")
        
        # Check if results match (indicating cache was used)
        results_match = daemon_result == direct_result
        print(f"   Results match: {'✅' if results_match else '❌'}")
        
        # Test a new prompt via daemon to verify caching works
        new_prompt = "What is the capital of France?"
        daemon_result_2 = client.generate(new_prompt)
        print(f"   New prompt result: {daemon_result_2[:50]}...")
        
        # Verify this new result was cached
        cached_result_2 = cache.get(new_prompt)
        new_cached = cached_result_2 is not None and cached_result_2 == daemon_result_2
        print(f"   New result cached: {'✅' if new_cached else '❌'}")
        
        client.disconnect()
        
        # Step 3: Test that direct mode can use daemon-cached results
        print("🔧 Testing direct mode with daemon-cached results...")
        os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"
        
        direct_result_2 = generate(new_prompt)
        cached_consistency = direct_result_2 == daemon_result_2
        print(f"   Direct mode uses daemon cache: {'✅' if cached_consistency else '❌'}")
        
        success = cache_hit and connected and results_match and new_cached and cached_consistency
        
        if success:
            print("🎉 All cache integration tests passed! Daemon cache is working properly.")
        else:
            print("❌ Some cache integration tests failed.")
            
        return success
        
    finally:
        # Cleanup
        server.running = False
        os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"


if __name__ == "__main__":
    success = test_cache_integration()
    sys.exit(0 if success else 1)