#!/usr/bin/env python3
"""
Demo of SteadyText daemon functionality.

This script demonstrates:
1. Starting daemon programmatically
2. Using daemon for generation and embedding
3. Performance comparison with/without daemon
"""

import time
import os
import steadytext as st


def demo_without_daemon():
    """Time operations without daemon."""
    print("Testing WITHOUT daemon...")

    # First call loads model
    start = time.time()
    st.generate("Hello world", size="small")
    time1 = time.time() - start
    print(f"  First generation: {time1:.2f}s")

    # Second call reuses loaded model
    start = time.time()
    st.generate("Hello again", size="small")
    time2 = time.time() - start
    print(f"  Second generation: {time2:.2f}s")

    # Embedding
    start = time.time()
    st.embed("Test text")
    time3 = time.time() - start
    print(f"  Embedding: {time3:.2f}s")

    return time1 + time2 + time3


def demo_with_daemon():
    """Time operations with daemon."""
    print("\nTesting WITH daemon...")

    # Enable daemon usage (daemon is enabled by default in v1.3+)
    # Remove DISABLE flag if it exists
    daemon_was_disabled = os.environ.get("STEADYTEXT_DISABLE_DAEMON") == "1"
    if daemon_was_disabled:
        del os.environ["STEADYTEXT_DISABLE_DAEMON"]

    try:
        # First call via daemon
        start = time.time()
        st.generate("Hello world", size="small")
        time1 = time.time() - start
        print(f"  First generation: {time1:.2f}s")

        # Second call via daemon
        start = time.time()
        st.generate("Hello again", size="small")
        time2 = time.time() - start
        print(f"  Second generation: {time2:.2f}s")

        # Embedding via daemon
        start = time.time()
        st.embed("Test text")
        time3 = time.time() - start
        print(f"  Embedding: {time3:.2f}s")

        return time1 + time2 + time3
    finally:
        # Restore original state if daemon was disabled
        if daemon_was_disabled:
            os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"


def main():
    print("SteadyText Daemon Demo")
    print("=" * 50)

    # Note: Daemon should be started separately with:
    # st daemon start

    print("\nMake sure daemon is running with: st daemon start")
    print("Check status with: st daemon status")

    # Test without daemon
    time_without = demo_without_daemon()

    # Test with daemon
    time_with = demo_with_daemon()

    print("\n" + "=" * 50)
    print(f"Total time WITHOUT daemon: {time_without:.2f}s")
    print(f"Total time WITH daemon: {time_with:.2f}s")
    print(f"Speedup: {time_without / time_with:.2f}x")

    # Demo context manager
    print("\nUsing context manager:")
    with st.use_daemon():
        text = st.generate("Context manager test")
        if text is not None:
            print(f"  Generated {len(text)} characters")
        else:
            print("  Generation failed")


if __name__ == "__main__":
    main()
