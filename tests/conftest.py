"""
Pytest configuration for SteadyText tests.

AIDEV-NOTE: This file configures test environment settings that apply to all tests.
Key configuration: Disables daemon by default to prevent slow test execution.
"""

import os


def pytest_configure(config):
    """Configure pytest environment before tests run."""
    # AIDEV-NOTE: Disable daemon for all tests by default
    # This prevents connection timeouts that slow down test execution
    # Individual tests can re-enable if they specifically test daemon functionality
    os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"

    # AIDEV-NOTE: Set a short failure cache duration for tests
    # This ensures tests don't wait long between retries if needed
    os.environ["STEADYTEXT_DAEMON_FAILURE_CACHE_SECONDS"] = "1"

    # AIDEV-NOTE: Use even shorter timeout for tests (50ms)
    # This makes individual connection attempts fail faster
    os.environ["STEADYTEXT_DAEMON_TIMEOUT_MS"] = "50"
