"""
Pytest configuration for SteadyText tests.

AIDEV-NOTE: This file configures the test environment settings that apply to all tests. It disables the daemon by default to prevent slow test execution. Environment variables are set at the module level to run before any imports. The pytest_addoption hook is used to set environment variables before test collection and imports.
"""

import os


# AIDEV-NOTE: pytest_addoption runs before ANY test collection or imports
# This is the earliest hook we can use to set environment variables
def pytest_addoption(parser):
    """Add custom options and set environment variables BEFORE any imports."""
    # Set environment variables immediately, before any test modules are imported
    os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"
    # Only set ALLOW_MODEL_DOWNLOADS to false if not explicitly set to true
    if os.environ.get("STEADYTEXT_ALLOW_MODEL_DOWNLOADS", "").lower() != "true":
        os.environ["STEADYTEXT_ALLOW_MODEL_DOWNLOADS"] = "false"
        # Only skip model loading when downloads are not allowed
        os.environ["STEADYTEXT_SKIP_MODEL_LOAD"] = "1"
    os.environ["STEADYTEXT_DAEMON_FAILURE_CACHE_SECONDS"] = "1"
    os.environ["STEADYTEXT_DAEMON_TIMEOUT_MS"] = "50"
    os.environ["STEADYTEXT_SKIP_CACHE_INIT"] = "1"


def pytest_configure(config):
    """Configure pytest environment before tests run."""
    # AIDEV-NOTE: Re-set environment variables to be absolutely sure
    # This provides a second layer of protection
    os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"
    # Only set ALLOW_MODEL_DOWNLOADS to false if not explicitly set to true
    if os.environ.get("STEADYTEXT_ALLOW_MODEL_DOWNLOADS", "").lower() != "true":
        os.environ["STEADYTEXT_ALLOW_MODEL_DOWNLOADS"] = "false"
        # Only skip model loading when downloads are not allowed
        os.environ["STEADYTEXT_SKIP_MODEL_LOAD"] = "1"
    os.environ["STEADYTEXT_DAEMON_FAILURE_CACHE_SECONDS"] = "1"
    os.environ["STEADYTEXT_DAEMON_TIMEOUT_MS"] = "50"
    os.environ["STEADYTEXT_SKIP_CACHE_INIT"] = "1"
