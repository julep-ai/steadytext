"""
Pytest configuration for SteadyText tests.

AIDEV-NOTE: This file configures test environment settings that apply to all tests.
Key configuration: Disables daemon by default to prevent slow test execution.
AIDEV-NOTE: Environment variables are set at module level to run before any imports
AIDEV-NOTE: Fixed pytest hanging issue (v2.0.1+) - Environment variables must be set
in pytest_addoption hook which runs BEFORE test collection and imports. Without this,
pytest --collect-only --noconftest would hang due to module-level code execution.
"""

import os


# AIDEV-NOTE: pytest_addoption runs before ANY test collection or imports
# This is the earliest hook we can use to set environment variables
def pytest_addoption(parser):
    """Add custom options and set environment variables BEFORE any imports."""
    # Set environment variables immediately, before any test modules are imported
    os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"
    os.environ["STEADYTEXT_ALLOW_MODEL_DOWNLOADS"] = "false"
    os.environ["STEADYTEXT_SKIP_MODEL_LOAD"] = "1"
    os.environ["STEADYTEXT_DAEMON_FAILURE_CACHE_SECONDS"] = "1"
    os.environ["STEADYTEXT_DAEMON_TIMEOUT_MS"] = "50"
    os.environ["STEADYTEXT_SKIP_CACHE_INIT"] = "1"


def pytest_configure(config):
    """Configure pytest environment before tests run."""
    # AIDEV-NOTE: Re-set environment variables to be absolutely sure
    # This provides a second layer of protection
    os.environ["STEADYTEXT_DISABLE_DAEMON"] = "1"
    os.environ["STEADYTEXT_ALLOW_MODEL_DOWNLOADS"] = "false"
    os.environ["STEADYTEXT_SKIP_MODEL_LOAD"] = "1"
    os.environ["STEADYTEXT_DAEMON_FAILURE_CACHE_SECONDS"] = "1"
    os.environ["STEADYTEXT_DAEMON_TIMEOUT_MS"] = "50"
    os.environ["STEADYTEXT_SKIP_CACHE_INIT"] = "1"
