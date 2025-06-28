"""
pg_steadytext - PostgreSQL extension Python modules
AIDEV-NOTE: This package provides Python functionality for the pg_steadytext PostgreSQL extension
"""

from .daemon_connector import SteadyTextDaemonClient
from .cache_manager import CacheManager, FrecencyCache
from .security import InputValidator, RateLimiter
from .config import ConfigManager

__version__ = "1.0.0"
__all__ = [
    "SteadyTextDaemonClient",
    "CacheManager",
    "FrecencyCache",
    "InputValidator",
    "RateLimiter",
    "ConfigManager",
]

# AIDEV-NOTE: Ensure all modules can be imported from PostgreSQL's plpython3u environment
