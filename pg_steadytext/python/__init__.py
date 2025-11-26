"""
pg_steadytext - PostgreSQL extension Python modules
AIDEV-NOTE: This package provides Python functionality for the pg_steadytext PostgreSQL extension
"""

from .security import (
    raise_sqlstate,
    sanitize_cache_key,
    validate_host,
    validate_port,
    validate_table_name,
)
from .prompt_registry import validate_template, render_template, extract_variables

__version__ = "2025.11.25"
__all__ = [
    "raise_sqlstate",
    "sanitize_cache_key",
    "validate_host",
    "validate_port",
    "validate_table_name",
    "validate_template",
    "render_template",
    "extract_variables",
]
