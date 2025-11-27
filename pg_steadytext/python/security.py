"""
security.py - Minimal utilities for pg_steadytext

This module provides only essential utilities for PostgreSQL integration.
Input validation for prompts/text is the application's responsibility.
"""

import re
import hashlib
from typing import Any, Optional, Dict

IN_POSTGRES = False
try:
    import plpy  # type: ignore[unresolved-import]

    IN_POSTGRES = True
except ImportError:
    pass

_RAISE_PLANS: Dict[str, Any] = {}
_EXT_SCHEMA: Optional[str] = None


def _get_extension_schema() -> str:
    """
    Get the schema where pg_steadytext extension is installed.

    Returns:
        str: The schema name (defaults to 'public' if not found)
    """
    global _EXT_SCHEMA
    if _EXT_SCHEMA is not None:
        return _EXT_SCHEMA
    result = plpy.execute("""
        SELECT nspname FROM pg_extension e
        JOIN pg_namespace n ON e.extnamespace = n.oid
        WHERE e.extname = 'pg_steadytext'
    """)
    _EXT_SCHEMA = result[0]["nspname"] if result else "public"
    return _EXT_SCHEMA


def raise_sqlstate(message: str, sqlstate: str = "P0001") -> None:
    """
    Raise a PostgreSQL error with a specific SQLSTATE code.

    Args:
        message: The error message to display
        sqlstate: The SQLSTATE error code (default: P0001 for raise_exception)

    Raises:
        ValueError: If called outside PostgreSQL context
    """
    if not IN_POSTGRES:
        raise ValueError(message)
    if sqlstate != "P0001":
        plpy.error(message)
        return
    plan_key = f"raise_plan_{sqlstate}"
    plan = _RAISE_PLANS.get(plan_key)
    if plan is None:
        ext_schema = _get_extension_schema()
        plan = plpy.prepare(
            f"SELECT {plpy.quote_ident(ext_schema)}._steadytext_raise_p0001($1)",
            ["text"],
        )
        _RAISE_PLANS[plan_key] = plan
    plpy.execute(plan, [message])


def sanitize_cache_key(key: str) -> str:
    """
    Sanitize a cache key to ensure it's safe for use as a cache identifier.

    If the key is already an MD5 hash (32 hex chars), returns it unchanged.
    Otherwise, returns the MD5 hash of the key.

    Args:
        key: The cache key to sanitize

    Returns:
        str: A sanitized 32-character MD5 hash
    """
    if re.match(r"^[a-f0-9]{32}$", key):
        return key
    return hashlib.md5(key.encode()).hexdigest()


def validate_host(host: str) -> tuple[bool, Optional[str]]:
    """
    Validate a hostname string for safety.

    Args:
        host: The hostname to validate

    Returns:
        tuple: (is_valid, error_message) - error_message is None if valid
    """
    if (
        host is None
        or not isinstance(host, str)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", host.strip())
    ):
        return False, "Invalid host format"
    return True, None


def validate_port(port_value: str) -> tuple[bool, Optional[str]]:
    """
    Validate a port number string.

    Args:
        port_value: The port value to validate (as string)

    Returns:
        tuple: (is_valid, error_message) - error_message is None if valid
    """
    if port_value is None:
        return False, "Invalid port format"
    try:
        port_int = int(str(port_value).strip())
    except (TypeError, ValueError):
        return False, "Invalid port format"
    if port_int < 1 or port_int > 65535:
        return False, "Invalid port format"
    return True, None


def validate_table_name(name: str) -> tuple[bool, Optional[str]]:
    """
    Validate a PostgreSQL table name for safety.

    Args:
        name: The table name to validate

    Returns:
        tuple: (is_valid, error_message) - error_message is None if valid
    """
    if (
        name is None
        or not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name.strip())
    ):
        return False, "Invalid table name"
    return True, None


__all__ = [
    "raise_sqlstate",
    "sanitize_cache_key",
    "validate_host",
    "validate_port",
    "validate_table_name",
    "_get_extension_schema",
]
