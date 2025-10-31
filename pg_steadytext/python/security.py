# AIDEV-SECTION: SECURITY
"""
security.py - Security and input validation for pg_steadytext

AIDEV-NOTE: This module provides input validation, sanitization, and security
features for the PostgreSQL extension to prevent injection attacks and abuse.
"""

import re
import hashlib
import logging
from typing import Optional, Dict, Any

# AIDEV-NOTE: PostgreSQL-specific imports
IN_POSTGRES = False
try:
    import plpy  # type: ignore[unresolved-import]

    IN_POSTGRES = True
except ImportError:
    # Not running inside PostgreSQL
    pass

# Configure logging
logger = logging.getLogger(__name__)

# AIDEV-NOTE: Security constants - adjust based on your requirements
MAX_PROMPT_LENGTH = 8000  # Maximum characters in a prompt before standard rejection
PROMPT_MEMORY_THRESHOLD = 200000  # Threshold for treating requests as memory exhaustion
MAX_EMBEDDING_TEXT_LENGTH = 10000  # Maximum characters for embedding requests
EMBEDDING_BUFFER_THRESHOLD = 500000  # Threshold for buffer overflow style errors
MAX_TOKENS_LIMIT = 4096  # Maximum tokens that can be requested
MIN_TOKENS_LIMIT = 1  # Minimum tokens
MAX_BATCH_SIZE = 100  # Maximum items in batch operations

# Cached raise plans per sqlstate (only used inside PostgreSQL)
_RAISE_PLANS: Dict[str, Any] = {}
_EXT_SCHEMA: Optional[str] = None


def _get_extension_schema() -> str:
    """Return the schema where pg_steadytext is installed."""

    global _EXT_SCHEMA
    if _EXT_SCHEMA is not None:
        return _EXT_SCHEMA

    result = plpy.execute(
        """
        SELECT nspname
        FROM pg_extension e
        JOIN pg_namespace n ON e.extnamespace = n.oid
        WHERE e.extname = 'pg_steadytext'
        """,
    )
    _EXT_SCHEMA = result[0]["nspname"] if result else "public"
    return _EXT_SCHEMA


def raise_sqlstate(message: str, sqlstate: str = "P0001") -> None:
    """Raise a PostgreSQL error with the given SQLSTATE."""

    if not IN_POSTGRES:
        # Outside PostgreSQL we can't use plpy, so raise ValueError for visibility
        raise ValueError(message)

    # Currently only P0001 is used in pg_steadytext; fall back to plpy.error for others
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


# AIDEV-NOTE: Removed unused SAFE_TEXT_PATTERN regex. The validate_prompt method
# uses a more nuanced approach that logs dangerous patterns but doesn't restrict
# legitimate special characters that users might need in their prompts.


class SecurityValidator:
    """
    Validates and sanitizes inputs for pg_steadytext.

    AIDEV-NOTE: This class provides methods to validate various inputs
    to prevent security issues like SQL injection, command injection,
    and resource exhaustion attacks.
    """

    @staticmethod
    def validate_prompt(prompt: str) -> tuple[bool, Optional[str]]:
        """
        Validate a text prompt for safety and length.

        AIDEV-NOTE: Checks prompt length and content to prevent abuse.
        Does NOT restrict special characters as they may be needed for
        legitimate prompts, but logs suspicious patterns.

        Args:
            prompt: Input prompt to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not prompt:
            return False, "Prompt cannot be empty"

        if not isinstance(prompt, str):
            return False, "Prompt must be a string"

        if len(prompt) > PROMPT_MEMORY_THRESHOLD:
            return False, "Request would exceed memory limits"

        if len(prompt) > MAX_PROMPT_LENGTH:
            return False, "Prompt exceeds maximum length"

        if any(ord(ch) < 32 and ch not in {"\n", "\r", "\t"} for ch in prompt):
            return False, "Prompt contains invalid control characters"

        # Check for potential command injection patterns
        dangerous_patterns = [
            r";\s*DROP\s+",  # SQL injection
            r";\s*DELETE\s+",  # SQL injection
            r";\s*UPDATE\s+",  # SQL injection
            r";\s*INSERT\s+",  # SQL injection
            r"<script",  # XSS
            r"javascript:",  # XSS
            r"\x00",  # Null bytes
            r"\\x[0-9a-fA-F]{2}",  # Hex escapes that might bypass filters
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                logger.warning(
                    f"Potentially dangerous pattern detected in prompt: {pattern}"
                )
                # AIDEV-NOTE: We log but don't block - adjust based on security requirements

        return True, None

    @staticmethod
    def validate_max_tokens(max_tokens: Any) -> tuple[bool, Optional[str]]:
        """
        Validate max_tokens parameter.

        AIDEV-NOTE: Ensures max_tokens is within reasonable bounds to
        prevent resource exhaustion.

        Args:
            max_tokens: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if max_tokens is None:
            return True, None  # Will use default

        try:
            tokens = int(max_tokens)
        except (ValueError, TypeError):
            return False, "max_tokens must be an integer"

        if tokens < MIN_TOKENS_LIMIT:
            return False, f"max_tokens must be at least {MIN_TOKENS_LIMIT}"

        if tokens > MAX_TOKENS_LIMIT:
            return False, "max_tokens exceeds system limit"

        return True, None

    @staticmethod
    def validate_model_name(model_name: str) -> tuple[bool, Optional[str]]:
        """
        Validate model name parameter.

        AIDEV-NOTE: Ensures model name is from allowed list to prevent
        arbitrary model loading.

        Args:
            model_name: Model name to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # AIDEV-NOTE: Extend this list as more models are supported
        ALLOWED_MODELS = {
            "qwen3-1.7b",
            "qwen3-0.6b",
            "qwen3-embedding",
            "small",
            "medium",
            "large",
        }

        if not model_name:
            return True, None  # Will use default

        if not isinstance(model_name, str):
            return False, "Model name must be a string"

        if model_name.lower() not in ALLOWED_MODELS:
            return False, f"Model '{model_name}' is not in allowed list"

        return True, None

    @staticmethod
    def sanitize_cache_key(key: str) -> str:
        """
        Sanitize a cache key for safe storage.

        AIDEV-NOTE: Ensures cache keys don't contain characters that
        could cause issues in PostgreSQL or file systems.

        Args:
            key: Raw cache key

        Returns:
            Sanitized cache key
        """
        # If key is already a hash, return as-is
        if re.match(r"^[a-f0-9]{32}$", key):
            return key

        # Otherwise, hash it to ensure safety
        return hashlib.md5(key.encode()).hexdigest()

    @staticmethod
    def validate_batch_size(batch_items: list) -> tuple[bool, Optional[str]]:
        """
        Validate batch operation size.

        AIDEV-NOTE: Prevents resource exhaustion from overly large batches.

        Args:
            batch_items: List of items in batch

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(batch_items, list):
            return False, "Batch must be a list"

        if len(batch_items) == 0:
            return False, "Batch cannot be empty"

        if len(batch_items) > MAX_BATCH_SIZE:
            return False, f"Batch size cannot exceed {MAX_BATCH_SIZE} items"

        return True, None

    @staticmethod
    def validate_embedding_text(text: Any) -> tuple[bool, Optional[str]]:
        """Validate text provided for embedding generation."""

        if text is None:
            return False, "Text cannot be null"

        if not isinstance(text, str):
            return False, "Text must be a string"

        if len(text) > EMBEDDING_BUFFER_THRESHOLD:
            return False, "Input exceeds buffer limits"

        if len(text) > MAX_EMBEDDING_TEXT_LENGTH:
            return False, "Text exceeds maximum length for embedding"

        if any(ord(ch) < 32 and ch not in {"\n", "\r", "\t"} for ch in text):
            return False, "Text contains invalid control characters"

        return True, None

    @staticmethod
    def validate_host(host: str) -> tuple[bool, Optional[str]]:
        """Validate configuration host strings."""

        if host is None:
            return False, "Invalid host format"

        if not isinstance(host, str):
            return False, "Invalid host format"

        if not re.fullmatch(r"[A-Za-z0-9._-]+", host.strip()):
            return False, "Invalid host format"

        return True, None

    @staticmethod
    def validate_port(port_value: str) -> tuple[bool, Optional[str]]:
        """Validate configuration port strings."""

        if port_value is None:
            return False, "Invalid port format"

        try:
            port_int = int(str(port_value).strip())
        except (TypeError, ValueError):
            return False, "Invalid port format"

        if port_int < 1 or port_int > 65535:
            return False, "Invalid port format"

        return True, None

    @staticmethod
    def validate_table_name(name: str) -> tuple[bool, Optional[str]]:
        """Validate table name inputs for configuration."""

        if name is None:
            return False, "Invalid table name"

        if not isinstance(name, str):
            return False, "Invalid table name"

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name.strip()):
            return False, "Invalid table name"

        return True, None

    @staticmethod
    def validate_regex_pattern(pattern: Any) -> tuple[bool, Optional[str]]:
        """Validate regex patterns to avoid obvious injection attempts."""

        if pattern is None:
            return False, "Invalid or dangerous regex pattern"

        if not isinstance(pattern, str) or pattern.strip() == "":
            return False, "Invalid or dangerous regex pattern"

        lowered = pattern.lower()
        dangerous_tokens = [";", "--", " drop ", " alter ", "\x00"]
        if any(token in lowered for token in dangerous_tokens):
            return False, "Invalid or dangerous regex pattern"

        try:
            re.compile(pattern)
        except re.error:
            return False, "Invalid or dangerous regex pattern"

        return True, None

    @staticmethod
    def validate_choice_list(choices: Any) -> tuple[bool, Optional[str]]:
        """Validate choice lists used for constrained generation."""

        if choices is None:
            return False, "Choices cannot be null"

        if not isinstance(choices, list):
            return False, "Choices must be an array"

        if len(choices) == 0:
            return False, "Choices array cannot be empty"

        if len(choices) == 1:
            return False, "Choices array must contain at least 2 options"

        for choice in choices:
            if choice is None or not isinstance(choice, str) or choice.strip() == "":
                return False, "Choices contain dangerous strings"
            lowered = choice.lower()
            if (
                ";" in lowered
                or "--" in lowered
                or " drop " in lowered
                or "\x00" in lowered
            ):
                return False, "Choices contain dangerous strings"

        if len(set(choices)) != len(choices):
            return False, "Choices array cannot contain duplicates"

        return True, None

    @staticmethod
    def validate_json_params(params: Any) -> tuple[bool, Optional[str]]:
        """
        Validate JSON parameters.

        AIDEV-NOTE: Ensures params are safe JSON-serializable dict.

        Args:
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if params is None:
            return True, None

        if not isinstance(params, dict):
            return False, "Parameters must be a dictionary"

        # Check for dangerous keys
        dangerous_keys = ["__proto__", "constructor", "prototype"]
        for key in params:
            if key in dangerous_keys:
                return False, f"Parameter key '{key}' is not allowed"

        # Validate common parameter values
        if "temperature" in params:
            try:
                temp = float(params["temperature"])
                if temp < 0 or temp > 2:
                    return False, "Temperature must be between 0 and 2"
            except (ValueError, TypeError):
                return False, "Temperature must be a number"

        return True, None


class RateLimiter:
    """
    Rate limiting implementation for pg_steadytext.

    AIDEV-NOTE: This is a placeholder for rate limiting logic.
    In production, this would integrate with PostgreSQL's rate
    limiting tables to track and enforce limits per user.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id

    def check_rate_limit(self) -> tuple[bool, Optional[str]]:
        """
        Check if user is within rate limits.

        AIDEV-NOTE: Implements sliding window rate limiting with minute/hour/day buckets.
        Updates counters atomically and resets based on time windows.

        Returns:
            Tuple of (is_allowed, error_message)
        """
        if not IN_POSTGRES:
            return True, None

        try:
            # Get current time for logging if needed
            # now_result = plpy.execute("SELECT NOW() as now")
            # current_time = now_result[0]["now"]

            # Upsert user rate limit record and check/update counters atomically
            check_plan = plpy.prepare(
                """
                WITH rate_check AS (
                    INSERT INTO steadytext_rate_limits (user_id)
                    VALUES ($1)
                    ON CONFLICT (user_id) DO UPDATE
                    SET
                        -- Reset minute counter if more than 1 minute passed
                        current_minute_count = CASE
                            WHEN EXTRACT(EPOCH FROM (NOW() - last_reset_minute)) >= 60
                            THEN 1
                            ELSE current_minute_count + 1
                        END,
                        last_reset_minute = CASE
                            WHEN EXTRACT(EPOCH FROM (NOW() - last_reset_minute)) >= 60
                            THEN NOW()
                            ELSE last_reset_minute
                        END,
                        -- Reset hour counter if more than 1 hour passed
                        current_hour_count = CASE
                            WHEN EXTRACT(EPOCH FROM (NOW() - last_reset_hour)) >= 3600
                            THEN 1
                            ELSE current_hour_count + 1
                        END,
                        last_reset_hour = CASE
                            WHEN EXTRACT(EPOCH FROM (NOW() - last_reset_hour)) >= 3600
                            THEN NOW()
                            ELSE last_reset_hour
                        END,
                        -- Reset day counter if more than 1 day passed
                        current_day_count = CASE
                            WHEN EXTRACT(EPOCH FROM (NOW() - last_reset_day)) >= 86400
                            THEN 1
                            ELSE current_day_count + 1
                        END,
                        last_reset_day = CASE
                            WHEN EXTRACT(EPOCH FROM (NOW() - last_reset_day)) >= 86400
                            THEN NOW()
                            ELSE last_reset_day
                        END
                    RETURNING
                        current_minute_count,
                        current_hour_count,
                        current_day_count,
                        requests_per_minute,
                        requests_per_hour,
                        requests_per_day
                )
                SELECT * FROM rate_check
            """,
                ["text"],
            )

            result = plpy.execute(check_plan, [self.user_id])
            if not result:
                return False, "Failed to check rate limits"

            limits = result[0]

            # Check if any limit is exceeded
            if limits["current_minute_count"] > limits["requests_per_minute"]:
                return (
                    False,
                    f"Rate limit exceeded: {limits['requests_per_minute']} requests per minute",
                )

            if limits["current_hour_count"] > limits["requests_per_hour"]:
                return (
                    False,
                    f"Rate limit exceeded: {limits['requests_per_hour']} requests per hour",
                )

            if limits["current_day_count"] > limits["requests_per_day"]:
                return (
                    False,
                    f"Rate limit exceeded: {limits['requests_per_day']} requests per day",
                )

            return True, None

        except Exception as e:
            plpy.warning(f"Rate limit check failed: {e}")
            # On error, allow the request but log the issue
            return True, None


def validate_generation_request(
    prompt: str,
    max_tokens: Optional[int] = None,
    model_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """
    Validate a complete generation request.

    AIDEV-NOTE: Comprehensive validation for text generation requests.
    Use this in PostgreSQL functions before processing.

    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        model_name: Model to use
        params: Additional parameters
        user_id: User identifier for rate limiting

    Returns:
        Tuple of (is_valid, error_message)
    """
    validator = SecurityValidator()

    # Validate prompt
    valid, error = validator.validate_prompt(prompt)
    if not valid:
        return False, error

    # Validate max_tokens
    valid, error = validator.validate_max_tokens(max_tokens)
    if not valid:
        return False, error

    # Validate model name
    if model_name:
        valid, error = validator.validate_model_name(model_name)
        if not valid:
            return False, error

    # Validate parameters
    if params:
        valid, error = validator.validate_json_params(params)
        if not valid:
            return False, error

    # Check rate limits
    if user_id:
        limiter = RateLimiter(user_id)
        allowed, error = limiter.check_rate_limit()
        if not allowed:
            return False, error

    return True, None


# AIDEV-NOTE: Export key functions for PostgreSQL integration
__all__ = [
    "SecurityValidator",
    "RateLimiter",
    "validate_generation_request",
    "MAX_PROMPT_LENGTH",
    "MAX_TOKENS_LIMIT",
]
