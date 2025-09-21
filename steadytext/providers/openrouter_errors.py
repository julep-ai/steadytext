"""OpenRouter-specific exception hierarchy.

AIDEV-ANCHOR: OpenRouter error classes
This module defines the exception hierarchy for OpenRouter provider errors,
following the contract specification for proper error handling and classification.
"""

from typing import Optional, Dict, Any


class OpenRouterError(RuntimeError):
    """Base exception class for OpenRouter-specific errors.

    AIDEV-NOTE: Inherits from RuntimeError as specified in data model.
    All OpenRouter errors should inherit from this base class.

    Attributes:
        status_code: HTTP status code from OpenRouter API response
        response_data: Raw error response data from OpenRouter API
        message: Human-readable error message
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
    ):
        """Initialize OpenRouter error.

        Args:
            message: Human-readable error message
            status_code: HTTP status code from API response
            response_data: Raw error response data from OpenRouter
        """
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}
        self.message = message

    def __str__(self) -> str:
        """Return formatted error message with context."""
        if self.status_code:
            return f"OpenRouter API Error {self.status_code}: {self.message}"
        return f"OpenRouter Error: {self.message}"


class OpenRouterAuthError(OpenRouterError):
    """Authentication and authorization errors.

    AIDEV-NOTE: Triggered by:
    - Missing or invalid API key
    - HTTP 401 responses
    - API key format validation failures
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
    ):
        """Initialize authentication error.

        Args:
            message: Error message (defaults to generic auth failure)
            status_code: HTTP status code (typically 401)
            response_data: API response data
        """
        super().__init__(message, status_code, response_data)


class OpenRouterRateLimitError(OpenRouterError):
    """Rate limiting errors with retry information.

    AIDEV-NOTE: Triggered by:
    - HTTP 429 responses
    - Rate limit headers in response

    Attributes:
        retry_after: Suggested retry delay in seconds from API headers
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        retry_after: Optional[int] = None,
    ):
        """Initialize rate limit error.

        Args:
            message: Error message
            status_code: HTTP status code (typically 429)
            response_data: API response data
            retry_after: Suggested retry delay in seconds
        """
        super().__init__(message, status_code, response_data)
        self.retry_after = retry_after

    def __str__(self) -> str:
        """Return formatted error message with retry information."""
        base_msg = super().__str__()
        if self.retry_after:
            return f"{base_msg} (retry after {self.retry_after} seconds)"
        return base_msg


class OpenRouterModelError(OpenRouterError):
    """Model-related errors.

    AIDEV-NOTE: Triggered by:
    - Invalid model names
    - Model not available
    - HTTP 404 responses for model endpoints
    """

    def __init__(
        self,
        message: str = "Model error",
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
    ):
        """Initialize model error.

        Args:
            message: Error message
            status_code: HTTP status code (typically 404)
            response_data: API response data
        """
        super().__init__(message, status_code, response_data)


class OpenRouterTimeoutError(OpenRouterError):
    """Request timeout errors.

    AIDEV-NOTE: For connection and read timeouts during API requests.
    """

    def __init__(
        self,
        message: str = "Request timeout",
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
    ):
        """Initialize timeout error.

        Args:
            message: Error message
            status_code: HTTP status code if available
            response_data: API response data if available
        """
        super().__init__(message, status_code, response_data)


class OpenRouterConnectionError(OpenRouterError):
    """Network connection errors.

    AIDEV-NOTE: For network-level connection failures.
    """

    def __init__(
        self,
        message: str = "Connection failed",
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
    ):
        """Initialize connection error.

        Args:
            message: Error message
            status_code: HTTP status code if available
            response_data: API response data if available
        """
        super().__init__(message, status_code, response_data)


def map_http_error_to_exception(
    status_code: int,
    response_data: Dict[str, Any],
    default_message: str = "API request failed",
) -> OpenRouterError:
    """Map HTTP status codes to appropriate OpenRouter exception types.

    AIDEV-ANCHOR: HTTP error mapping
    Maps HTTP status codes from OpenRouter API to specific exception types
    for proper error handling and classification.

    Args:
        status_code: HTTP status code from response
        response_data: Raw response data from OpenRouter API
        default_message: Fallback error message

    Returns:
        Appropriate OpenRouter exception instance
    """
    # Extract error message from response if available
    message = default_message
    if isinstance(response_data, dict):
        # Try common error message fields
        for field in ["error", "message", "detail", "error_message"]:
            if field in response_data:
                error_data = response_data[field]
                if isinstance(error_data, str):
                    message = error_data
                elif isinstance(error_data, dict) and "message" in error_data:
                    message = error_data["message"]
                break

    # Map status codes to exception types
    if status_code == 401:
        return OpenRouterAuthError(message, status_code, response_data)
    elif status_code == 403:
        return OpenRouterAuthError(f"Forbidden: {message}", status_code, response_data)
    elif status_code == 404:
        return OpenRouterModelError(f"Not found: {message}", status_code, response_data)
    elif status_code == 429:
        # Extract retry-after header if available
        retry_after = None
        if "headers" in response_data:
            headers = response_data["headers"]
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after:
                try:
                    retry_after = int(retry_after)
                except (ValueError, TypeError):
                    retry_after = None

        return OpenRouterRateLimitError(
            message, status_code, response_data, retry_after
        )
    elif status_code >= 500:
        return OpenRouterError(f"Server error: {message}", status_code, response_data)
    else:
        return OpenRouterError(message, status_code, response_data)


def should_retry_error(error: Exception) -> bool:
    """Determine if an error should trigger a retry attempt.

    Args:
        error: Exception that occurred

    Returns:
        True if retry should be attempted
    """
    # Retry for rate limits and temporary server errors
    if isinstance(error, OpenRouterRateLimitError):
        return True

    if isinstance(error, OpenRouterError) and error.status_code:
        # Retry for server errors (5xx) and some client errors
        if error.status_code >= 500:
            return True
        # Retry for specific temporary errors
        if error.status_code in [408, 502, 503, 504]:
            return True

    # Retry for connection and timeout errors
    if isinstance(error, (OpenRouterConnectionError, OpenRouterTimeoutError)):
        return True

    return False
