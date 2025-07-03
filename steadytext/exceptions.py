"""Custom exceptions for SteadyText.

AIDEV-NOTE: This module defines custom exceptions for better error handling
and user feedback throughout the SteadyText library.
"""

from typing import Optional


class SteadyTextError(Exception):
    """Base exception for all SteadyText errors."""

    pass


class ContextLengthExceededError(SteadyTextError):
    """Raised when input exceeds the model's context window.

    AIDEV-NOTE: This exception provides detailed information about token counts
    to help users understand and fix the issue.
    """

    def __init__(
        self,
        input_tokens: int,
        max_tokens: int,
        input_text: Optional[str] = None,
        message: Optional[str] = None,
    ):
        """Initialize the exception with token count details.

        Args:
            input_tokens: Number of tokens in the input
            max_tokens: Maximum allowed tokens (context window size)
            input_text: Optional preview of the input text
            message: Optional custom error message
        """
        self.input_tokens = input_tokens
        self.max_tokens = max_tokens
        self.input_text = input_text

        if message is None:
            message = (
                f"Input exceeds context window: {input_tokens} tokens "
                f"(maximum: {max_tokens} tokens). "
                f"Please reduce input length or use a model with larger context window."
            )

            if input_text:
                preview = (
                    input_text[:100] + "..." if len(input_text) > 100 else input_text
                )
                message += f"\nInput preview: '{preview}'"

        super().__init__(message)


class ModelNotLoadedError(SteadyTextError):
    """Raised when attempting to use a model that failed to load.

    AIDEV-NOTE: This provides clearer error messages than returning None,
    while still maintaining the "never fails" principle through fallbacks.
    """

    def __init__(self, model_type: str = "generator", reason: Optional[str] = None):
        """Initialize the exception.

        Args:
            model_type: Type of model that failed ("generator" or "embedder")
            reason: Optional reason for the failure
        """
        self.model_type = model_type
        self.reason = reason

        message = f"The {model_type} model could not be loaded."
        if reason:
            message += f" Reason: {reason}"
        message += " Using fallback mode."

        super().__init__(message)


class InvalidModelError(SteadyTextError):
    """Raised when an invalid model is specified.

    AIDEV-NOTE: Helps users identify typos or unsupported model names.
    """

    def __init__(self, model_name: str, available_models: Optional[list] = None):
        """Initialize the exception.

        Args:
            model_name: The invalid model name provided
            available_models: Optional list of valid model names
        """
        self.model_name = model_name
        self.available_models = available_models

        message = f"Invalid model name: '{model_name}'."
        if available_models:
            message += f" Available models: {', '.join(available_models)}"

        super().__init__(message)


# AIDEV-TODO: Add TokenizationError for tokenizer-related issues
# AIDEV-TODO: Add CacheError for cache-related failures
# AIDEV-TODO: Consider adding DaemonConnectionError for daemon mode issues
