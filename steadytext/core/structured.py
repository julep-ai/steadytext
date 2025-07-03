"""Structured generation support using Outlines.

This module provides deterministic structured text generation with support for:
- JSON schemas (dict or Pydantic models)
- Regular expression patterns
- Choice constraints (multiple choice)
- Type constraints (int, float, bool, str)
"""

import logging
import re
from typing import Any, Dict, List, Union, Type, Optional

try:
    import outlines
    from pydantic import BaseModel

    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    BaseModel = None  # type: ignore[assignment, misc]

from ..models.loader import get_generator_model_instance
from ..utils import suppress_llama_output
from .generator import _validate_input_length

logger = logging.getLogger(__name__)


class StructuredGenerator:
    """Handles structured text generation using Outlines."""

    def __init__(self):
        """Initialize the structured generator."""
        self._model = None
        self._outlines_model = None

    def _ensure_model_loaded(self):
        """Ensure the model is loaded and wrapped with Outlines."""
        if self._outlines_model is None:
            if not OUTLINES_AVAILABLE:
                raise ImportError(
                    "Outlines is not installed. Install with: pip install outlines"
                )

            # Get the llama.cpp model instance
            llama_model = get_generator_model_instance()
            if llama_model is None:
                raise RuntimeError("Failed to load generation model")

            # AIDEV-NOTE: Known issue with Outlines 1.0.3+ and certain model vocabularies
            # Some models (e.g., Gemma-3n, Qwen1.5, Phi-2) have tokens that cannot be
            # converted to bytes, causing RuntimeError. This is tracked in:
            # - https://github.com/outlines-dev/outlines/issues/820
            # - https://github.com/dottxt-ai/outlines/issues/1261
            try:
                # Wrap with Outlines
                with suppress_llama_output():
                    self._outlines_model = outlines.from_llamacpp(llama_model)  # type: ignore[attr-defined]
                self._model = llama_model
            except RuntimeError as e:
                if "Cannot convert token" in str(e) and "to bytes" in str(e):
                    logger.error(
                        "Model vocabulary incompatibility with Outlines: %s. "
                        "This is a known issue with certain models. See "
                        "https://github.com/outlines-dev/outlines/issues/820",
                        str(e),
                    )
                    raise RuntimeError(
                        f"Model vocabulary is incompatible with Outlines structured generation: {e}"
                    ) from e
                else:
                    raise

    def generate_json(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type["BaseModel"], Type],
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """Generate JSON that conforms to a schema.

        Args:
            prompt: The input prompt
            schema: JSON schema dict, Pydantic model, or Python type
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            JSON string that conforms to the schema
        """
        self._ensure_model_loaded()

        # Validate input length
        _validate_input_length(self._model, prompt, max_tokens)

        # AIDEV-NOTE: Add structured generation instruction to prompt
        structured_prompt = (
            prompt
            + "\n\nYou may output json if relevant at the end inside <json-output></json-output> xml tags"
        )

        # First, generate thoughts up to <json- tag
        with suppress_llama_output():
            # Set stop token to generate thoughts first
            thoughts = self._model(
                structured_prompt, max_tokens=max_tokens, stop=["<json-"], **kwargs
            )["choices"][0]["text"]

        # Now generate the structured JSON
        full_prompt = structured_prompt + thoughts + "<json-output>"

        # Create the structured generator
        # AIDEV-NOTE: The token conversion error can also happen here when creating
        # generators for specific schemas, not just during model wrapping
        try:
            if isinstance(schema, dict):
                # JSON schema dict
                generator = outlines.generate.json(self._outlines_model, schema)
            elif isinstance(schema, type) and issubclass(schema, BaseModel):
                # Pydantic model
                generator = outlines.generate.json(self._outlines_model, schema)
            elif isinstance(schema, type):
                # Basic Python type
                generator = outlines.generate.json(self._outlines_model, schema)
            else:
                raise ValueError(f"Unsupported schema type: {type(schema)}")
        except RuntimeError as e:
            if "Cannot convert token" in str(e) and "to bytes" in str(e):
                logger.error(
                    "Model vocabulary incompatibility with Outlines: %s. "
                    "This is a known issue with certain models. See "
                    "https://github.com/outlines-dev/outlines/issues/820",
                    str(e),
                )
                raise RuntimeError(
                    f"Model vocabulary is incompatible with Outlines structured generation: {e}"
                ) from e
            else:
                raise

        # Generate the JSON
        with suppress_llama_output():
            json_output = generator(full_prompt, max_tokens=max_tokens, **kwargs)

        # Return the complete output with XML tags
        return thoughts + "<json-output>" + json_output + "</json-output>"

    def generate_regex(
        self, prompt: str, pattern: str, max_tokens: int = 512, **kwargs
    ) -> str:
        """Generate text that matches a regex pattern.

        Args:
            prompt: The input prompt
            pattern: Regular expression pattern
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Text that matches the pattern
        """
        self._ensure_model_loaded()

        # Validate input length
        _validate_input_length(self._model, prompt, max_tokens)

        # Validate regex pattern
        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

        # Create the regex generator
        try:
            generator = outlines.generate.regex(self._outlines_model, pattern)
        except RuntimeError as e:
            if "Cannot convert token" in str(e) and "to bytes" in str(e):
                logger.error(
                    "Model vocabulary incompatibility with Outlines: %s. "
                    "This is a known issue with certain models. See "
                    "https://github.com/outlines-dev/outlines/issues/820",
                    str(e),
                )
                raise RuntimeError(
                    f"Model vocabulary is incompatible with Outlines structured generation: {e}"
                ) from e
            else:
                raise

        # Generate text matching the pattern
        with suppress_llama_output():
            result = generator(prompt, max_tokens=max_tokens, **kwargs)

        return result

    def generate_choice(
        self, prompt: str, choices: List[str], max_tokens: int = 512, **kwargs
    ) -> str:
        """Generate text that is one of the given choices.

        Args:
            prompt: The input prompt
            choices: List of allowed string choices
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            One of the provided choices
        """
        self._ensure_model_loaded()

        # Validate input length
        _validate_input_length(self._model, prompt, max_tokens)

        if not choices:
            raise ValueError("Choices list cannot be empty")

        # Create the choice generator
        try:
            generator = outlines.generate.choice(self._outlines_model, choices)
        except RuntimeError as e:
            if "Cannot convert token" in str(e) and "to bytes" in str(e):
                logger.error(
                    "Model vocabulary incompatibility with Outlines: %s. "
                    "This is a known issue with certain models. See "
                    "https://github.com/outlines-dev/outlines/issues/820",
                    str(e),
                )
                raise RuntimeError(
                    f"Model vocabulary is incompatible with Outlines structured generation: {e}"
                ) from e
            else:
                raise

        # Generate one of the choices
        with suppress_llama_output():
            result = generator(prompt, **kwargs)

        return result

    def generate_format(
        self, prompt: str, format_type: Type, max_tokens: int = 512, **kwargs
    ) -> str:
        """Generate text of a specific type (int, float, bool, str).

        Args:
            prompt: The input prompt
            format_type: Python type (int, float, bool, str)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Text formatted as the specified type
        """
        self._ensure_model_loaded()

        # Validate input length
        _validate_input_length(self._model, prompt, max_tokens)

        # Use the appropriate generator based on type
        if format_type in (int, float, bool, str):
            generator = outlines.generate.format(self._outlines_model, format_type)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

        # Generate formatted text
        with suppress_llama_output():
            result = generator(prompt, max_tokens=max_tokens, **kwargs)

        return str(result)


# Singleton instance
_structured_generator: Optional[StructuredGenerator] = None


def get_structured_generator() -> StructuredGenerator:
    """Get the singleton structured generator instance."""
    global _structured_generator
    if _structured_generator is None:
        _structured_generator = StructuredGenerator()
    assert _structured_generator is not None  # Help type checker
    return _structured_generator  # type: ignore[invalid-return-type]


# AIDEV-NOTE: Public API functions for structured generation
def generate_json(
    prompt: str,
    schema: Union[Dict[str, Any], Type["BaseModel"], Type],
    max_tokens: int = 512,
    **kwargs,
) -> str:
    """Generate JSON that conforms to a schema.

    This function generates text that conforms to a JSON schema, Pydantic model,
    or basic Python type. The output is wrapped in <json-output> tags.

    Args:
        prompt: The input prompt
        schema: JSON schema dict, Pydantic model, or Python type
        max_tokens: Maximum tokens to generate
        **kwargs: Additional generation parameters

    Returns:
        JSON string with thoughts and structured output in XML tags

    Examples:
        >>> # Using a JSON schema
        >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        >>> result = generate_json("Create a person", schema)

        >>> # Using a Pydantic model
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> result = generate_json("Create a person", Person)

        >>> # Using a basic type
        >>> result = generate_json("Pick a number", int)
    """
    generator = get_structured_generator()
    return generator.generate_json(prompt, schema, max_tokens, **kwargs)


def generate_regex(prompt: str, pattern: str, max_tokens: int = 512, **kwargs) -> str:
    """Generate text that matches a regex pattern.

    Args:
        prompt: The input prompt
        pattern: Regular expression pattern
        max_tokens: Maximum tokens to generate
        **kwargs: Additional generation parameters

    Returns:
        Text that matches the pattern

    Examples:
        >>> # Generate a phone number
        >>> result = generate_regex("Call me at", r"\d{3}-\d{3}-\d{4}")

        >>> # Generate an email
        >>> result = generate_regex("Email:", r"[a-z]+@[a-z]+\.[a-z]+")
    """
    generator = get_structured_generator()
    return generator.generate_regex(prompt, pattern, max_tokens, **kwargs)


def generate_choice(
    prompt: str, choices: List[str], max_tokens: int = 512, **kwargs
) -> str:
    """Generate text that is one of the given choices.

    Args:
        prompt: The input prompt
        choices: List of allowed string choices
        max_tokens: Maximum tokens to generate
        **kwargs: Additional generation parameters

    Returns:
        One of the provided choices

    Examples:
        >>> # Multiple choice question
        >>> result = generate_choice(
        ...     "Is Python good?",
        ...     ["yes", "no", "maybe"]
        ... )
    """
    generator = get_structured_generator()
    return generator.generate_choice(prompt, choices, max_tokens, **kwargs)


def generate_format(
    prompt: str, format_type: Type, max_tokens: int = 512, **kwargs
) -> str:
    """Generate text of a specific type.

    Args:
        prompt: The input prompt
        format_type: Python type (int, float, bool, str)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional generation parameters

    Returns:
        Text formatted as the specified type

    Examples:
        >>> # Generate an integer
        >>> result = generate_format("How many?", int)

        >>> # Generate a boolean
        >>> result = generate_format("True or false?", bool)
    """
    generator = get_structured_generator()
    return generator.generate_format(prompt, format_type, max_tokens, **kwargs)
