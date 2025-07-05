"""Structured generation support using llama.cpp's native grammar support.

This module provides deterministic structured text generation with support for:
- JSON schemas (dict or Pydantic models)
- Regular expression patterns
- Choice constraints (multiple choice)
- Type constraints (int, float, bool, str)

AIDEV-NOTE: This implementation uses llama.cpp's native GBNF grammar support
instead of Outlines to avoid compatibility issues with models like Gemma-3n.
"""

import logging
import re
from typing import Any, Dict, List, Union, Type, Optional

from pydantic import BaseModel

from ..models.loader import get_generator_model_instance
from ..utils import suppress_llama_output
from .generator import _validate_input_length
from .grammar import json_schema_to_grammar, regex_to_grammar, choices_to_grammar

# AIDEV-NOTE: Import LlamaGrammar for creating grammar objects from GBNF strings
# llama-cpp-python expects LlamaGrammar objects, not raw GBNF strings
# Fixed issue #28: AttributeError: 'str' object has no attribute '_grammar'
from llama_cpp import LlamaGrammar


logger = logging.getLogger(__name__)


class StructuredGenerator:
    """Handles structured text generation using llama.cpp grammars."""

    def __init__(self):
        """Initialize the structured generator."""
        self._model = None

    def _ensure_model_loaded(self):
        """Ensure the model is loaded."""
        if self._model is None:
            # Get the llama.cpp model instance
            llama_model = get_generator_model_instance()
            if llama_model is None:
                raise RuntimeError("Failed to load generation model")
            self._model = llama_model

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

        # Convert schema to JSON schema if needed
        json_schema = self._schema_to_json_schema(schema)

        # Convert JSON schema to GBNF grammar
        grammar_str = json_schema_to_grammar(json_schema)

        # AIDEV-NOTE: Create LlamaGrammar object from GBNF string
        if LlamaGrammar is not None:
            grammar = LlamaGrammar.from_string(grammar_str)
        else:
            # Fallback: pass the string directly (for older versions)
            grammar = grammar_str

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

        # Generate JSON using grammar
        with suppress_llama_output():
            # AIDEV-NOTE: llama-cpp-python accepts grammar as a LlamaGrammar object
            result = self._model(
                full_prompt,
                max_tokens=max_tokens,
                grammar=grammar,
                stop=["</json-output>"],
                **kwargs,
            )
            json_output = result["choices"][0]["text"]

        # Return the complete output with XML tags
        return thoughts + "<json-output>" + json_output + "</json-output>"

    def _schema_to_json_schema(
        self, schema: Union[Dict[str, Any], Type["BaseModel"], Type]
    ) -> Dict[str, Any]:
        """Convert various schema types to JSON schema.

        Args:
            schema: JSON schema dict, Pydantic model, or Python type

        Returns:
            JSON schema dictionary
        """
        if isinstance(schema, dict):
            # Already a JSON schema
            return schema
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            # Pydantic model - convert to JSON schema
            # AIDEV-NOTE: Use Pydantic v2 method if available, else v1
            try:
                # Pydantic v2
                return schema.model_json_schema()
            except AttributeError:
                # Pydantic v1
                return schema.schema()
        elif isinstance(schema, type):
            # Basic Python type
            if schema is int:
                return {"type": "integer"}
            elif schema is float:
                return {"type": "number"}
            elif schema is str:
                return {"type": "string"}
            elif schema is bool:
                return {"type": "boolean"}
            else:
                raise ValueError(f"Unsupported Python type: {schema}")
        else:
            raise ValueError(f"Unsupported schema type: {type(schema)}")

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

        # Convert regex to GBNF grammar
        grammar_str = regex_to_grammar(pattern)

        # AIDEV-NOTE: Create LlamaGrammar object from GBNF string
        if LlamaGrammar is not None:
            grammar = LlamaGrammar.from_string(grammar_str)
        else:
            # Fallback: pass the string directly (for older versions)
            grammar = grammar_str

        # Generate text matching the pattern
        with suppress_llama_output():
            result = self._model(
                prompt, max_tokens=max_tokens, grammar=grammar, **kwargs
            )
            return result["choices"][0]["text"]

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

        # Convert choices to GBNF grammar string
        grammar_str = choices_to_grammar(choices)

        # AIDEV-NOTE: Create LlamaGrammar object from GBNF string
        if LlamaGrammar is not None:
            grammar = LlamaGrammar.from_string(grammar_str)
        else:
            # Fallback: pass the string directly (for older versions)
            grammar = grammar_str

        # Generate one of the choices
        with suppress_llama_output():
            result = self._model(
                prompt, max_tokens=max_tokens, grammar=grammar, **kwargs
            )
            return result["choices"][0]["text"]

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

        # Convert type to JSON schema then to grammar
        json_schema = self._schema_to_json_schema(format_type)
        grammar_str = json_schema_to_grammar(json_schema)

        # AIDEV-NOTE: Create LlamaGrammar object from GBNF string
        if LlamaGrammar is not None:
            grammar = LlamaGrammar.from_string(grammar_str)
        else:
            # Fallback: pass the string directly (for older versions)
            grammar = grammar_str

        # Generate formatted text
        with suppress_llama_output():
            result = self._model(
                prompt, max_tokens=max_tokens, grammar=grammar, **kwargs
            )
            return result["choices"][0]["text"]


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
