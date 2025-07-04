"""Structured generation support using llama.cpp grammars.

This module provides deterministic structured text generation with support for:
- JSON schemas (dict or Pydantic models)
- Regular expression patterns
- Choice constraints (multiple choice)
- Type constraints (int, float, bool, str)

AIDEV-NOTE: This replaces the Outlines implementation which has compatibility
issues with Gemma-3n models. Uses llama.cpp's native grammar support instead.
"""

import json
import logging
import re
from typing import Any, Dict, List, Union, Type, Optional

try:
    from pydantic import BaseModel
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None  # type: ignore[assignment, misc]

from ..models.loader import get_generator_model_instance
from ..utils import suppress_llama_output, LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC, DEFAULT_STOP_SEQUENCES
from .generator import _validate_input_length
from .grammar import GrammarConverter

logger = logging.getLogger(__name__)


class StructuredGenerator:
    """Handles structured text generation using llama.cpp grammars.
    
    AIDEV-NOTE: Class name kept as StructuredGenerator for backward compatibility,
    but implementation now uses llama.cpp grammars instead of Outlines.
    """

    def __init__(self):
        """Initialize the structured generator."""
        self._model = None
        self._grammar_converter = GrammarConverter()

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

        AIDEV-NOTE: Uses llama.cpp grammar-based generation instead of Outlines.
        """
        self._ensure_model_loaded()

        # Validate input length
        _validate_input_length(self._model, prompt, max_tokens)

        # Convert schema to JSON schema if needed
        json_schema = None
        if isinstance(schema, dict):
            json_schema = schema
        elif PYDANTIC_AVAILABLE and isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = self._grammar_converter.pydantic_to_json_schema(schema)
        elif isinstance(schema, type):
            # Basic Python type
            type_map = {
                int: {"type": "integer"},
                float: {"type": "number"},
                str: {"type": "string"},
                bool: {"type": "boolean"},
                dict: {"type": "object"},
                list: {"type": "array"},
            }
            json_schema = type_map.get(schema, {"type": "string"})
        else:
            raise ValueError(f"Unsupported schema type: {type(schema)}")

        # Convert JSON schema to GBNF grammar
        grammar = self._grammar_converter.json_schema_to_gbnf(json_schema)
        
        # AIDEV-NOTE: Add structured generation instruction to prompt
        structured_prompt = (
            prompt
            + "\n\nYou may output json if relevant at the end inside <json-output></json-output> xml tags"
        )

        # First, generate thoughts up to <json- tag
        with suppress_llama_output():
            # Set stop token to generate thoughts first
            sampling_params = {**LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC}
            sampling_params["stop"] = ["<json-"] + DEFAULT_STOP_SEQUENCES
            sampling_params["max_tokens"] = max_tokens
            
            # Use the model's generation method
            output = self._model.create_chat_completion(
                messages=[{"role": "user", "content": structured_prompt}],
                **sampling_params
            )
            
            thoughts = ""
            if output and "choices" in output and len(output["choices"]) > 0:
                choice = output["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    thoughts = choice["message"]["content"].strip()

        # Now generate the structured JSON with grammar
        full_prompt = structured_prompt + thoughts + "<json-output>"
        
        # Generate JSON with grammar constraint
        with suppress_llama_output():
            sampling_params = {**LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC}
            sampling_params["stop"] = ["</json-output>"] + DEFAULT_STOP_SEQUENCES
            sampling_params["max_tokens"] = max_tokens
            sampling_params["grammar"] = grammar  # AIDEV-NOTE: Key addition - grammar constraint
            
            output = self._model.create_chat_completion(
                messages=[{"role": "user", "content": full_prompt}],
                **sampling_params
            )
            
            json_output = ""
            if output and "choices" in output and len(output["choices"]) > 0:
                choice = output["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    json_output = choice["message"]["content"].strip()

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

        AIDEV-NOTE: Uses llama.cpp grammar-based generation for regex patterns.
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
        grammar = self._grammar_converter.regex_to_gbnf(pattern)

        # Generate text matching the pattern with grammar constraint
        with suppress_llama_output():
            sampling_params = {**LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC}
            sampling_params["stop"] = DEFAULT_STOP_SEQUENCES
            sampling_params["max_tokens"] = max_tokens
            sampling_params["grammar"] = grammar  # AIDEV-NOTE: Grammar constraint for regex
            
            output = self._model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **sampling_params
            )
            
            result = ""
            if output and "choices" in output and len(output["choices"]) > 0:
                choice = output["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    result = choice["message"]["content"].strip()

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

        AIDEV-NOTE: Uses llama.cpp grammar to constrain output to provided choices.
        """
        self._ensure_model_loaded()

        # Validate input length
        _validate_input_length(self._model, prompt, max_tokens)

        if not choices:
            raise ValueError("Choices list cannot be empty")

        # Convert choices to GBNF grammar
        grammar = self._grammar_converter.choices_to_gbnf(choices)

        # Generate one of the choices with grammar constraint
        with suppress_llama_output():
            sampling_params = {**LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC}
            sampling_params["stop"] = DEFAULT_STOP_SEQUENCES
            sampling_params["max_tokens"] = max_tokens
            sampling_params["grammar"] = grammar  # AIDEV-NOTE: Grammar constraint for choices
            
            output = self._model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **sampling_params
            )
            
            result = ""
            if output and "choices" in output and len(output["choices"]) > 0:
                choice = output["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    result = choice["message"]["content"].strip()

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

        AIDEV-NOTE: Uses JSON schema conversion for type constraints.
        """
        self._ensure_model_loaded()

        # Validate input length
        _validate_input_length(self._model, prompt, max_tokens)

        # Convert type to schema and generate
        if format_type == int:
            schema = {"type": "integer"}
        elif format_type == float:
            schema = {"type": "number"}
        elif format_type == bool:
            schema = {"type": "boolean"}
        elif format_type == str:
            schema = {"type": "string"}
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

        # Convert to grammar
        grammar = self._grammar_converter.json_schema_to_gbnf(schema)

        # Generate formatted text with grammar constraint
        with suppress_llama_output():
            sampling_params = {**LLAMA_CPP_GENERATION_SAMPLING_PARAMS_DETERMINISTIC}
            sampling_params["stop"] = DEFAULT_STOP_SEQUENCES
            sampling_params["max_tokens"] = max_tokens
            sampling_params["grammar"] = grammar
            
            output = self._model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **sampling_params
            )
            
            result = ""
            if output and "choices" in output and len(output["choices"]) > 0:
                choice = output["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    result = choice["message"]["content"].strip()

        return result


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

    AIDEV-NOTE: Now uses llama.cpp grammars instead of Outlines for compatibility
    with Gemma-3n models.
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

    AIDEV-NOTE: Now uses llama.cpp grammars for regex pattern matching.
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

    AIDEV-NOTE: Now uses llama.cpp grammars to constrain output to choices.
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

    AIDEV-NOTE: Now uses llama.cpp grammars for type constraints.
    """
    generator = get_structured_generator()
    return generator.generate_format(prompt, format_type, max_tokens, **kwargs)