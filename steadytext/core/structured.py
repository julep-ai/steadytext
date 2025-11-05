"""Structured generation support using llama.cpp's native grammar support.

This module provides deterministic structured text generation with support for:
- JSON schemas (dict or Pydantic models)
- Regular expression patterns
- Choice constraints (multiple choice)
- Type constraints (int, float, bool, str)

AIDEV-NOTE: This implementation uses llama.cpp's native GBNF grammar support
instead of Outlines to avoid compatibility issues with models like Gemma-3n.

AIDEV-FIXED: Mini model (Gemma-3-270M QAT) compatibility issue resolved by using
LlamaGrammar.from_json_schema() instead of custom GBNF grammar generation.
The mini model would hang or produce invalid output with custom grammars but
works correctly with llama.cpp's built-in JSON schema conversion.
"""

import json
import logging
import os
import re
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    overload,
    cast,
)

from pydantic import BaseModel
from llama_cpp import Llama, LlamaGrammar

from ..models.loader import get_generator_model_instance
from ..utils import suppress_llama_output, DEFAULT_SEED
from .generator import _validate_input_length, core_generate
from .grammar import (
    json_schema_to_grammar,
    regex_to_grammar,
    choices_to_grammar,
)

# AIDEV-NOTE: Import LlamaGrammar for creating grammar objects from GBNF strings
# llama-cpp-python expects LlamaGrammar objects, not raw GBNF strings
# Fixed issue #28: AttributeError: 'str' object has no attribute '_grammar'

# Type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)


logger = logging.getLogger(__name__)


def _is_mini_model_active() -> bool:
    """Check if mini models are currently active.

    AIDEV-NOTE: Mini models like Gemma-3-270M QAT have issues with complex
    GBNF grammars, so we need special handling for them.
    """
    return os.environ.get("STEADYTEXT_USE_MINI_MODELS", "").lower() == "true"


class StructuredGenerator:
    """Handles structured text generation using llama.cpp grammars."""

    def __init__(self):
        """Initialize the structured generator."""
        self._model: Optional[Llama] = None

    def _ensure_model_loaded(self):
        """Ensure the model is loaded."""
        if self._model is None:
            # Get the llama.cpp model instance
            llama_model = get_generator_model_instance()
            if llama_model is None:
                raise RuntimeError("Failed to load generation model")
            self._model = llama_model

    def _get_model(self) -> Llama:
        """Return the loaded llama.cpp model, ensuring availability."""
        self._ensure_model_loaded()
        if self._model is None:
            raise RuntimeError("Generation model is not loaded")
        return self._model

    def _call_model(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Call the underlying llama.cpp model and normalize the response."""
        model = self._get_model()
        result = model(*args, **kwargs)
        if isinstance(result, dict):
            return cast(Dict[str, Any], result)
        if isinstance(result, Iterator):
            first_chunk: Optional[Dict[str, Any]] = None
            last_chunk: Optional[Dict[str, Any]] = None
            text_parts: List[str] = []
            delta_parts: List[str] = []
            final_message_content: Optional[str] = None
            final_message_role: Optional[str] = None
            usage_payload: Optional[Dict[str, Any]] = None

            for chunk in result:
                if not isinstance(chunk, dict):
                    continue
                if first_chunk is None:
                    first_chunk = chunk
                last_chunk = chunk
                chunk_usage = chunk.get("usage")
                if isinstance(chunk_usage, dict):
                    usage_payload = chunk_usage

                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue

                choice = choices[0]
                if not isinstance(choice, dict):
                    continue

                text_piece = choice.get("text")
                if isinstance(text_piece, str):
                    text_parts.append(text_piece)

                message = choice.get("message")
                if isinstance(message, dict):
                    message_content = message.get("content")
                    if isinstance(message_content, str):
                        final_message_content = message_content
                    message_role = message.get("role")
                    if isinstance(message_role, str):
                        final_message_role = message_role

                delta = choice.get("delta")
                if isinstance(delta, dict):
                    delta_content = delta.get("content")
                    if isinstance(delta_content, str):
                        delta_parts.append(delta_content)
                    delta_role = delta.get("role")
                    if isinstance(delta_role, str) and final_message_role is None:
                        final_message_role = delta_role

            if first_chunk is None:
                raise RuntimeError("Model iterator returned no response")

            combined_result = dict(first_chunk)

            first_choices = first_chunk.get("choices")
            combined_choice: Dict[str, Any] = {}
            if isinstance(first_choices, list) and first_choices:
                first_choice = first_choices[0]
                if isinstance(first_choice, dict):
                    combined_choice.update(first_choice)

            aggregated_text = "".join(text_parts)
            aggregated_delta = "".join(delta_parts)
            message_content = final_message_content or aggregated_delta

            if aggregated_text:
                combined_choice["text"] = aggregated_text
            if message_content:
                message_dict = combined_choice.get("message")
                if not isinstance(message_dict, dict):
                    message_dict = {}
                message_dict["content"] = message_content
                if final_message_role:
                    message_dict.setdefault("role", final_message_role)
                combined_choice["message"] = message_dict

            if not aggregated_text and not message_content:
                raise RuntimeError("Model iterator yielded no textual content")

            if last_chunk:
                last_choices = last_chunk.get("choices")
                if isinstance(last_choices, list) and last_choices:
                    last_choice = last_choices[0]
                    if isinstance(last_choice, dict):
                        for key in ("finish_reason", "logprobs", "index"):
                            if key in last_choice:
                                combined_choice[key] = last_choice[key]

            if usage_payload is not None:
                combined_result["usage"] = usage_payload

            # Drop streaming-specific delta payload to avoid leaking partial chunks
            combined_choice.pop("delta", None)

            combined_result["choices"] = [combined_choice]
            return cast(Dict[str, Any], combined_result)
        raise TypeError(f"Unexpected result type from model: {type(result)!r}")

    @staticmethod
    def _extract_choice_text(result: Dict[str, Any]) -> str:
        """Extract text content from a llama.cpp completion response."""
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                text_value = choice.get("text")
                if isinstance(text_value, str):
                    return text_value
                message = choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
        raise RuntimeError("Model response did not include textual content")

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
        model = self._get_model()

        # Validate input length
        _validate_input_length(model, prompt, max_tokens)

        # Convert schema to JSON schema if needed
        json_schema = self._schema_to_json_schema(schema)

        # AIDEV-NOTE: Use LlamaGrammar.from_json_schema() which is more reliable
        # than custom grammar generation, especially for mini models
        try:
            if hasattr(LlamaGrammar, "from_json_schema"):
                # Use the built-in JSON schema to grammar conversion
                grammar = LlamaGrammar.from_json_schema(json.dumps(json_schema))
                logger.debug(
                    "Using LlamaGrammar.from_json_schema() for grammar generation"
                )
            else:
                # Fall back to GBNF string generation for older versions
                grammar_str = json_schema_to_grammar(json_schema)
                grammar = LlamaGrammar.from_string(grammar_str)
                logger.debug(
                    "Using GBNF string generation (older llama-cpp-python version)"
                )
        except Exception as e:
            logger.error(f"Failed to create grammar from JSON schema: {e}")
            # Fall back to unconstrained generation
            return self._generate_json_without_grammar(
                prompt, schema, max_tokens, **kwargs
            )

        # AIDEV-NOTE: Add structured generation instruction to prompt
        structured_prompt = (
            prompt
            + "\n\nYou may output json if relevant at the end inside <json-output></json-output> xml tags"
        )

        try:
            # First, generate thoughts up to <json- tag
            with suppress_llama_output():
                # Set stop token to generate thoughts first
                thoughts_result = self._call_model(
                    structured_prompt, max_tokens=max_tokens, stop=["<json-"], **kwargs
                )
                thoughts = self._extract_choice_text(thoughts_result)

            # Now generate the structured JSON
            full_prompt = structured_prompt + thoughts + "<json-output>"

            # Generate JSON using grammar
            with suppress_llama_output():
                # AIDEV-NOTE: llama-cpp-python accepts grammar as a LlamaGrammar object
                result = self._call_model(
                    full_prompt,
                    max_tokens=max_tokens,
                    grammar=grammar,
                    stop=["</json-output>"],
                    **kwargs,
                )
                json_output = self._extract_choice_text(result)

            # Return the complete output with XML tags
            return thoughts + "<json-output>" + json_output + "</json-output>"
        except Exception as e:
            logger.error(f"Grammar-constrained generation failed: {e}")
            # Fall back to unconstrained generation
            return self._generate_json_without_grammar(
                prompt, schema, max_tokens, **kwargs
            )

    def _generate_json_without_grammar(
        self,
        prompt: str,
        schema: Union[Dict[str, Any], Type["BaseModel"], Type],
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """Generate JSON without grammar constraints as a fallback.

        AIDEV-NOTE: This is used when grammar generation or parsing fails,
        particularly with mini models. We use strong prompting to encourage
        JSON output in the correct format.
        """
        # Build a descriptive prompt based on the schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # For Pydantic models, describe the expected fields
            fields_desc = []
            for field_name, field_info in schema.model_fields.items():
                field_type = field_info.annotation
                if field_type is str:
                    type_desc = "string"
                elif field_type is int:
                    type_desc = "integer"
                elif field_type is float:
                    type_desc = "number"
                elif field_type is bool:
                    type_desc = "boolean"
                else:
                    type_desc = "value"
                fields_desc.append(f'"{field_name}": <{type_desc}>')

            json_template = "{" + ", ".join(fields_desc) + "}"
            enhanced_prompt = (
                f"{prompt}\n\n"
                f"Output valid JSON in this exact format: {json_template}\n"
                f"Remember to use proper JSON syntax with quoted strings.\n\n"
                f"<json-output>"
            )
        else:
            enhanced_prompt = (
                f"{prompt}\n\n"
                f"Output valid JSON that matches the required schema.\n"
                f"Remember to use proper JSON syntax.\n\n"
                f"<json-output>"
            )

        # Generate without grammar constraints
        with suppress_llama_output():
            result = self._call_model(
                enhanced_prompt,
                max_tokens=max_tokens,
                stop=["</json-output>"],
                **kwargs,
            )
            json_output = self._extract_choice_text(result)

        # Return with wrapper tags
        return "<json-output>" + json_output + "</json-output>"

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
                return schema.schema()  # type: ignore[attr-defined]
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
        model = self._get_model()

        # Validate input length
        _validate_input_length(model, prompt, max_tokens)

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
            result = self._call_model(
                prompt, max_tokens=max_tokens, grammar=grammar, **kwargs
            )
            return self._extract_choice_text(result)

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
        model = self._get_model()

        # Validate input length
        _validate_input_length(model, prompt, max_tokens)

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
            result = self._call_model(
                prompt, max_tokens=max_tokens, grammar=grammar, **kwargs
            )
            return self._extract_choice_text(result)

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
        model = self._get_model()

        # Validate input length
        _validate_input_length(model, prompt, max_tokens)

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
            result = self._call_model(
                prompt, max_tokens=max_tokens, grammar=grammar, **kwargs
            )
            return self._extract_choice_text(result)


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
# Overload for when return_pydantic=True with a Pydantic model
@overload
def generate_json(
    prompt: str,
    schema: Type[T],
    max_tokens: int = 512,
    model: Optional[str] = None,
    unsafe_mode: bool = False,
    seed: int = DEFAULT_SEED,
    return_pydantic: Literal[True] = True,
    **kwargs,
) -> T: ...


# Overload for when return_pydantic=False (default)
@overload
def generate_json(
    prompt: str,
    schema: Union[Dict[str, Any], Type["BaseModel"], Type],
    max_tokens: int = 512,
    model: Optional[str] = None,
    unsafe_mode: bool = False,
    seed: int = DEFAULT_SEED,
    return_pydantic: Literal[False] = False,
    **kwargs,
) -> str: ...


# Overload for when return_pydantic is a bool variable (runtime determined)
@overload
def generate_json(
    prompt: str,
    schema: Union[Dict[str, Any], Type["BaseModel"], Type],
    max_tokens: int = 512,
    model: Optional[str] = None,
    unsafe_mode: bool = False,
    seed: int = DEFAULT_SEED,
    return_pydantic: bool = False,
    **kwargs,
) -> Union[str, "BaseModel"]: ...


def generate_json(
    prompt: str,
    schema: Union[Dict[str, Any], Type["BaseModel"], Type],
    max_tokens: int = 512,
    model: Optional[str] = None,
    unsafe_mode: bool = False,
    seed: int = DEFAULT_SEED,
    return_pydantic: bool = False,
    **kwargs,
) -> Union[str, "BaseModel"]:
    """Generate JSON that conforms to a schema.

    This function generates text that conforms to a JSON schema, Pydantic model,
    or basic Python type. The output is wrapped in <json-output> tags by default,
    or can return a Pydantic model instance when return_pydantic=True.

    Args:
        prompt: The input prompt
        schema: JSON schema dict, Pydantic model, or Python type
        max_tokens: Maximum tokens to generate
        model: Optional model name for remote models
        unsafe_mode: Enable remote models with best-effort determinism
        seed: Random seed for deterministic generation
        return_pydantic: If True and schema is a Pydantic model, return the instantiated model
        **kwargs: Additional generation parameters

    Returns:
        If return_pydantic=False: JSON string with thoughts and structured output in XML tags
        If return_pydantic=True and schema is a Pydantic model: Instantiated Pydantic model
        Otherwise: JSON string with thoughts and structured output in XML tags

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

        >>> # Using a remote model
        >>> result = generate_json(
        ...     "Create a person",
        ...     {"type": "object", "properties": {"name": {"type": "string"}}},
        ...     model="openai:gpt-4o-mini",
        ...     unsafe_mode=True
        ... )
    """
    # AIDEV-NOTE: Check if this is a remote model request
    from ..providers.registry import is_remote_model

    if model and is_remote_model(model):
        # For remote models, check if they support structured output

        # Validate unsafe_mode requirement
        if not unsafe_mode:
            raise ValueError(f"Remote model '{model}' requires unsafe_mode=True")

        # Use core_generate with schema parameter for remote models
        result = core_generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            schema=schema,
            model=model,
            unsafe_mode=unsafe_mode,
            seed=seed,
            **kwargs,
        )

        if result is None:
            raise RuntimeError(f"Failed to generate JSON with remote model '{model}'")

        # Extract string result (handle logprobs tuple if present)
        if isinstance(result, tuple):
            first_part = result[0]
            if not isinstance(first_part, str):
                raise TypeError(
                    "Remote structured generation returned non-string result"
                )
            result_str = first_part
        elif isinstance(result, str):
            result_str = result
        else:
            raise TypeError("Remote structured generation returned non-string result")

        # Handle Pydantic model instantiation for remote models
        if (
            return_pydantic
            and isinstance(schema, type)
            and issubclass(schema, BaseModel)
        ):
            # Extract JSON from tags
            json_start = result_str.find("<json-output>") + len("<json-output>")
            json_end = result_str.find("</json-output>")
            if json_start > len("<json-output>") - 1 and json_end > json_start:
                json_str = result_str[json_start:json_end]
                return schema.model_validate_json(json_str)

        return result_str

    # For local models, use the structured generator
    generator = get_structured_generator()
    result = generator.generate_json(prompt, schema, max_tokens, seed=seed, **kwargs)

    # AIDEV-NOTE: Handle Pydantic model instantiation if requested
    if return_pydantic and isinstance(schema, type) and issubclass(schema, BaseModel):
        # Extract JSON from the <json-output> tags
        json_start = result.find("<json-output>") + len("<json-output>")
        json_end = result.find("</json-output>")
        if json_start > len("<json-output>") - 1 and json_end > json_start:
            json_str = result[json_start:json_end]
            # Parse and validate with Pydantic
            return schema.model_validate_json(json_str)

    return result


def generate_regex(
    prompt: str,
    pattern: str,
    max_tokens: int = 512,
    model: Optional[str] = None,
    unsafe_mode: bool = False,
    seed: int = DEFAULT_SEED,
    **kwargs,
) -> str:
    r"""Generate text that matches a regex pattern.

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

        >>> # Using a remote model
        >>> result = generate_regex(
        ...     "Call me at",
        ...     r"\d{3}-\d{3}-\d{4}",
        ...     model="openai:gpt-4o-mini",
        ...     unsafe_mode=True
        ... )
    """
    # AIDEV-NOTE: Check if this is a remote model request
    from ..providers.registry import is_remote_model

    if model and is_remote_model(model):
        # For remote models, check if they support structured output

        # Validate unsafe_mode requirement
        if not unsafe_mode:
            raise ValueError(f"Remote model '{model}' requires unsafe_mode=True")

        # Use core_generate with regex parameter for remote models
        result = core_generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            regex=pattern,
            model=model,
            unsafe_mode=unsafe_mode,
            seed=seed,
            **kwargs,
        )

        if result is None:
            raise RuntimeError(f"Failed to generate regex with remote model '{model}'")

        # Extract string result (handle logprobs tuple if present)
        if isinstance(result, tuple):
            text_part = result[0]
            if not isinstance(text_part, str):
                raise TypeError("Remote regex generation returned non-string result")
            return text_part
        if not isinstance(result, str):
            raise TypeError("Remote regex generation returned non-string result")
        return result

    # For local models, use the structured generator
    generator = get_structured_generator()
    return generator.generate_regex(prompt, pattern, max_tokens, seed=seed, **kwargs)


def generate_choice(
    prompt: str,
    choices: List[str],
    max_tokens: int = 512,
    model: Optional[str] = None,
    unsafe_mode: bool = False,
    seed: int = DEFAULT_SEED,
    **kwargs,
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

        >>> # Using a remote model
        >>> result = generate_choice(
        ...     "Is Python good?",
        ...     ["yes", "no", "maybe"],
        ...     model="openai:gpt-4o-mini",
        ...     unsafe_mode=True
        ... )
    """
    # AIDEV-NOTE: Check if this is a remote model request
    from ..providers.registry import is_remote_model

    if model and is_remote_model(model):
        # For remote models, check if they support structured output

        # Validate unsafe_mode requirement
        if not unsafe_mode:
            raise ValueError(f"Remote model '{model}' requires unsafe_mode=True")

        # Use core_generate with choices parameter for remote models
        result = core_generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            choices=choices,
            model=model,
            unsafe_mode=unsafe_mode,
            seed=seed,
            **kwargs,
        )

        if result is None:
            raise RuntimeError(f"Failed to generate choice with remote model '{model}'")

        # Extract string result (handle logprobs tuple if present)
        if isinstance(result, tuple):
            text_part = result[0]
            if not isinstance(text_part, str):
                raise TypeError("Remote choice generation returned non-string result")
            return text_part
        if not isinstance(result, str):
            raise TypeError("Remote choice generation returned non-string result")
        return result

    # For local models, use the structured generator
    generator = get_structured_generator()
    return generator.generate_choice(prompt, choices, max_tokens, seed=seed, **kwargs)


def generate_format(
    prompt: str,
    format_type: Type,
    max_tokens: int = 512,
    model: Optional[str] = None,
    unsafe_mode: bool = False,
    seed: int = DEFAULT_SEED,
    **kwargs,
) -> str:
    r"""Generate text of a specific type.

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

        >>> # Using a remote model
        >>> result = generate_format(
        ...     "How many?",
        ...     int,
        ...     model="openai:gpt-4o-mini",
        ...     unsafe_mode=True
        ... )
    """
    # AIDEV-NOTE: Check if this is a remote model request
    from ..providers.registry import is_remote_model

    if model and is_remote_model(model):
        # For remote models, validate unsafe_mode requirement
        if not unsafe_mode:
            raise ValueError(f"Remote model '{model}' requires unsafe_mode=True")

        # Convert format_type to JSON schema
        generator = get_structured_generator()
        json_schema = generator._schema_to_json_schema(format_type)

        # Use core_generate with schema parameter for remote models
        result = core_generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            schema=json_schema,
            model=model,
            unsafe_mode=unsafe_mode,
            seed=seed,
            **kwargs,
        )

        if result is None:
            raise RuntimeError(f"Failed to generate format with remote model '{model}'")

        # Extract string result (handle logprobs tuple if present)
        if isinstance(result, tuple):
            text_part = result[0]
            if not isinstance(text_part, str):
                raise TypeError(
                    "Remote formatted generation returned non-string result"
                )
            return text_part
        if not isinstance(result, str):
            raise TypeError("Remote formatted generation returned non-string result")
        return result

    # For local models, use the structured generator
    generator = get_structured_generator()
    return generator.generate_format(
        prompt, format_type, max_tokens, seed=seed, **kwargs
    )


def generate_pydantic(
    prompt: str,
    model_class: Type[T],
    max_tokens: int = 512,
    model: Optional[str] = None,
    unsafe_mode: bool = False,
    seed: int = DEFAULT_SEED,
    **kwargs,
) -> T:
    """Generate a Pydantic model instance that conforms to the given model class.

    This is a convenience function that always returns an instantiated Pydantic model.
    It's equivalent to calling generate_json() with return_pydantic=True.

    Args:
        prompt: The input prompt
        model_class: Pydantic BaseModel class to generate
        max_tokens: Maximum tokens to generate
        model: Optional model name for remote models
        unsafe_mode: Enable remote models with best-effort determinism
        seed: Random seed for deterministic generation
        **kwargs: Additional generation parameters

    Returns:
        An instance of the provided Pydantic model class

    Raises:
        ValueError: If model_class is not a Pydantic BaseModel
        ValidationError: If the generated JSON doesn't match the model schema

    Examples:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>>
        >>> user = generate_pydantic("Create user Alice age 30", User)
        >>> print(user.name)  # "Alice"
        >>> print(user.age)   # 30

        >>> # Using a remote model
        >>> user = generate_pydantic(
        ...     "Create user Bob age 25",
        ...     User,
        ...     model="openai:gpt-4o-mini",
        ...     unsafe_mode=True
        ... )
    """
    # AIDEV-NOTE: Validate that model_class is a Pydantic BaseModel
    if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
        raise ValueError(
            f"model_class must be a Pydantic BaseModel class, got {type(model_class)}"
        )

    # Call generate_json with return_pydantic=True
    result = generate_json(
        prompt=prompt,
        schema=model_class,
        max_tokens=max_tokens,
        model=model,
        unsafe_mode=unsafe_mode,
        seed=seed,
        return_pydantic=True,
        **kwargs,
    )

    # AIDEV-NOTE: Result should always be a Pydantic model instance at this point
    # but we assert for type safety
    assert isinstance(result, BaseModel)
    return result
