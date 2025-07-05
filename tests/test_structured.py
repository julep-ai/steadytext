"""Tests for structured text generation.

AIDEV-NOTE: These tests verify the structured generation capabilities including
JSON schemas, regex patterns, and choice constraints.

AIDEV-NOTE: All structured generation tests are skipped when STEADYTEXT_SKIP_MODEL_LOAD=1
because Outlines requires an actual model instance to function properly. These tests
run when STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true is set.
"""

import json
import os
import pytest
import re
from typing import List, Optional
from unittest.mock import patch

from steadytext import (
    generate,
    generate_json,
    generate_regex,
    generate_choice,
    generate_format,
)
from steadytext.exceptions import ContextLengthExceededError

from pydantic import BaseModel


# Test models for structured generation


class Person(BaseModel):
    name: str
    age: int
    hobbies: Optional[List[str]] = None


class ColorList(BaseModel):
    colors: List[str]


@pytest.mark.skipif(
    os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1",
    reason="Structured generation requires model loading (set STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true to run)",
)
class TestStructuredGeneration:
    """Test structured text generation features."""

    def test_generate_with_json_schema(self):
        """Test generation with JSON schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["name", "count"],
        }

        result = generate("Create an item", schema=schema)

        # Check that result is not None
        assert result is not None
        assert isinstance(result, str)

        # Check that result contains JSON output tags
        assert "<json-output>" in result
        assert "</json-output>" in result

        # Extract and validate JSON
        json_start = result.find("<json-output>") + len("<json-output>")
        json_end = result.find("</json-output>")
        json_str = result[json_start:json_end]

        # Should be valid JSON
        data = json.loads(json_str)
        assert isinstance(data, dict)
        assert "name" in data
        assert "count" in data
        assert isinstance(data["name"], str)
        assert isinstance(data["count"], int)

    def test_generate_with_pydantic_model(self):
        """Test generation with Pydantic model."""
        result = generate("Create a person named Alice who is 30", schema=Person)

        # Check result is valid
        assert result is not None
        assert isinstance(result, str)

        # Check structure
        assert "<json-output>" in result
        assert "</json-output>" in result

        # Extract JSON
        json_start = result.find("<json-output>") + len("<json-output>")
        json_end = result.find("</json-output>")
        json_str = result[json_start:json_end]

        # Validate with Pydantic
        person = Person.model_validate_json(json_str)
        assert isinstance(person.name, str)
        assert isinstance(person.age, int)

    def test_generate_with_regex(self):
        """Test generation with regex pattern."""
        # Phone number pattern
        pattern = r"\d{3}-\d{3}-\d{4}"
        result = generate("My phone number is", regex=pattern)

        # Result should match the pattern
        assert result is not None
        assert isinstance(result, str)
        assert re.match(pattern, result) is not None

    def test_generate_with_choices(self):
        """Test generation with choice constraints."""
        choices = ["yes", "no", "maybe"]
        result = generate("Is Python a good language?", choices=choices)

        # Result should be one of the choices
        assert result is not None
        assert isinstance(result, str)
        assert result in choices

    def test_generate_with_response_format(self):
        """Test generation with response_format parameter."""
        result = generate("List three fruits", response_format={"type": "json_object"})

        # Should generate JSON
        assert result is not None
        assert isinstance(result, str)
        assert "<json-output>" in result
        assert "</json-output>" in result

    def test_generate_json_function(self):
        """Test the generate_json convenience function."""
        schema = {"type": "array", "items": {"type": "string"}}
        result = generate_json("List colors", schema)

        # Check structure
        assert "<json-output>" in result
        assert "</json-output>" in result

        # Extract and validate JSON
        json_start = result.find("<json-output>") + len("<json-output>")
        json_end = result.find("</json-output>")
        json_str = result[json_start:json_end]

        data = json.loads(json_str)
        assert isinstance(data, list)
        assert all(isinstance(item, str) for item in data)

    def test_generate_regex_function(self):
        """Test the generate_regex convenience function."""
        # Email pattern
        pattern = r"[a-z]+@[a-z]+\.[a-z]+"
        result = generate_regex("Contact me at", pattern)

        assert re.match(pattern, result) is not None

    def test_generate_choice_function(self):
        """Test the generate_choice convenience function."""
        choices = ["red", "green", "blue"]
        result = generate_choice("My favorite color is", choices)

        assert result in choices

    def test_generate_format_function(self):
        """Test the generate_format convenience function."""
        # Integer
        result = generate_format("How many apples?", int)
        assert result.isdigit()

        # Boolean
        result = generate_format("Is the sky blue?", bool)
        assert result in ["True", "False", "true", "false"]

    def test_structured_with_custom_seed(self):
        """Test that structured generation respects seed parameter."""
        schema = {"type": "string"}

        # Generate with same seed should produce same result
        result1 = generate("Say hello", schema=schema, seed=123)
        result2 = generate("Say hello", schema=schema, seed=123)

        assert result1 == result2

        # Different seed should (likely) produce different result
        generate("Say hello", schema=schema, seed=456)
        # Note: This could theoretically be the same, but very unlikely

    def test_structured_with_max_tokens(self):
        """Test structured generation with max_tokens parameter."""
        schema = {"type": "string"}
        result = generate_json("Write a story", schema, max_tokens=50)

        # Should still have valid structure
        assert "<json-output>" in result
        assert "</json-output>" in result

    def test_invalid_regex_pattern(self):
        """Test handling of invalid regex pattern."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            generate("Test", regex="[invalid(")

    def test_empty_choices_list(self):
        """Test handling of empty choices list."""
        with pytest.raises(ValueError, match="Choices list cannot be empty"):
            generate("Choose", choices=[])

    def test_structured_with_logprobs_warning(self):
        """Test that structured generation warns about logprobs."""
        schema = {"type": "string"}

        with patch("steadytext.logger") as mock_logger:
            result = generate("Test", schema=schema, return_logprobs=True)

            # Should warn about logprobs not being supported
            mock_logger.warning.assert_called_with(
                "Structured generation does not support logprobs. Ignoring return_logprobs=True."
            )

            # Should return tuple with None logprobs
            assert isinstance(result, tuple)
            assert result[1] is None

    def test_generate_json_with_pydantic(self):
        """Test generate_json with Pydantic model."""
        result = generate_json("List some colors", schema=ColorList)

        # Check result is valid
        assert result is not None
        assert isinstance(result, str)

        # Extract JSON and validate
        json_start = result.find("<json-output>") + len("<json-output>")
        json_end = result.find("</json-output>")
        json_str = result[json_start:json_end]

        color_list = ColorList.model_validate_json(json_str)
        assert isinstance(color_list.colors, list)
        assert all(isinstance(color, str) for color in color_list.colors)

    def test_generate_with_basic_type(self):
        """Test generation with basic Python types."""
        # Generate an integer
        result = generate("Pick a number between 1 and 10", schema=int)

        # Handle case where model returns None (as per v2.1.0+ fallback removal)
        if result is None:
            raise pytest.skip.Exception("Model not loaded, skipping test")

        assert isinstance(result, str)

        json_start = result.find("<json-output>")
        json_end = result.find("</json-output>")

        # Validate that tags exist
        if json_start == -1 or json_end == -1:
            raise pytest.fail.Exception(
                f"Expected JSON tags not found in result: {result}"
            )

        json_start += len("<json-output>")
        json_str = result[json_start:json_end]

        value = json.loads(json_str)
        assert isinstance(value, int)

    def test_context_length_validation(self):
        """Test that structured generation validates context length."""
        # Create a very long prompt
        long_prompt = "x" * 100000

        with pytest.raises(ContextLengthExceededError):
            generate(long_prompt, schema={"type": "string"})

    def test_generate_json_with_thoughts(self):
        """Test that JSON generation includes thoughts before the output."""
        schema = {"type": "string"}
        result = generate_json("Think about colors then output one", schema)

        # Should have content before <json-output>
        json_start = result.find("<json-output>")
        assert json_start > 0  # There should be thoughts before JSON

        # Thoughts should contain some text
        thoughts = result[:json_start]
        assert len(thoughts.strip()) > 0


# AIDEV-NOTE: Additional tests for edge cases and integration
@pytest.mark.skipif(
    os.environ.get("STEADYTEXT_SKIP_MODEL_LOAD") == "1",
    reason="Structured generation requires model loading (set STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true to run)",
)
class TestStructuredEdgeCases:
    """Test edge cases for structured generation."""

    def test_nested_json_schema(self):
        """Test generation with nested JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                },
                "items": {"type": "array", "items": {"type": "string"}},
            },
        }

        result = generate("Create a user with items", schema=schema)

        # Check result is valid
        assert result is not None
        assert isinstance(result, str)

        # Extract and validate nested structure
        json_start = result.find("<json-output>") + len("<json-output>")
        json_end = result.find("</json-output>")
        json_str = result[json_start:json_end]

        data = json.loads(json_str)
        assert "user" in data
        assert isinstance(data["user"], dict)
        assert "name" in data["user"]
        assert "email" in data["user"]

    def test_regex_with_anchors(self):
        """Test regex patterns with anchors."""
        # Full string match
        pattern = r"^[A-Z]{3}$"
        result = generate_regex("Code:", pattern)

        assert re.match(pattern, result) is not None
        assert len(result) == 3
        assert result.isupper()

    def test_multiple_structured_params_error(self):
        """Test that multiple structured parameters are handled correctly."""
        # Should use schema when multiple are provided
        result = generate(
            "Test", schema={"type": "string"}, regex=r"\d+", choices=["a", "b"]
        )

        # Should prioritize schema
        assert result is not None
        assert isinstance(result, str)
        assert "<json-output>" in result

    def test_format_with_unsupported_type(self):
        """Test format generation with unsupported type."""
        with pytest.raises(ValueError, match="Unsupported format type"):
            generate_format("Test", dict)  # dict is not a supported format type


# AIDEV-NOTE: This test specifically verifies behavior when model loading is disabled
class TestStructuredWithoutModel:
    """Test structured generation behavior when model is not available."""

    def test_structured_without_model(self):
        """Test structured generation when model is not available."""
        with patch.dict(os.environ, {"STEADYTEXT_SKIP_MODEL_LOAD": "1"}):
            # Should raise an error when model is not loaded
            with pytest.raises(RuntimeError, match="Failed to load generation model"):
                generate("Test", schema={"type": "string"})
