"""Grammar support for llama.cpp structured generation.

This module provides utilities for converting JSON schemas, regex patterns,
and choice constraints to GBNF (BNF-like) grammars that can be used with
llama.cpp for constrained generation.

AIDEV-NOTE: This replaces Outlines which has compatibility issues with Gemma-3n models.
Based on: https://github.com/ggml-org/llama.cpp/blob/master/examples/json_schema_to_grammar.py
"""

import json
import re
from typing import Any, Dict, List, Union, Optional, Type, Set, Tuple
import logging

logger = logging.getLogger(__name__)

# AIDEV-NOTE: Basic GBNF grammar components for JSON generation
GBNF_GRAMMAR_CONSTANTS = {
    "boolean": 'boolean ::= "true" | "false"',
    "null": 'null ::= "null"',
    "ws": 'ws ::= [ \t\n]*',
    "string": r'''string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""''',
    "number": '''number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?''',
    "integer": '''integer ::= "-"? ([0-9] | [1-9] [0-9]*)''',
    "value": "value ::= object | array | string | number | boolean | null",
    "object": 'object ::= "{" ws ( string ws ":" ws value ws ("," ws string ws ":" ws value ws)* )? "}"',
    "array": 'array ::= "[" ws ( value ws ("," ws value ws)* )? "]"',
}


class GrammarConverter:
    """Converts various schema types to GBNF grammar for llama.cpp."""

    def __init__(self):
        """Initialize the grammar converter."""
        self.rules: Dict[str, str] = {}
        self.rule_counter = 0

    def _get_next_rule_name(self, prefix: str = "rule") -> str:
        """Generate a unique rule name."""
        name = f"{prefix}{self.rule_counter}"
        self.rule_counter += 1
        return name

    def _add_rule(self, name: str, definition: str) -> str:
        """Add a rule to the grammar and return its name."""
        self.rules[name] = definition
        return name

    def json_schema_to_gbnf(self, schema: Dict[str, Any]) -> str:
        """Convert a JSON schema to GBNF grammar.

        Args:
            schema: JSON schema dictionary

        Returns:
            GBNF grammar string

        AIDEV-NOTE: This handles basic JSON schema features including:
        - object with properties
        - arrays
        - primitive types (string, number, integer, boolean, null)
        - required fields
        - enums
        """
        self.rules = {}
        self.rule_counter = 0

        # Start with the root rule
        root_rule = self._process_schema(schema, "root")
        
        # Add basic rules that might be referenced
        grammar_parts = [f"root ::= {root_rule}"]
        
        # Add all generated rules
        for name, definition in self.rules.items():
            grammar_parts.append(f"{name} ::= {definition}")
        
        # Add constants if they're used
        used_constants = set()
        full_grammar = "\n".join(grammar_parts)
        
        if "string" in full_grammar and "string ::=" not in full_grammar:
            used_constants.add("string")
        if "number" in full_grammar and "number ::=" not in full_grammar:
            used_constants.add("number")
        if "integer" in full_grammar and "integer ::=" not in full_grammar:
            used_constants.add("integer")
        if "boolean" in full_grammar and "boolean ::=" not in full_grammar:
            used_constants.add("boolean")
        if "null" in full_grammar and "null ::=" not in full_grammar:
            used_constants.add("null")
        if "ws" in full_grammar:
            used_constants.add("ws")
        
        # Add used constants
        for const in ["ws", "string", "number", "integer", "boolean", "null"]:
            if const in used_constants:
                grammar_parts.append(GBNF_GRAMMAR_CONSTANTS[const])
        
        return "\n".join(grammar_parts)

    def _process_schema(self, schema: Dict[str, Any], rule_name: str) -> str:
        """Process a schema component and return its rule reference."""
        schema_type = schema.get("type", "object")
        
        if "enum" in schema:
            # Handle enums
            return self._process_enum(schema["enum"])
        
        if schema_type == "object":
            return self._process_object(schema, rule_name)
        elif schema_type == "array":
            return self._process_array(schema, rule_name)
        elif schema_type == "string":
            return "string"
        elif schema_type == "number":
            return "number"
        elif schema_type == "integer":
            return "integer"
        elif schema_type == "boolean":
            return "boolean"
        elif schema_type == "null":
            return "null"
        else:
            # Default to string for unknown types
            logger.warning(f"Unknown schema type: {schema_type}, defaulting to string")
            return "string"

    def _process_object(self, schema: Dict[str, Any], rule_name: str) -> str:
        """Process an object schema."""
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        
        if not properties:
            # Empty object
            return '"{" ws "}"'
        
        # Build the object rule
        prop_rules = []
        
        # Process required properties first
        for prop_name in required:
            if prop_name in properties:
                prop_schema = properties[prop_name]
                prop_rule = self._process_schema(prop_schema, f"{rule_name}_{prop_name}")
                prop_rules.append(f'"\\"" "{prop_name}" "\\"" ws ":" ws {prop_rule}')
        
        # Process optional properties
        optional_props = []
        for prop_name, prop_schema in properties.items():
            if prop_name not in required:
                prop_rule = self._process_schema(prop_schema, f"{rule_name}_{prop_name}")
                optional_props.append(f'"\\"" "{prop_name}" "\\"" ws ":" ws {prop_rule}')
        
        # Build the complete object rule
        if prop_rules and optional_props:
            # Has both required and optional
            required_part = " ws \",\" ws ".join(prop_rules)
            optional_part = " | ".join(f"({opt})" for opt in optional_props)
            object_def = f'"{{{" ws {required_part} ws (\",\" ws ({optional_part}) ws)* "}}"'
        elif prop_rules:
            # Only required properties
            props_part = " ws \",\" ws ".join(prop_rules)
            object_def = f'"{{{" ws {props_part} ws "}}"'
        else:
            # Only optional properties
            optional_part = " | ".join(f"({opt})" for opt in optional_props)
            object_def = f'"{{{" ws (({optional_part}) ws (\",\" ws ({optional_part}) ws)*)? "}}"'
        
        return object_def

    def _process_array(self, schema: Dict[str, Any], rule_name: str) -> str:
        """Process an array schema."""
        items_schema = schema.get("items", {"type": "string"})
        item_rule = self._process_schema(items_schema, f"{rule_name}_item")
        
        # Build array rule
        return f'"[" ws ({item_rule} ws ("," ws {item_rule} ws)*)? "]"'

    def _process_enum(self, enum_values: List[Any]) -> str:
        """Process enum values."""
        if not enum_values:
            return '""'  # Empty string as fallback
        
        # Convert all values to strings and escape quotes
        escaped_values = []
        for val in enum_values:
            if isinstance(val, str):
                escaped = val.replace('"', '\\"')
                escaped_values.append(f'"\\"" "{escaped}" "\\""')
            else:
                # Convert to JSON string
                escaped_values.append(f'"{json.dumps(val)}"')
        
        return " | ".join(escaped_values)

    def regex_to_gbnf(self, pattern: str) -> str:
        """Convert a regex pattern to GBNF grammar.

        Args:
            pattern: Regular expression pattern

        Returns:
            GBNF grammar string

        AIDEV-NOTE: This is a simplified converter that handles common regex patterns.
        Complex regex features may not be fully supported.
        """
        # Reset state
        self.rules = {}
        self.rule_counter = 0
        
        # Convert regex to GBNF
        gbnf_pattern = self._convert_regex_to_gbnf(pattern)
        
        grammar_parts = [f"root ::= {gbnf_pattern}"]
        
        # Add any generated rules
        for name, definition in self.rules.items():
            grammar_parts.append(f"{name} ::= {definition}")
        
        return "\n".join(grammar_parts)

    def _convert_regex_to_gbnf(self, pattern: str) -> str:
        """Convert a regex pattern to GBNF notation.
        
        AIDEV-NOTE: This handles basic regex patterns. Complex features like
        lookahead/lookbehind, backreferences, etc. are not supported.
        """
        # Handle basic patterns
        if pattern == r"\d+":
            return '[0-9]+'
        elif pattern == r"\d{3}-\d{3}-\d{4}":
            # Phone number pattern
            return '[0-9] [0-9] [0-9] "-" [0-9] [0-9] [0-9] "-" [0-9] [0-9] [0-9] [0-9]'
        elif pattern == r"[a-z]+@[a-z]+\.[a-z]+":
            # Simple email pattern
            return '[a-z]+ "@" [a-z]+ "." [a-z]+'
        elif pattern.startswith("^") and pattern.endswith("$"):
            # Remove anchors
            return self._convert_regex_to_gbnf(pattern[1:-1])
        else:
            # Try to handle character classes and basic patterns
            gbnf = pattern
            
            # Replace \d with [0-9]
            gbnf = gbnf.replace(r"\d", "[0-9]")
            
            # Replace \w with [a-zA-Z0-9_]
            gbnf = gbnf.replace(r"\w", "[a-zA-Z0-9_]")
            
            # Replace \s with [ \t\n]
            gbnf = gbnf.replace(r"\s", "[ \t\n]")
            
            # Handle quantifiers
            gbnf = re.sub(r"\{(\d+)\}", lambda m: f' {m.group(1)}', gbnf)
            gbnf = re.sub(r"\{(\d+),(\d+)\}", lambda m: f' {m.group(1)}-{m.group(2)}', gbnf)
            
            # Quote literal strings
            # This is a simplified approach - a full implementation would need
            # proper parsing
            if not any(c in gbnf for c in ['[', ']', '|', '+', '*', '?']):
                # If it's just a literal string, quote it
                gbnf = f'"{gbnf}"'
            
            return gbnf

    def choices_to_gbnf(self, choices: List[str]) -> str:
        """Convert a list of choices to GBNF grammar.

        Args:
            choices: List of string choices

        Returns:
            GBNF grammar string
        """
        if not choices:
            return 'root ::= ""'
        
        # Escape quotes in choices and create alternatives
        escaped_choices = []
        for choice in choices:
            escaped = choice.replace('"', '\\"')
            escaped_choices.append(f'"{escaped}"')
        
        return f"root ::= {' | '.join(escaped_choices)}"

    def pydantic_to_json_schema(self, model_class: Type) -> Dict[str, Any]:
        """Convert a Pydantic model to JSON schema.

        Args:
            model_class: Pydantic model class

        Returns:
            JSON schema dictionary

        AIDEV-NOTE: This uses Pydantic's built-in schema generation.
        """
        try:
            # Pydantic v2
            if hasattr(model_class, "model_json_schema"):
                return model_class.model_json_schema()
            # Pydantic v1
            elif hasattr(model_class, "schema"):
                return model_class.schema()
            else:
                raise ValueError(f"Cannot extract schema from {model_class}")
        except Exception as e:
            logger.error(f"Failed to convert Pydantic model to schema: {e}")
            raise