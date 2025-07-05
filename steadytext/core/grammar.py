"""Grammar-based structured generation using llama.cpp's native GBNF support.

This module provides JSON schema to GBNF (Backus-Naur Form) grammar conversion
for structured text generation without requiring Outlines.

AIDEV-NOTE: This implementation is based on llama.cpp's json_schema_to_grammar.py
and provides native grammar support for Gemma-3n and other models that have
compatibility issues with Outlines.
"""

import re
from typing import Any, Dict, List, Set, Union


class GrammarConverter:
    """Converts JSON schemas and patterns to GBNF grammars for llama.cpp."""

    def __init__(self):
        """Initialize the grammar converter."""
        self._primitive_rules = {
            "boolean": 'boolean ::= "true" | "false"',
            "number": 'number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?',
            "integer": 'integer ::= "-"? ([0-9] | [1-9] [0-9]*)',
            "string": 'string ::= "\\"" ([^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""',
            "null": 'null ::= "null"',
        }

    def json_schema_to_gbnf(self, schema: Dict[str, Any]) -> str:
        """Convert a JSON schema to GBNF grammar.

        Args:
            schema: JSON schema dictionary

        Returns:
            GBNF grammar string
        """
        rules = []
        self._defined_rules: Set[str] = set()
        self._additional_rules: List[
            str
        ] = []  # Collect rules generated during traversal

        # Add primitive rules
        for name, rule in self._primitive_rules.items():
            rules.append(rule)
            self._defined_rules.add(name)

        # Add whitespace rule
        rules.append("ws ::= [ \\t\\n]*")
        self._defined_rules.add("ws")

        # Generate root rule
        root_rule = self._generate_rule("root", schema)
        rules.insert(0, root_rule)

        # Add any additional rules that were generated
        rules.extend(self._additional_rules)

        return "\n".join(rules)

    def _generate_rule(self, name: str, schema: Dict[str, Any]) -> str:
        """Generate a GBNF rule for a schema component.

        Args:
            name: Rule name
            schema: Schema component

        Returns:
            GBNF rule string
        """
        schema_type = schema.get("type")

        if schema_type == "object":
            return self._generate_object_rule(name, schema)
        elif schema_type == "array":
            return self._generate_array_rule(name, schema)
        elif schema_type == "string":
            if "enum" in schema:
                return self._generate_enum_rule(name, schema["enum"])
            elif "pattern" in schema:
                # AIDEV-NOTE: Pattern support is limited in GBNF
                # For now, fall back to generic string
                return f"{name} ::= string"
            else:
                return f"{name} ::= string"
        elif schema_type == "number":
            return f"{name} ::= number"
        elif schema_type == "integer":
            return f"{name} ::= integer"
        elif schema_type == "boolean":
            return f"{name} ::= boolean"
        elif schema_type == "null":
            return f"{name} ::= null"
        else:
            # Handle union types or missing type
            if "anyOf" in schema:
                return self._generate_union_rule(name, schema["anyOf"])
            elif "oneOf" in schema:
                return self._generate_union_rule(name, schema["oneOf"])
            else:
                # Default to any JSON value
                return f"{name} ::= object | array | string | number | boolean | null"

    def _generate_object_rule(self, name: str, schema: Dict[str, Any]) -> str:
        """Generate GBNF rule for object type.

        Args:
            name: Rule name
            schema: Object schema

        Returns:
            GBNF rule string
        """
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        additional_properties = schema.get("additionalProperties", True)

        if not properties and additional_properties:
            # Empty object or any object
            return f'{name} ::= "{{" ws "}}" | "{{" ws {name}_kv ("," ws {name}_kv)* ws "}}"'

        # Build property rules
        prop_kv_rules = []
        for prop_name, prop_schema in properties.items():
            prop_rule_name = f"{name}_{self._sanitize_name(prop_name)}"
            if prop_rule_name not in self._defined_rules:
                prop_rule = self._generate_rule(prop_rule_name, prop_schema)
                self._defined_rules.add(prop_rule_name)
                # Add the rule to the additional rules collection
                self._additional_rules.append(prop_rule)

            # Create key-value rule for this property
            kv_rule = f'{name}_{self._sanitize_name(prop_name)}_kv ::= "\\"" "{prop_name}" "\\"" ws ":" ws {prop_rule_name}'
            self._additional_rules.append(kv_rule)

            if prop_name in required:
                prop_kv_rules.append(f"{name}_{self._sanitize_name(prop_name)}_kv")

        # Build the object rule
        if properties:
            # Fixed order for required properties
            required_props = [
                f"{name}_{self._sanitize_name(p)}_kv"
                for p in properties.keys()
                if p in required
            ]
            optional_props = [
                f"{name}_{self._sanitize_name(p)}_kv"
                for p in properties.keys()
                if p not in required
            ]

            rule_parts = ['"{"']
            if required_props:
                joiner = ' ws "," ws '
                rule_parts.append(f" ws {joiner.join(required_props)}")
            if optional_props:
                # AIDEV-NOTE: Simplified optional property handling
                # Full implementation would need more complex grammar
                pass
            rule_parts.append(' ws "}"')

            return f"{name} ::= {''.join(rule_parts)}"
        else:
            return f'{name} ::= "{{" ws "}}"'

    def _generate_array_rule(self, name: str, schema: Dict[str, Any]) -> str:
        """Generate GBNF rule for array type.

        Args:
            name: Rule name
            schema: Array schema

        Returns:
            GBNF rule string
        """
        items = schema.get("items", {})
        item_rule_name = f"{name}_item"

        if item_rule_name not in self._defined_rules:
            item_rule = self._generate_rule(item_rule_name, items)
            self._defined_rules.add(item_rule_name)
            self._additional_rules.append(item_rule)

        return f'{name} ::= "[" ws "]" | "[" ws {item_rule_name} ("," ws {item_rule_name})* ws "]"'

    def _generate_enum_rule(self, name: str, values: List[Any]) -> str:
        """Generate GBNF rule for enum values.

        Args:
            name: Rule name
            values: List of enum values

        Returns:
            GBNF rule string
        """
        # Convert all values to quoted strings
        options = [f'"\\"" "{str(v)}" "\\""' for v in values]
        return f"{name} ::= {' | '.join(options)}"

    def _generate_union_rule(self, name: str, schemas: List[Dict[str, Any]]) -> str:
        """Generate GBNF rule for union types.

        Args:
            name: Rule name
            schemas: List of schemas in the union

        Returns:
            GBNF rule string
        """
        options = []
        for i, schema in enumerate(schemas):
            option_name = f"{name}_option{i}"
            if option_name not in self._defined_rules:
                option_rule = self._generate_rule(option_name, schema)
                self._defined_rules.add(option_name)
                self._additional_rules.append(option_rule)
            options.append(option_name)

        return f"{name} ::= {' | '.join(options)}"

    def _sanitize_name(self, name: str) -> str:
        """Sanitize property names for use in GBNF rules.

        Args:
            name: Property name

        Returns:
            Sanitized name
        """
        # Replace non-alphanumeric characters with underscores
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    def regex_to_gbnf(self, pattern: str) -> str:
        """Convert a simple regex pattern to GBNF grammar.

        Args:
            pattern: Regular expression pattern

        Returns:
            GBNF grammar string

        Note:
            This only supports a subset of regex patterns that can be
            represented in GBNF. Complex patterns may not convert correctly.
        """
        # AIDEV-NOTE: This is a simplified implementation
        # Full regex to GBNF conversion would be quite complex

        # Handle some common patterns
        if pattern == r"\d+":
            return "root ::= [0-9]+"
        elif pattern == r"\d{3}-\d{3}-\d{4}":
            # Phone number pattern
            return 'root ::= [0-9] [0-9] [0-9] "-" [0-9] [0-9] [0-9] "-" [0-9] [0-9] [0-9] [0-9]'
        elif pattern == r"[A-Z]{2}\d{4}":
            # License plate pattern
            return "root ::= [A-Z] [A-Z] [0-9] [0-9] [0-9] [0-9]"
        elif pattern == r"[0-9]+":
            # Alternative for digit sequences
            return "root ::= [0-9]+"
        elif pattern == r"[a-zA-Z]+":
            # Letters only
            return "root ::= [a-zA-Z]+"
        elif pattern == r"[a-z]+":
            # Lowercase letters
            return "root ::= [a-z]+"
        elif pattern == r"[A-Z]+":
            # Uppercase letters
            return "root ::= [A-Z]+"
        elif pattern == r"\w+":
            # Word characters (letters, digits, underscore)
            return "root ::= [a-zA-Z0-9_]+"
        elif pattern == r"[0-9]{4}":
            # 4-digit year/PIN
            return "root ::= [0-9] [0-9] [0-9] [0-9]"
        elif pattern == r"[0-9]{2}":
            # 2-digit number
            return "root ::= [0-9] [0-9]"
        elif pattern == r"[a-zA-Z0-9]+":
            # Alphanumeric
            return "root ::= [a-zA-Z0-9]+"
        elif pattern == r"(true|false)":
            # Boolean
            return 'root ::= "true" | "false"'
        elif pattern == r"(yes|no)":
            # Yes/No
            return 'root ::= "yes" | "no"'
        else:
            # For unsupported patterns, fall back to generic string
            # and log a warning
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Complex regex pattern '{pattern}' cannot be fully converted to GBNF. "
                "Using generic string rule."
            )
            return 'root ::= string\nstring ::= "\\"" ([^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""'

    def choices_to_gbnf(self, choices: List[str]) -> str:
        """Convert a list of choices to GBNF grammar.

        Args:
            choices: List of string choices

        Returns:
            GBNF grammar string
        """
        # Quote each choice
        options = [f'"{choice}"' for choice in choices]
        return f"root ::= {' | '.join(options)}"


# AIDEV-NOTE: Singleton instance for reuse
_grammar_converter = GrammarConverter()


def json_schema_to_grammar(schema: Union[Dict[str, Any], type]) -> str:
    """Convert a JSON schema or Python type to GBNF grammar.

    Args:
        schema: JSON schema dict or Python type

    Returns:
        GBNF grammar string
    """
    if isinstance(schema, type):
        # Convert Python type to JSON schema
        if schema is int:
            schema = {"type": "integer"}
        elif schema is float:
            schema = {"type": "number"}
        elif schema is str:
            schema = {"type": "string"}
        elif schema is bool:
            schema = {"type": "boolean"}
        else:
            raise ValueError(f"Unsupported Python type: {schema}")

    return _grammar_converter.json_schema_to_gbnf(schema)


def regex_to_grammar(pattern: str) -> str:
    """Convert a regex pattern to GBNF grammar.

    Args:
        pattern: Regular expression pattern

    Returns:
        GBNF grammar string
    """
    return _grammar_converter.regex_to_gbnf(pattern)


def choices_to_grammar(choices: List[str]) -> str:
    """Convert a list of choices to GBNF grammar.

    Args:
        choices: List of string choices

    Returns:
        GBNF grammar string
    """
    return _grammar_converter.choices_to_gbnf(choices)
