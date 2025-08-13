"""Tests for GBNF grammar generation.

AIDEV-NOTE: These tests verify the GBNF grammar converter functionality,
specifically testing the fixes for quote escaping in property names and enum values.
"""

import pytest
from steadytext.core.grammar import GrammarConverter


class TestGrammarConverter:
    """Test GBNF grammar conversion functionality."""

    def test_json_object_property_escaping(self):
        """Test that JSON object properties are correctly escaped in GBNF rules.
        
        This test verifies the fix for issue #89 where property names were incorrectly
        escaped as '\"\" \"property\" \"\"' instead of '\"property\"'.
        """
        converter = GrammarConverter()
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
        
        grammar = converter.json_schema_to_gbnf(schema)
        
        # Check that property rules are correctly formatted
        # Should be: root_name_kv ::= "\"name\"" ws ":" ws string
        # Not: root_name_kv ::= "\"" "name" "\"" ws ":" ws string
        assert '"\\\"name\\\""' in grammar or '"\\"name\\""' in grammar
        assert '"\\\"age\\\""' in grammar or '"\\"age\\""' in grammar
        
        # Make sure the incorrect format is NOT present
        assert '"\\\"" "name" "\\\""' not in grammar
        assert '"\\"" "name" "\\""' not in grammar
        assert '"\\\"" "age" "\\\""' not in grammar
        assert '"\\"" "age" "\\""' not in grammar
        
    def test_enum_value_escaping(self):
        """Test that enum values are correctly escaped in GBNF rules.
        
        This test verifies the fix for enum value escaping where values were incorrectly
        escaped as '\"\" \"value\" \"\"' instead of '\"value\"'.
        """
        converter = GrammarConverter()
        
        schema = {
            "type": "string",
            "enum": ["red", "green", "blue"]
        }
        
        grammar = converter.json_schema_to_gbnf(schema)
        
        # Check that enum values are correctly formatted
        # Should be: root ::= "\"red\"" | "\"green\"" | "\"blue\""
        # Not: root ::= "\"" "red" "\"" | "\"" "green" "\"" | "\"" "blue" "\""
        assert '"\\\"red\\\""' in grammar or '"\\"red\\""' in grammar
        assert '"\\\"green\\\""' in grammar or '"\\"green\\""' in grammar
        assert '"\\\"blue\\\""' in grammar or '"\\"blue\\""' in grammar
        
        # Make sure the incorrect format is NOT present
        assert '"\\\"" "red" "\\\""' not in grammar
        assert '"\\"" "red" "\\""' not in grammar
        assert '"\\\"" "green" "\\\""' not in grammar
        assert '"\\"" "green" "\\""' not in grammar
        
    def test_complex_object_schema(self):
        """Test grammar generation for a complex object schema."""
        converter = GrammarConverter()
        
        schema = {
            "type": "object",
            "properties": {
                "firstName": {"type": "string"},
                "lastName": {"type": "string"},
                "age": {"type": "integer"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "country": {"type": "string"}
                    },
                    "required": ["street", "city"]
                },
                "phoneNumbers": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["firstName", "lastName"]
        }
        
        grammar = converter.json_schema_to_gbnf(schema)
        
        # Verify that all property names are correctly escaped
        assert '"\\\"firstName\\\""' in grammar or '"\\"firstName\\""' in grammar
        assert '"\\\"lastName\\\""' in grammar or '"\\"lastName\\""' in grammar
        assert '"\\\"street\\\""' in grammar or '"\\"street\\""' in grammar
        assert '"\\\"city\\\""' in grammar or '"\\"city\\""' in grammar
        
        # Verify incorrect escaping is not present
        assert '"\\\"" "firstName" "\\\""' not in grammar
        assert '"\\"" "firstName" "\\""' not in grammar
        
    def test_primitive_types(self):
        """Test that primitive type rules are included in the grammar."""
        converter = GrammarConverter()
        
        schema = {"type": "string"}
        grammar = converter.json_schema_to_gbnf(schema)
        
        # Check that primitive rules are present
        assert 'boolean ::=' in grammar
        assert 'number ::=' in grammar
        assert 'integer ::=' in grammar
        assert 'string ::=' in grammar
        assert 'null ::=' in grammar
        assert 'ws ::=' in grammar
        
    def test_array_with_enum_items(self):
        """Test grammar generation for arrays containing enum values."""
        converter = GrammarConverter()
        
        schema = {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["small", "medium", "large"]
            }
        }
        
        grammar = converter.json_schema_to_gbnf(schema)
        
        # Verify enum values are correctly escaped
        assert '"\\\"small\\\""' in grammar or '"\\"small\\""' in grammar
        assert '"\\\"medium\\\""' in grammar or '"\\"medium\\""' in grammar
        assert '"\\\"large\\\""' in grammar or '"\\"large\\""' in grammar
        
        # Verify incorrect escaping is not present
        assert '"\\\"" "small" "\\\""' not in grammar
        assert '"\\"" "small" "\\""' not in grammar
        
    def test_nested_object_property_escaping(self):
        """Test that nested object properties are correctly escaped."""
        converter = GrammarConverter()
        
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "email": {"type": "string"}
                    },
                    "required": ["id"]
                }
            },
            "required": ["user"]
        }
        
        grammar = converter.json_schema_to_gbnf(schema)
        
        # Check nested properties are correctly escaped
        assert '"\\\"user\\\""' in grammar or '"\\"user\\""' in grammar
        assert '"\\\"id\\\""' in grammar or '"\\"id\\""' in grammar
        assert '"\\\"email\\\""' in grammar or '"\\"email\\""' in grammar
        
        # Verify incorrect escaping is not present
        assert '"\\\"" "user" "\\\""' not in grammar
        assert '"\\"" "user" "\\""' not in grammar
        assert '"\\\"" "id" "\\\""' not in grammar
        assert '"\\"" "id" "\\""' not in grammar