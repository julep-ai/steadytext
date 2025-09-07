"""
Prompt Registry Module for pg_steadytext
"""
# AIDEV-ANCHOR: prompt: jinja2 validator
# AIDEV-NOTE: Provides Jinja2 template validation and rendering support

from typing import List, Tuple, Optional, Dict, Any

try:
    from jinja2 import (
        Environment,
        meta,
        TemplateSyntaxError,
        StrictUndefined,
        Undefined,
    )

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


def validate_template(template: str) -> Tuple[bool, Optional[List[str]], Optional[str]]:
    """
    Validate a Jinja2 template and extract required variables.

    AIDEV-NOTE: This function is called from PL/Python to validate templates
    before storing them in the database.

    Args:
        template: The Jinja2 template string to validate

    Returns:
        Tuple of (is_valid, required_variables, error_message)
    """
    if not JINJA2_AVAILABLE:
        return (
            False,
            None,
            "Jinja2 is not installed. Please install it using: pip install jinja2",
        )

    # Use a restricted environment: no loader and minimal builtins
    env = Environment(loader=None, autoescape=False)
    env.globals.clear()
    env.filters.clear()

    try:
        ast = env.parse(template)
        variables = meta.find_undeclared_variables(ast)
        required_vars = sorted(list(variables))
        return (True, required_vars, None)

    except TemplateSyntaxError as e:
        return (False, None, str(e))
    except Exception as e:
        return (False, None, f"Unexpected error: {str(e)}")


def render_template(
    template: str, variables: Dict[str, Any], strict: bool = True
) -> str:
    """
    Render a Jinja2 template with provided variables.

    AIDEV-NOTE: This function handles the actual template rendering
    with support for strict and non-strict modes.

    Args:
        template: The Jinja2 template string
        variables: Dictionary of variables to substitute
        strict: If True, raise error on undefined variables

    Returns:
        The rendered template string

    Raises:
        UndefinedError: If strict mode and variables are missing
        TemplateSyntaxError: If template syntax is invalid
    """
    if not JINJA2_AVAILABLE:
        raise ImportError(
            "Jinja2 is not installed. Please install it using: pip install jinja2"
        )

    # Create Jinja2 environment
    if strict:
        env = Environment(undefined=StrictUndefined)
    else:
        env = Environment(undefined=Undefined)

    # Compile and render template
    compiled_template = env.from_string(template)
    rendered = compiled_template.render(**variables)

    return rendered


def extract_variables(template: str) -> List[str]:
    """
    Extract all variable names from a Jinja2 template.

    AIDEV-NOTE: Utility function to get list of variables without full validation

    Args:
        template: The Jinja2 template string

    Returns:
        List of variable names found in the template
    """
    if not JINJA2_AVAILABLE:
        return []

    try:
        env = Environment()
        ast = env.parse(template)
        variables = meta.find_undeclared_variables(ast)
        return sorted(list(variables))
    except Exception:
        return []


def validate_slug(slug: str) -> bool:
    """
    Validate a prompt slug format.

    AIDEV-NOTE: Slugs must be lowercase letters, numbers, and hyphens only
    Length between 3 and 100 characters

    Args:
        slug: The slug to validate

    Returns:
        True if valid, False otherwise
    """
    import re

    pattern = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
    return bool(re.match(pattern, slug)) and 3 <= len(slug) <= 100


# AIDEV-NOTE: Template cache for performance optimization
# This cache is used by the PL/Python functions to avoid recompiling templates
_template_cache = {}


def get_cached_template(cache_key: str, template: str, strict: bool = True):
    """
    Get a compiled template from cache or compile and cache it.

    AIDEV-NOTE: Used by PL/Python functions to cache compiled templates in GD

    Args:
        cache_key: Unique key for this template version
        template: The template string
        strict: Whether to use strict undefined mode

    Returns:
        Compiled Jinja2 template object
    """
    if not JINJA2_AVAILABLE:
        raise ImportError("Jinja2 is not installed")

    if cache_key not in _template_cache:
        if strict:
            env = Environment(undefined=StrictUndefined)
        else:
            env = Environment(undefined=Undefined)
        _template_cache[cache_key] = env.from_string(template)

    return _template_cache[cache_key]


def clear_template_cache():
    """Clear the template cache."""
    global _template_cache
    _template_cache = {}
