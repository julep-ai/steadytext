# Research: PostgreSQL Prompt Registry

**Feature**: Prompt Registry for pg_steadytext
**Date**: 2025-09-06

## Research Findings

### 1. Jinja2 Dependency in PL/Python3u Environment

**Decision**: Install Jinja2 as part of extension installation process
**Rationale**: 
- pg_steadytext already has a pattern for installing Python dependencies via pip
- The Makefile installs packages to `$(pkglibdir)/pg_steadytext/site-packages`
- Jinja2 is a pure Python package with minimal dependencies
- Extension already handles SteadyText package installation similarly

**Alternatives considered**:
- Embedding Jinja2 source: Too complex, maintenance burden
- Using string.Template: Too limited, no loops/conditionals
- Custom templating: Reinventing the wheel, error-prone

**Implementation approach**:
```python
# In prompt_registry.py
try:
    from jinja2 import Template, Environment, meta, TemplateSyntaxError
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    # Fallback to simple string replacement
```

### 2. Template Validation in PostgreSQL

**Decision**: Validate at insert/update time using Jinja2's parser
**Rationale**:
- Jinja2 provides `Environment().parse()` for syntax validation
- Can extract required variables using `meta.find_undeclared_variables()`
- Fail-fast approach prevents invalid templates from being stored
- Clear error messages help users fix templates immediately

**Alternatives considered**:
- No validation: Could store broken templates
- Validation on render only: Too late, poor user experience
- External validation service: Unnecessary complexity

**Validation approach**:
```python
def validate_template(template_str):
    try:
        env = Environment()
        ast = env.parse(template_str)
        required_vars = meta.find_undeclared_variables(ast)
        return True, list(required_vars)
    except TemplateSyntaxError as e:
        return False, str(e)
```

### 3. Schema Design for Version Tracking

**Decision**: Two-table design with prompts and prompt_versions
**Rationale**:
- Separates identity (prompt) from versions (prompt_versions)
- Allows efficient queries for latest version
- Maintains full history without deletion
- Simple foreign key relationship

**Alternatives considered**:
- Single table with version column: Harder to query latest, duplicated metadata
- Event sourcing pattern: Overkill for this use case
- Soft deletes: Unnecessary complexity

**Schema design**:
```sql
-- Main prompt identity
CREATE TABLE steadytext_prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT DEFAULT current_user
);

-- Immutable version history
CREATE TABLE steadytext_prompt_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_id UUID REFERENCES steadytext_prompts(id),
    version INTEGER NOT NULL,
    template TEXT NOT NULL,
    required_variables TEXT[], -- Array of required variable names
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by TEXT DEFAULT current_user,
    UNIQUE(prompt_id, version)
);
```

### 4. Dynamic Schema Resolution Pattern

**Decision**: Use dynamic schema resolution for all table access
**Rationale**:
- pg_steadytext already uses this pattern successfully
- Required for TimescaleDB continuous aggregate compatibility
- Ensures extension works regardless of installation schema

**Implementation**:
```python
# Get extension schema dynamically
ext_schema_result = plpy.execute("""
    SELECT nspname FROM pg_extension e 
    JOIN pg_namespace n ON e.extnamespace = n.oid 
    WHERE e.extname = 'pg_steadytext'
""")
ext_schema = ext_schema_result[0]['nspname'] if ext_schema_result else 'public'

# Use with plpy.quote_ident for safety
query = f"SELECT * FROM {plpy.quote_ident(ext_schema)}.steadytext_prompts"
```

### 5. Error Handling Strategy

**Decision**: Raise descriptive errors, no silent failures
**Rationale**:
- PostgreSQL users expect clear error messages
- Template errors should be actionable
- Consistency with existing pg_steadytext patterns

**Error types**:
- `TemplateSyntaxError`: Invalid Jinja2 syntax
- `MissingVariableError`: Required variable not provided
- `PromptNotFoundError`: Slug doesn't exist
- `VersionNotFoundError`: Specific version doesn't exist

### 6. Performance Optimization

**Decision**: Cache compiled templates in GD (global dictionary)
**Rationale**:
- Templates are immutable once versioned
- Compilation is the expensive operation
- GD persists across function calls in session
- LRU cache with size limit prevents memory issues

**Implementation**:
```python
# In _steadytext_init_python()
if 'template_cache' not in GD:
    GD['template_cache'] = {}  # Will add LRU logic
```

### 7. Handling Undefined Variables

**Decision**: Strict mode by default, optional lenient mode
**Rationale**:
- User requested clarification on undefined variable handling
- Strict mode prevents silent errors
- Optional lenient mode for flexibility

**Implementation**:
```sql
-- Default strict mode
steadytext_prompt_render(slug, variables) -- errors on missing vars

-- Optional lenient mode
steadytext_prompt_render(slug, variables, strict => false) -- undefined vars become empty
```

## Summary of Decisions

1. **Jinja2 Installation**: Via pip in Makefile, with graceful fallback
2. **Validation**: At insert/update time using Jinja2 parser
3. **Schema**: Two-table design (prompts + versions)
4. **Schema Resolution**: Dynamic resolution pattern from pg_steadytext
5. **Error Handling**: Descriptive errors, no silent failures
6. **Performance**: Template caching in GD
7. **Undefined Variables**: Strict by default, optional lenient mode

All NEEDS CLARIFICATION items from the specification have been resolved.