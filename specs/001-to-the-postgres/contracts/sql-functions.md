# SQL Function Contracts: Prompt Registry

## Core Functions

### steadytext_prompt_create
Creates a new prompt with its first version.

**Signature**:
```sql
steadytext_prompt_create(
    slug TEXT,
    template TEXT,
    description TEXT DEFAULT NULL,
    metadata JSONB DEFAULT '{}'
) RETURNS UUID
```

**Parameters**:
- `slug`: Unique identifier for the prompt (lowercase, hyphenated)
- `template`: Jinja2 template content
- `description`: Optional description of the prompt
- `metadata`: Optional JSON metadata

**Returns**: UUID of the created prompt

**Errors**:
- `unique_violation`: If slug already exists
- `invalid_text_representation`: If slug format invalid
- `syntax_error`: If template has invalid Jinja2 syntax

---

### steadytext_prompt_update
Creates a new version of an existing prompt.

**Signature**:
```sql
steadytext_prompt_update(
    slug TEXT,
    template TEXT,
    metadata JSONB DEFAULT '{}'
) RETURNS UUID
```

**Parameters**:
- `slug`: Existing prompt slug
- `template`: New template content
- `metadata`: Optional version metadata

**Returns**: UUID of the new version

**Errors**:
- `no_data_found`: If slug doesn't exist
- `syntax_error`: If template has invalid Jinja2 syntax

---

### steadytext_prompt_get
Retrieves a prompt template.

**Signature**:
```sql
steadytext_prompt_get(
    slug TEXT,
    version INTEGER DEFAULT NULL
) RETURNS TABLE(
    prompt_id UUID,
    version INTEGER,
    template TEXT,
    required_variables TEXT[],
    metadata JSONB,
    created_at TIMESTAMPTZ,
    created_by TEXT
)
```

**Parameters**:
- `slug`: Prompt slug to retrieve
- `version`: Optional specific version (NULL for latest)

**Returns**: Single row with prompt details

**Errors**:
- `no_data_found`: If slug or version doesn't exist

---

### steadytext_prompt_render
Renders a prompt template with variables.

**Signature**:
```sql
steadytext_prompt_render(
    slug TEXT,
    variables JSONB,
    version INTEGER DEFAULT NULL,
    strict BOOLEAN DEFAULT TRUE
) RETURNS TEXT
```

**Parameters**:
- `slug`: Prompt slug to render
- `variables`: JSON object with variable values
- `version`: Optional specific version (NULL for latest)
- `strict`: If true, error on missing variables; if false, leave undefined

**Returns**: Rendered text with variables substituted

**Errors**:
- `no_data_found`: If slug or version doesn't exist
- `invalid_parameter_value`: If required variables missing (strict mode)
- `data_exception`: If template rendering fails

---

### steadytext_prompt_list
Lists all prompts with their latest version info.

**Signature**:
```sql
steadytext_prompt_list() RETURNS TABLE(
    prompt_id UUID,
    slug TEXT,
    description TEXT,
    latest_version INTEGER,
    total_versions INTEGER,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
)
```

**Returns**: Set of rows, one per prompt

---

### steadytext_prompt_versions
Lists all versions of a specific prompt.

**Signature**:
```sql
steadytext_prompt_versions(
    slug TEXT
) RETURNS TABLE(
    version INTEGER,
    template TEXT,
    required_variables TEXT[],
    metadata JSONB,
    created_at TIMESTAMPTZ,
    created_by TEXT,
    is_active BOOLEAN
)
```

**Parameters**:
- `slug`: Prompt slug to get versions for

**Returns**: Set of rows, one per version, ordered by version DESC

**Errors**:
- `no_data_found`: If slug doesn't exist

---

### steadytext_prompt_delete
Deletes a prompt and all its versions.

**Signature**:
```sql
steadytext_prompt_delete(
    slug TEXT
) RETURNS BOOLEAN
```

**Parameters**:
- `slug`: Prompt slug to delete

**Returns**: TRUE if deleted, FALSE if not found

---

## Convenience Aliases

All functions will have short aliases with `st_` prefix:
- `st_prompt_create()`
- `st_prompt_update()`
- `st_prompt_get()`
- `st_prompt_render()`
- `st_prompt_list()`
- `st_prompt_versions()`
- `st_prompt_delete()`

## Internal Helper Functions

### _validate_jinja2_template
Internal function to validate template syntax.

**Signature**:
```sql
_validate_jinja2_template(
    template TEXT
) RETURNS TABLE(
    is_valid BOOLEAN,
    required_variables TEXT[],
    error_message TEXT
)
```

**Used by**: create and update functions

---

### _get_next_version
Internal function to get next version number.

**Signature**:
```sql
_get_next_version(
    prompt_id UUID
) RETURNS INTEGER
```

**Used by**: update function