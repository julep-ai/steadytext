# Data Model: PostgreSQL Prompt Registry

**Feature**: Prompt Registry for pg_steadytext
**Date**: 2025-09-06

## Entity Relationship Diagram

```
steadytext_prompts (1) ----< (N) steadytext_prompt_versions
```

## Entities

### steadytext_prompts
Main prompt identity table that holds the unique slug and metadata for each prompt template.

**Fields**:
- `id` (UUID, PK): Unique identifier, auto-generated
- `slug` (TEXT, UNIQUE, NOT NULL): Human-friendly unique identifier (e.g., "welcome-email", "code-review")
- `description` (TEXT): Optional description of the prompt's purpose
- `created_at` (TIMESTAMPTZ): Timestamp of prompt creation
- `created_by` (TEXT): User who created the prompt
- `updated_at` (TIMESTAMPTZ): Last modification timestamp
- `updated_by` (TEXT): User who last modified metadata

**Constraints**:
- `slug` must be unique across all prompts
- `slug` must match pattern: `^[a-z0-9]+(?:-[a-z0-9]+)*$` (lowercase, numbers, hyphens)
- `slug` length between 3 and 100 characters

**Indexes**:
- Primary key on `id`
- Unique index on `slug`
- Index on `created_at` for time-based queries

### steadytext_prompt_versions
Immutable version history for each prompt template.

**Fields**:
- `id` (UUID, PK): Unique identifier for this version
- `prompt_id` (UUID, FK): Reference to parent prompt
- `version` (INTEGER, NOT NULL): Version number (auto-incremented per prompt)
- `template` (TEXT, NOT NULL): The Jinja2 template content
- `required_variables` (TEXT[]): Array of required variable names extracted from template
- `metadata` (JSONB): Additional metadata (author notes, tags, etc.)
- `created_at` (TIMESTAMPTZ): When this version was created
- `created_by` (TEXT): User who created this version
- `is_active` (BOOLEAN): Whether this is the current active version

**Constraints**:
- Foreign key to `steadytext_prompts(id)` with CASCADE delete
- Unique constraint on `(prompt_id, version)`
- `version` must be > 0
- `template` cannot be empty
- Once created, records are immutable (no updates allowed)

**Indexes**:
- Primary key on `id`
- Unique index on `(prompt_id, version)`
- Index on `(prompt_id, is_active)` for fast latest version lookup
- Index on `created_at` for time-based queries

## Relationships

### prompts → versions (One-to-Many)
- One prompt can have multiple versions
- Versions are immutable once created
- Deleting a prompt cascades to delete all its versions
- Each prompt must have at least one version to be useful

## State Transitions

### Prompt Lifecycle
1. **Created**: New prompt with version 1
2. **Updated**: New version created, previous versions retained
3. **Deleted**: Soft delete via flag or hard delete with CASCADE

### Version Lifecycle
1. **Created**: Immutable once created
2. **Active**: One version per prompt marked as active
3. **Historical**: Previous versions retained for audit

## Validation Rules

### On Insert (steadytext_prompts)
- Slug must be unique
- Slug must match pattern
- Description limited to 500 characters

### On Insert (steadytext_prompt_versions)
- Template must be valid Jinja2 syntax
- Version number auto-generated (max + 1)
- Required variables auto-extracted from template
- Previous active version's is_active set to FALSE

### On Update
- steadytext_prompts: Only description and updated_* fields
- steadytext_prompt_versions: No updates allowed (immutable)

### On Delete
- CASCADE delete removes all versions
- Optional: Prevent deletion if prompt used in production

## Sample Data

```sql
-- Sample prompt
INSERT INTO steadytext_prompts (slug, description) 
VALUES ('code-review', 'Template for code review requests');

-- Sample version
INSERT INTO steadytext_prompt_versions (
    prompt_id, 
    version, 
    template,
    required_variables,
    metadata
) VALUES (
    [prompt_id],
    1,
    'Please review this {{ language }} code:\n\n{{ code }}\n\nFocus on: {{ focus_areas }}',
    ARRAY['language', 'code', 'focus_areas'],
    '{"tags": ["code", "review"], "model": "gpt-4"}'::jsonb
);
```

## Query Patterns

### Get Latest Version
```sql
SELECT pv.* 
FROM steadytext_prompt_versions pv
JOIN steadytext_prompts p ON p.id = pv.prompt_id
WHERE p.slug = 'code-review' 
  AND pv.is_active = true;
```

### Get Specific Version
```sql
SELECT pv.* 
FROM steadytext_prompt_versions pv
JOIN steadytext_prompts p ON p.id = pv.prompt_id
WHERE p.slug = 'code-review' 
  AND pv.version = 3;
```

### List All Prompts
```sql
SELECT p.*, 
       MAX(pv.version) as latest_version,
       COUNT(pv.id) as total_versions
FROM steadytext_prompts p
LEFT JOIN steadytext_prompt_versions pv ON p.id = pv.prompt_id
GROUP BY p.id
ORDER BY p.created_at DESC;
```

## Migration Considerations

### From Existing System
If migrating from hardcoded prompts:
1. Create prompt records for each existing template
2. Insert as version 1
3. Mark as active

### Rollback Strategy
- Versions are immutable, can always revert to previous
- Keep version history for audit trail
- Consider archive table for deleted prompts