# Quickstart: PostgreSQL Prompt Registry

## Installation

The prompt registry is included in pg_steadytext extension version 2025.9.6+.

```sql
-- Upgrade to latest version
ALTER EXTENSION pg_steadytext UPDATE TO '2025.9.6';

-- Verify installation
SELECT extversion FROM pg_extension WHERE extname = 'pg_steadytext';
```

## Basic Usage

### 1. Create Your First Prompt

```sql
-- Create a simple greeting prompt
SELECT steadytext_prompt_create(
    'greeting',
    'Hello {{ name }}! Welcome to {{ product }}.',
    'A simple greeting template'
);

-- Or use the short alias
SELECT st_prompt_create(
    'code-review',
    'Please review this {{ language }} code:
    
{{ code }}

Focus areas: {{ focus_areas }}

Specific concerns: {{ concerns | default("None specified") }}',
    'Template for requesting code reviews'
);
```

### 2. Render a Prompt

```sql
-- Render with required variables
SELECT st_prompt_render(
    'greeting',
    '{"name": "Alice", "product": "PostgreSQL"}'::jsonb
);
-- Output: "Hello Alice! Welcome to PostgreSQL."

-- Render code review prompt
SELECT st_prompt_render(
    'code-review',
    '{
        "language": "Python",
        "code": "def hello():\n    print(\"Hello World\")",
        "focus_areas": "style, performance"
    }'::jsonb
);
```

### 3. Update a Prompt (Creates New Version)

```sql
-- Update the greeting to be more formal
SELECT st_prompt_update(
    'greeting',
    'Dear {{ name }},

We are pleased to welcome you to {{ product }}.

Best regards,
The {{ product }} Team'
);

-- Check versions
SELECT * FROM st_prompt_versions('greeting');
```

### 4. Use Specific Versions

```sql
-- Render version 1 (original)
SELECT st_prompt_render(
    'greeting',
    '{"name": "Bob", "product": "SteadyText"}'::jsonb,
    version => 1
);

-- Render version 2 (formal)
SELECT st_prompt_render(
    'greeting',
    '{"name": "Bob", "product": "SteadyText"}'::jsonb,
    version => 2
);

-- Default (latest version)
SELECT st_prompt_render(
    'greeting',
    '{"name": "Bob", "product": "SteadyText"}'::jsonb
);
```

### 5. List Available Prompts

```sql
-- See all prompts with version info
SELECT * FROM st_prompt_list();

-- Get details of a specific prompt
SELECT * FROM st_prompt_get('code-review');
```

## Advanced Features

### Conditional Logic in Templates

```sql
-- Create a prompt with Jinja2 conditionals
SELECT st_prompt_create(
    'task-assignment',
    'Task: {{ title }}
{% if priority == "high" %}
⚠️ HIGH PRIORITY - Please address immediately!
{% elif priority == "medium" %}
📋 Medium priority - Complete within 2 days
{% else %}
📝 Low priority - Complete when possible
{% endif %}

Assigned to: {{ assignee }}
{% if due_date %}
Due: {{ due_date }}
{% endif %}',
    'Task assignment notification template'
);

-- Render with different priorities
SELECT st_prompt_render(
    'task-assignment',
    '{
        "title": "Fix database connection",
        "priority": "high",
        "assignee": "Alice",
        "due_date": "2025-09-07"
    }'::jsonb
);
```

### Loops in Templates

```sql
-- Create a prompt with loops
SELECT st_prompt_create(
    'team-standup',
    'Daily Standup - {{ date }}
    
Team Members Present:
{% for member in team_members %}
- {{ member }}
{% endfor %}

Topics to Discuss:
{% for topic in topics %}
{{ loop.index }}. {{ topic }}
{% endfor %}',
    'Daily standup meeting template'
);

-- Render with arrays
SELECT st_prompt_render(
    'team-standup',
    '{
        "date": "2025-09-06",
        "team_members": ["Alice", "Bob", "Charlie"],
        "topics": ["Sprint progress", "Blockers", "Today''s goals"]
    }'::jsonb
);
```

### Handling Missing Variables

```sql
-- Strict mode (default) - errors on missing variables
SELECT st_prompt_render(
    'greeting',
    '{"name": "Alice"}'::jsonb  -- Missing "product"
);
-- ERROR: Required variable 'product' not provided

-- Lenient mode - missing variables become empty
SELECT st_prompt_render(
    'greeting',
    '{"name": "Alice"}'::jsonb,
    strict => false
);
-- Output: "Hello Alice! Welcome to ."
```

## Integration with AI Generation

Combine prompt templates with SteadyText's AI generation:

```sql
-- Create an AI prompt template
SELECT st_prompt_create(
    'ai-code-review',
    'You are a {{ language }} expert. Please review this code:

```{{ language }}
{{ code }}
```

Focus on:
{% for area in focus_areas %}
- {{ area }}
{% endfor %}

Provide specific, actionable feedback.',
    'AI code review prompt template'
);

-- Render and generate AI response
WITH rendered_prompt AS (
    SELECT st_prompt_render(
        'ai-code-review',
        '{
            "language": "Python",
            "code": "def calc(x, y):\n    return x + y",
            "focus_areas": ["naming", "documentation", "error handling"]
        }'::jsonb
    ) AS prompt
)
SELECT st_generate(prompt, max_tokens => 500)
FROM rendered_prompt;
```

## Best Practices

1. **Use descriptive slugs**: `user-welcome-email` instead of `template1`
2. **Document required variables**: Include in description or metadata
3. **Version strategically**: Update for content changes, not typo fixes
4. **Use defaults**: `{{ var | default("N/A") }}` for optional variables
5. **Test templates**: Validate with sample data before production use

## Troubleshooting

### Common Errors

**Template Syntax Error**:
```sql
-- This will fail
SELECT st_prompt_create('bad', '{{ name }');  -- Unclosed variable
-- ERROR: Template syntax error: Unexpected end of template
```

**Missing Required Variable**:
```sql
-- Check required variables
SELECT * FROM st_prompt_get('greeting');
-- Shows required_variables: {name, product}
```

**Slug Already Exists**:
```sql
-- Use update instead of create for existing prompts
SELECT st_prompt_update('existing-slug', 'new template');
```

## Performance Tips

1. Templates are cached after first compilation
2. Use specific versions in production for consistency
3. Keep templates under 10KB for optimal performance
4. Minimize complex loops for faster rendering

## Migration from Hardcoded Prompts

```sql
-- Example: Migrate existing prompts
DO $$
DECLARE
    prompts RECORD;
BEGIN
    FOR prompts IN 
        SELECT * FROM (VALUES
            ('welcome', 'Welcome {{ user }}!'),
            ('goodbye', 'Goodbye {{ user }}, see you soon!')
        ) AS t(slug, template)
    LOOP
        PERFORM st_prompt_create(prompts.slug, prompts.template);
    END LOOP;
END $$;
```