# Prompt Registry Guide

The pg_steadytext Prompt Registry is a comprehensive Jinja2-based template management system that provides immutable versioning, automatic variable extraction, and rich metadata support for storing and rendering text templates directly within PostgreSQL.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Function Reference](#function-reference)
- [Advanced Usage](#advanced-usage)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)

## Overview

The Prompt Registry enables you to:

- Store Jinja2 templates with full syntax support (variables, loops, conditionals, filters)
- Maintain immutable version history for audit trails and rollback capabilities
- Automatically extract and validate template variables
- Organize templates with rich metadata and tagging
- Render templates with variable substitution in strict or permissive modes
- Cache compiled templates for optimal performance

## Key Features

### Jinja2 Template Engine
Full support for Jinja2 syntax including:
- Variable substitution: `{{ variable }}`
- Control structures: `{% if %}`, `{% for %}`, `{% endif %}`
- Filters: `{{ name|title }}`, `{{ price|round(2) }}`
- Comments: `{# This is a comment #}`
- Template inheritance and macros

### Immutable Versioning
- Every update creates a new version while preserving history
- Version numbers are automatically assigned (1, 2, 3, ...)
- Only one version is "active" at a time (latest by default)
- Historical versions remain accessible for rollback or comparison

### Automatic Variable Extraction
- Required variables are automatically detected from templates
- Variables are stored with each version for validation
- Rendering can enforce strict variable checking or allow missing variables

### Rich Metadata Support
- JSONB metadata for flexible categorization and tagging
- Timestamps and user tracking for audit trails
- Searchable metadata for organizing large template libraries

## Known Limitations

### Reserved JSON Key Names

**Important**: The following JSON key names conflict with Python dict methods and cannot be used in templates:
- `items` - Conflicts with dict.items() method
- `keys` - Conflicts with dict.keys() method  
- `values` - Conflicts with dict.values() method

**Workaround**: Use alternative names like `products`, `entries`, `elements`, etc.

```sql
-- ❌ BAD: Will cause "builtin_function_or_method object is not iterable" error
'{% for item in order.items %}...'

-- ✅ GOOD: Use alternative key names
'{% for product in order.products %}...'
```

### Available Jinja2 Filters

The following standard Jinja2 filters are available:

**String Filters:**
- `capitalize`, `lower`, `upper`, `title`
- `trim`, `truncate(length)`, `wordwrap(width)`
- `replace(old, new)`, `escape`, `urlize`

**Numeric Filters:**
- `abs`, `float`, `int`, `round(precision)`
- `sum` (for lists)

**List/Dict Filters:**
- `first`, `last`, `length`, `reverse`, `sort`
- `join(separator)`, `unique`, `select`, `reject`

**Formatting Filters:**
- `default(value)` - Provide default for undefined variables
- `format` - Python string formatting (e.g., `"%.2f"|format(price)`)
- `tojson` - Convert to JSON string

**Unavailable Filters** (require custom registration):
- `strftime` - Date formatting (use pre-formatted dates instead)
- `rjust`, `ljust`, `center` - String padding (use CSS/formatting in output layer)

## Getting Started

### Basic Template Creation

```sql
-- Create a simple greeting template
SELECT st_prompt_create(
    'greeting',
    'Hello {{ name }}, welcome to {{ service }}!',
    'Basic greeting template'
);

-- Render the template
SELECT st_prompt_render(
    'greeting',
    '{"name": "Alice", "service": "PostgreSQL"}'::jsonb
);
-- Result: "Hello Alice, welcome to PostgreSQL!"
```

### Template with Conditionals

```sql
-- Create a conditional template
SELECT st_prompt_create(
    'user-welcome',
    'Hi {{ user.name }}!

{% if user.is_premium -%}
Welcome back, premium member! You have access to all features.
{% else -%}
Welcome! Consider upgrading to premium for more features.
{% endif -%}

{% if user.last_login -%}
Last login: {{ user.last_login }}
{% endif %}'
);

-- Render for premium user
SELECT st_prompt_render(
    'user-welcome',
    '{"user": {"name": "John", "is_premium": true, "last_login": "2024-01-15"}}'::jsonb
);
```

### Template with Loops

```sql
-- Create a template with loops
SELECT st_prompt_create(
    'order-summary',
    'Order #{{ order.id }} Summary:

{% for product in order.products -%}
{{ loop.index }}. {{ product.name }} - Qty: {{ product.qty }} - ${{ product.price }}
{% endfor %}

Subtotal: ${{ order.subtotal }}
{% if order.discount and order.discount|float > 0 -%}
Discount: -${{ order.discount }}
{% endif -%}
Total: ${{ order.total }}'
);
```

## Function Reference

### Management Functions

#### `st_prompt_create(slug, template, description, metadata)`
Creates a new prompt template.

**Parameters:**
- `slug` (TEXT): Unique identifier (lowercase, hyphens, 3-100 chars)
- `template` (TEXT): Jinja2 template content
- `description` (TEXT): Optional description
- `metadata` (JSONB): Optional metadata for categorization

**Returns:** UUID of the created prompt

**Example:**

```sql
SELECT st_prompt_create(
    'email-confirmation',
    'Dear {{ customer.name }}, your order {{ order.id }} is confirmed.',
    'Order confirmation email template',
    '{"category": "email", "tags": ["order", "confirmation"], "owner": "sales-team"}'::jsonb
);
```

#### `st_prompt_update(slug, template, metadata)`
Creates a new version of an existing prompt.

**Parameters:**
- `slug` (TEXT): Existing prompt identifier
- `template` (TEXT): Updated template content
- `metadata` (JSONB): Optional updated metadata

**Returns:** UUID of the new version

**Example:**

```sql
-- Update the template (creates version 2)
SELECT st_prompt_update(
    'email-confirmation',
    'Hello {{ customer.name }}! 🎉 Your order {{ order.id }} has been confirmed and will ship soon.',
    '{"category": "email", "tags": ["order", "confirmation"], "version_note": "Added emoji and shipping info"}'::jsonb
);
```

#### `st_prompt_get(slug, version)`
Retrieves a prompt template.

**Parameters:**
- `slug` (TEXT): Prompt identifier
- `version` (INTEGER): Optional specific version (defaults to active version)

**Returns:** Table with prompt details

**Example:**

```sql
-- Get the active version
SELECT template, required_variables 
FROM st_prompt_get('email-confirmation');

-- Get a specific version
SELECT template, version_num, created_at 
FROM st_prompt_get('email-confirmation', 1);
```

#### `st_prompt_delete(slug)`
Deletes a prompt and all its versions.

**Parameters:**
- `slug` (TEXT): Prompt identifier

**Returns:** BOOLEAN (true if deleted, false if not found)

**Example:**

```sql
SELECT st_prompt_delete('old-template');
```

### Rendering Functions

#### `st_prompt_render(slug, variables, version, strict)`
Renders a template with variable substitution.

**Parameters:**
- `slug` (TEXT): Prompt identifier
- `variables` (JSONB): Variables to substitute
- `version` (INTEGER): Optional specific version
- `strict` (BOOLEAN): Enforce variable validation (default: true)

**Returns:** TEXT (rendered template)

**Example:**

```sql
-- Strict rendering (error if variables missing)
SELECT st_prompt_render(
    'order-summary',
    '{"order": {"id": "12345", "products": [{"name": "Widget", "qty": 2, "price": "19.99"}], "total": "39.98"}}'::jsonb,
    NULL,  -- use active version
    true   -- strict mode
);

-- Permissive rendering (missing variables become empty)
SELECT st_prompt_render(
    'order-summary',
    '{"order": {"id": "12345"}}'::jsonb,
    NULL,
    false  -- non-strict mode
);
```

### Discovery Functions

#### `st_prompt_list()`
Lists all prompts with summary information.

**Returns:** Table with prompt metadata

**Example:**

```sql
SELECT slug, description, latest_version_num, total_versions, created_at
FROM st_prompt_list()
ORDER BY created_at DESC;
```

#### `st_prompt_versions(slug)`
Lists all versions of a specific prompt.

**Parameters:**
- `slug` (TEXT): Prompt identifier

**Returns:** Table with version details

**Example:**

```sql
SELECT version_num, created_at, created_by, is_active,
       left(template, 50) || '...' as template_preview
FROM st_prompt_versions('email-confirmation')
ORDER BY version_num DESC;
```

## Advanced Usage

### Complex Data Structures

```sql
-- Template for handling nested data
SELECT st_prompt_create(
    'report-generator',
    'Monthly Report for {{ company.name }}
    
Generated: {{ report.date }}

{% for department in report.departments -%}
## {{ department.name|title }} Department

{% if department.metrics -%}
Key Metrics:
{% for metric in department.metrics -%}
- {{ metric.name }}: {{ metric.value }}{% if metric.change %} ({{ metric.change }}{% if metric.change > 0 %}▲{% else %}▼{% endif %}){% endif %}
{% endfor %}
{% endif -%}

{% if department.employees -%}
Top Performers:
{% for employee in department.employees[:3] -%}
{{ loop.index }}. {{ employee.name }} - {{ employee.performance_score }}/100
{% endfor %}
{% endif -%}

{% endfor %}'
);
```

### Template Inheritance Patterns

```sql
-- Base email template
SELECT st_prompt_create(
    'email-base',
    'Dear {{ recipient.name }},

{{ content }}

{% if signature -%}
{{ signature }}
{% else -%}
Best regards,
{{ sender.name|default("The Team") }}
{% endif -%}

{% if unsubscribe_link -%}
Unsubscribe: {{ unsubscribe_link }}
{% endif %}'
);

-- Specialized templates that extend the base
SELECT st_prompt_create(
    'welcome-email',
    '{% set content -%}
Welcome to {{ product.name }}! We''re excited to have you on board.

{% if user.trial_days -%}
You have {{ user.trial_days }} days of free trial remaining.
{% endif -%}

Get started: {{ product.getting_started_url }}
{% endset -%}

{% include "email-base" %}'
);
```

### Dynamic Content with Filters

```sql
SELECT st_prompt_create(
    'invoice-template',
    'INVOICE #{{ invoice.number }}

Date: {{ invoice.date }}
Due Date: {{ invoice.due_date }}

Bill To:
{{ customer.name|title }}
{{ customer.address|replace("\n", "\n") }}

{% for item in invoice.line_items -%}
{{ item.description|truncate(40) }}
Qty: {{ item.quantity }}  
Rate: ${{ "%.2f"|format(item.rate) }}
Amount: ${{ "%.2f"|format(item.amount) }}
{% endfor %}

Subtotal: ${{ "%.2f"|format(invoice.subtotal) }}
{% if invoice.tax > 0 -%}
Tax ({{ "%.1f"|format(invoice.tax_rate * 100) }}%): ${{ "%.2f"|format(invoice.tax) }}
{% endif -%}
{% if invoice.discount > 0 -%}
Discount: -${{ "%.2f"|format(invoice.discount) }}
{% endif -%}

TOTAL: ${{ "%.2f"|format(invoice.total) }}'
);
```

## Real-World Examples

### AI Prompt Management

```sql
-- Create AI prompts for different tasks
SELECT st_prompt_create(
    'code-review-prompt',
    'You are an expert {{ language }} developer. Please review the following code:

```{{ language }}
{{ code }}
```

Focus Areas:
{% for area in focus_areas -%}
- {{ area|title }}
{% endfor %}

{% if context -%}
Additional Context:
{{ context }}
{% endif -%}

Provide specific, actionable feedback with code examples where helpful.',
    'AI code review prompt with configurable focus areas',
    '{"category": "ai", "use_case": "code_review", "model": "gpt-4"}'::jsonb
);

-- Use with different languages and contexts
WITH ai_prompt AS (
    SELECT st_prompt_render(
        'code-review-prompt',
        '{
            "language": "Python",
            "code": "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "focus_areas": ["performance", "edge cases", "readability"],
            "context": "This function will be used in a high-performance financial system."
        }'::jsonb
    ) as rendered_prompt
)
SELECT steadytext_generate(rendered_prompt, max_tokens := 500)
FROM ai_prompt;
```

### E-commerce Email System

```sql
-- Abandoned cart email with personalization
SELECT st_prompt_create(
    'abandoned-cart-email',
    'Hi {{ customer.first_name }},

You left some great items in your cart! Don''t let them get away.

Your Cart:
{% for product in cart.products -%}
• {{ product.name }}{% if product.variant %} ({{ product.variant }}){% endif %}
  ${{ "%.2f"|format(product.price) }}{% if product.was_price > product.price %} <strike>${{ "%.2f"|format(product.was_price) }}</strike>{% endif %}
{% endfor %}

Cart Total: ${{ "%.2f"|format(cart.total) }}
{% if cart.shipping_required and cart.total < free_shipping_threshold -%}
Add ${{ "%.2f"|format(free_shipping_threshold - cart.total) }} more for free shipping!
{% elif cart.shipping_required -%}
🎉 You qualify for free shipping!
{% endif -%}

{% if customer.has_used_discount_code -%}
Complete your purchase: {{ cart.checkout_url }}
{% else -%}
Complete your purchase with 10% off: {{ cart.checkout_url }}?discount=COMEBACK10
{% endif -%}

{% if recommended_products -%}
You might also like:
{% for product in recommended_products[:3] -%}
• {{ product.name }} - ${{ "%.2f"|format(product.price) }}
{% endfor %}
{% endif -%}

Happy shopping!
The {{ store.name }} Team',
    'Personalized abandoned cart recovery email',
    '{"category": "email", "type": "marketing", "automation": "abandoned_cart"}'::jsonb
);
```

### Database Schema Generation

```sql
-- DDL template generator
SELECT st_prompt_create(
    'create-table-ddl',
    '-- Generated table: {{ table.schema }}.{{ table.name }}
-- Description: {{ table.description|default("Auto-generated table") }}
-- Created: {{ "now"|strftime("%Y-%m-%d %H:%M:%S") }}

CREATE TABLE {{ table.schema }}.{{ table.name }} (
{% for column in table.columns -%}
    {{ column.name|ljust(20) }} {{ column.type }}
    {%- if column.not_null %} NOT NULL{% endif %}
    {%- if column.default %} DEFAULT {{ column.default }}{% endif %}
    {%- if column.check %} CHECK ({{ column.check }}){% endif %}
    {%- if not loop.last %},{% endif %}
{% endfor %}
{% if table.primary_key -%}
    ,
    CONSTRAINT pk_{{ table.name }} PRIMARY KEY ({{ table.primary_key|join(", ") }})
{% endif -%}
{% if table.unique_constraints -%}
{% for constraint in table.unique_constraints -%}
    ,
    CONSTRAINT uk_{{ table.name }}_{{ constraint.name }} UNIQUE ({{ constraint.columns|join(", ") }})
{% endfor %}
{% endif -%}
);

{% if table.indexes -%}
-- Indexes
{% for index in table.indexes -%}
CREATE INDEX idx_{{ table.name }}_{{ index.name }}
    ON {{ table.schema }}.{{ table.name }} ({{ index.columns|join(", ") }})
    {%- if index.where %} WHERE {{ index.where }}{% endif %};
{% endfor %}

{% endif -%}
{% if table.comment -%}
COMMENT ON TABLE {{ table.schema }}.{{ table.name }} IS {{ table.comment|tojson }};

{% for column in table.columns -%}
{% if column.comment -%}
COMMENT ON COLUMN {{ table.schema }}.{{ table.name }}.{{ column.name }} IS {{ column.comment|tojson }};
{% endif -%}
{% endfor %}
{% endif %}'
);
```

## Best Practices

### Template Organization

1. **Use Descriptive Slugs**: Choose clear, hierarchical naming conventions
   ```sql
   -- Good
   'email-order-confirmation'
   'ai-code-review-python'
   'report-monthly-sales'
   
   -- Less clear
   'template1'
   'email'
   'prompt'
   ```

2. **Leverage Metadata**: Use rich metadata for organization and searchability
   ```sql
   SELECT st_prompt_create(
       'user-onboarding-email',
       '...',
       'Welcome email for new users',
       '{
           "category": "email",
           "type": "transactional", 
           "stage": "onboarding",
           "owner": "product-team",
           "tags": ["welcome", "user-journey", "automated"],
           "last_reviewed": "2024-01-15"
       }'::jsonb
   );
   ```

3. **Version Management**: Use meaningful version metadata
   ```sql
   SELECT st_prompt_update(
       'user-onboarding-email',
       '... updated template ...',
       '{
           "category": "email",
           "version_note": "Added personalized product recommendations",
           "breaking_changes": false,
           "updated_by": "alice@company.com"
       }'::jsonb
   );
   ```

### Template Design

1. **Plan Variable Structure**: Design consistent variable schemas
   ```json
   {
       "user": {
           "name": "string",
           "email": "string",
           "preferences": {}
       },
       "context": {
           "timestamp": "string",
           "source": "string"
       }
   }
   ```

2. **Use Default Values**: Handle missing data gracefully
   ```jinja2
   Hello {{ user.name|default("Valued Customer") }},
   
   {% if user.preferences.newsletter|default(true) -%}
   Here's your newsletter...
   {% endif %}
   ```

3. **Comment Complex Logic**: Document template logic
   ```jinja2
   {# Calculate discount based on user tier and order value #}
   {% if user.tier == "premium" and order.value > 100 -%}
       {% set discount = order.value * 0.15 %}
   {% elif user.tier == "gold" or order.value > 50 -%}
       {% set discount = order.value * 0.10 %}
   {% else -%}
       {% set discount = 0 %}
   {% endif %}
   ```

### Performance Optimization

1. **Use Non-Strict Mode Carefully**: Only when you need flexible variable handling
   ```sql
   -- Strict mode (recommended for most cases)
   SELECT st_prompt_render('template', variables, NULL, true);
   
   -- Non-strict mode (when variables may be optional)
   SELECT st_prompt_render('template', variables, NULL, false);
   ```

2. **Cache Frequently Used Templates**: Templates are automatically cached, but consider application-level caching for high-frequency rendering

3. **Batch Operations**: When rendering many templates, consider batching
   ```sql
   WITH template_data AS (
       SELECT 
           'welcome-email' as slug,
           jsonb_build_object('name', name, 'email', email) as vars
       FROM users 
       WHERE created_at > NOW() - INTERVAL '1 day'
   )
   SELECT 
       email,
       st_prompt_render(slug, vars) as rendered_email
   FROM template_data;
   ```

## Performance Considerations

### Template Compilation Caching

- Compiled Jinja2 templates are cached in PostgreSQL's global dictionary (GD)
- Cache keys include template hash to invalidate when templates change
- Cache survives for the duration of the database session

### Database Performance

- Prompt lookups use indexes on `slug` columns
- Version queries are optimized with compound indexes
- Variable extraction is pre-computed and stored

### Memory Usage

- Template cache grows with unique template/version combinations
- Consider the trade-off between compilation time and memory usage
- Monitor session memory if using many different templates

## Troubleshooting

### Common Errors

#### Template Syntax Errors

```sql
-- Error: Invalid Jinja2 template: unexpected end of template
SELECT st_prompt_create('bad-template', '{{ unclosed_variable');
```

**Solution**: Check for unclosed template tags and proper Jinja2 syntax.

#### Invalid Slug Format

```sql
-- Error: Invalid slug format. Use lowercase letters, numbers, and hyphens only
SELECT st_prompt_create('Bad_Slug!', 'Template');
```

**Solution**: Use only lowercase letters, numbers, and hyphens. Length must be 3-100 characters.

#### Missing Required Variables

```sql
-- Error: Missing required variables: name, email
SELECT st_prompt_render('template', '{}', NULL, true);
```

**Solution**: Provide all required variables or use non-strict mode.

#### Template Not Found

```sql
-- Error: Prompt with slug "nonexistent" not found
SELECT st_prompt_render('nonexistent', '{}');
```

**Solution**: Verify the slug exists using `st_prompt_list()`.

### Debugging Tips

1. **Check Required Variables**:

   ```sql
   SELECT required_variables 
   FROM st_prompt_get('your-template');
   ```

2. **Test with Simple Data**:

   ```sql
   -- Start with minimal variables
   SELECT st_prompt_render('template', '{"name": "test"}');
   ```

3. **Use Non-Strict Mode for Testing**:

   ```sql
   -- See what renders even with missing variables
   SELECT st_prompt_render('template', '{}', NULL, false);
   ```

4. **Check Version History**:

   ```sql
   SELECT version_num, created_at, 
          left(template, 100) as template_preview
   FROM st_prompt_versions('your-template')
   ORDER BY version_num DESC;
   ```

### Performance Issues

1. **Template Not Cached**: Each render compiles the template
   - Check if template content is changing frequently
   - Consider stabilizing template content

2. **Large Result Sets**: Rendering many templates at once
   - Consider pagination or batching
   - Use LIMIT clauses in discovery queries

3. **Complex Templates**: Heavy computation in templates
   - Pre-compute complex data outside templates
   - Simplify template logic where possible

---

For more information and updates, see the main [pg_steadytext README](https://github.com/julep-ai/steadytext/blob/main/pg_steadytext/README.md) and [Changelog](https://github.com/julep-ai/steadytext/blob/main/pg_steadytext/CHANGELOG.md).
