# Structured Generation

SteadyText v2.4.1 introduces powerful structured generation capabilities, allowing you to force the model's output to conform to a specific format. This is useful for a wide range of applications, from data extraction to building reliable applications on top of language models.

This feature is powered by llama.cpp's native grammar support, providing better compatibility and performance compared to external libraries.

## How it Works

Structured generation is enabled by passing one of the following parameters to the `steadytext.generate` function:

- `schema`: For generating JSON that conforms to a JSON schema, a Pydantic model, or a basic Python type.
- `regex`: For generating text that matches a regular expression.
- `choices`: For generating text that is one of a list of choices.

When one of these parameters is provided, SteadyText converts your constraint into a GBNF (Grammatical Backus-Naur Form) grammar that llama.cpp uses to guide the generation process. This ensures that the output is always valid according to the specified format.

The conversion process:
1. JSON schemas, Pydantic models, and Python types are converted to GBNF grammars that enforce the exact structure
2. Regular expressions are converted to equivalent GBNF patterns (when possible)
3. Choice lists are converted to simple alternative rules in GBNF

This native integration with llama.cpp provides deterministic, reliable structured output generation.

## JSON Generation

You can generate JSON output in several ways.

### With a JSON Schema

Pass a dictionary representing a JSON schema to the `schema` parameter.

```python
import steadytext
import json

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

result = steadytext.generate("Create a user named Alice, age 42", schema=schema)

# The result will contain a JSON object wrapped in <json-output> tags
# <json-output>{"name": "Alice", "age": 42}</json-output>

json_string = result.split('<json-output>')[1].split('</json-output>')[0]
data = json.loads(json_string)

assert data['name'] == "Alice"
assert data['age'] == 42
```

### With a Pydantic Model

You can also use a Pydantic model to define the structure of the JSON output.

```python
import steadytext
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

result = steadytext.generate("Create a user named Bob, age 30", schema=User)
```

### With `generate_json`

The `generate_json` convenience function can also be used.

```python
import steadytext
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

result = steadytext.generate_json("Create a user named Charlie, age 25", schema=User)
```

## Regex-Constrained Generation

Generate text that matches a regular expression using the `regex` parameter.

```python
import steadytext

# Generate a phone number
phone_number = steadytext.generate(
    "The support number is: ",
    regex=r"\d{3}-\d{3}-\d{4}"
)
print(phone_number)
# Output: 123-456-7890

# Generate a valid email address
email = steadytext.generate(
    "Contact email: ",
    regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)
print(email)
# Output: example@domain.com
```

You can also use the `generate_regex` convenience function.

```python
import steadytext

# Generate a date
date = steadytext.generate_regex(
    "Today's date is: ",
    pattern=r"\d{4}-\d{2}-\d{2}"
)
print(date)
# Output: 2025-07-03
```

## Multiple Choice

Force the model to choose from a list of options using the `choices` parameter.

```python
import steadytext

sentiment = steadytext.generate(
    "The movie was fantastic!",
    choices=["positive", "negative", "neutral"]
)
print(sentiment)
# Output: positive
```

The `generate_choice` convenience function is also available.

```python
import steadytext

answer = steadytext.generate_choice(
    "Is Python a statically typed language?",
    choices=["Yes", "No"]
)
print(answer)
# Output: No
```

## Type-Constrained Generation

You can also constrain the output to a specific Python type using the `generate_format` function.

```python
import steadytext

# Generate an integer
quantity = steadytext.generate_format("Number of items: ", int)
print(quantity)
# Output: 5

# Generate a boolean
is_active = steadytext.generate_format("Is the user active? ", bool)
print(is_active)
# Output: True
```

## PostgreSQL Extension Support

All structured generation features are fully supported in the PostgreSQL extension (pg_steadytext) as of v2.4.1. You can use structured generation directly in your SQL queries.

### SQL Functions

The PostgreSQL extension provides the following structured generation functions:

```sql
-- JSON generation with schema
steadytext_generate_json(
    prompt TEXT,
    schema JSONB,
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TEXT

-- Regex-constrained generation
steadytext_generate_regex(
    prompt TEXT,
    pattern TEXT,
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TEXT

-- Multiple choice generation
steadytext_generate_choice(
    prompt TEXT,
    choices TEXT[],
    max_tokens INTEGER DEFAULT 512,
    use_cache BOOLEAN DEFAULT true,
    seed INTEGER DEFAULT 42
) RETURNS TEXT
```

### PostgreSQL Examples

```sql
-- Generate structured user data
SELECT steadytext_generate_json(
    'Create a user profile for John Doe, age 35, software engineer',
    '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "occupation": {"type": "string"}}}'::jsonb
);

-- Generate formatted phone numbers
SELECT steadytext_generate_regex(
    'Customer service: ',
    '\(\d{3}\) \d{3}-\d{4}'
);

-- Sentiment classification
SELECT steadytext_generate_choice(
    'Sentiment of "This product is amazing!": ',
    ARRAY['positive', 'negative', 'neutral']
);
```

All functions support async variants as well:
- `steadytext_generate_json_async()`
- `steadytext_generate_regex_async()`
- `steadytext_generate_choice_async()`

For more PostgreSQL-specific examples, see the [PostgreSQL Integration Examples](examples/postgresql-integration.md#structured-generation-examples).
```
