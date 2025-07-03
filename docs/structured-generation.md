# Structured Generation

SteadyText v2.3.0 introduces powerful structured generation capabilities, allowing you to force the model's output to conform to a specific format. This is useful for a wide range of applications, from data extraction to building reliable applications on top of language models.

This feature is powered by the [Outlines](https://github.com/outlines-dev/outlines) library.

## How it Works

Structured generation is enabled by passing one of the following parameters to the `steadytext.generate` function:

- `schema`: For generating JSON that conforms to a JSON schema, a Pydantic model, or a basic Python type.
- `regex`: For generating text that matches a regular expression.
- `choices`: For generating text that is one of a list of choices.

When one of these parameters is provided, SteadyText will use Outlines to guide the generation process, ensuring that the output is always valid according to the specified format.

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
