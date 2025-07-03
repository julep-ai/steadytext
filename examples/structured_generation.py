#!/usr/bin/env python3
"""Examples of structured text generation with SteadyText.

This example demonstrates how to use SteadyText's structured generation
capabilities to generate text that conforms to JSON schemas, regex patterns,
and choice constraints.

AIDEV-NOTE: This example shows the integration with Outlines for structured output.
"""

import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from steadytext import (
    generate,
    generate_json,
    generate_regex,
    generate_choice,
    generate_format,
)


# Define Pydantic models for structured data
class Product(BaseModel):
    """Product information model."""

    name: str
    price: float
    description: str
    in_stock: bool
    tags: List[str]


class TodoItem(BaseModel):
    """Todo list item model."""

    task: str
    priority: str  # "high", "medium", "low"
    completed: bool = False
    due_date: Optional[str] = None


class ContactInfo(BaseModel):
    """Contact information model."""

    name: str
    email: str
    phone: str
    address: Optional[str] = None


def example_json_schema():
    """Example using JSON schema for structured output."""
    print("=== JSON Schema Example ===")

    # Define a JSON schema
    schema = {
        "type": "object",
        "properties": {
            "recipe": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "ingredients": {"type": "array", "items": {"type": "string"}},
                    "servings": {"type": "integer"},
                    "cooking_time": {"type": "string"},
                },
                "required": ["name", "ingredients", "servings"],
            }
        },
        "required": ["recipe"],
    }

    # Generate with schema
    result = generate("Create a recipe for chocolate chip cookies", schema=schema)

    print(f"Generated output:\n{result}\n")

    # Extract and parse JSON
    if result and isinstance(result, str):
        json_start = result.find("<json-output>") + len("<json-output>")
        json_end = result.find("</json-output>")
        json_str = result[json_start:json_end]

        recipe_data = json.loads(json_str)
    else:
        print("No result generated")
        return

    print(f"Parsed recipe: {json.dumps(recipe_data, indent=2)}\n")


def example_pydantic_model():
    """Example using Pydantic models for type-safe generation."""
    print("=== Pydantic Model Example ===")

    # Generate a product
    result = generate_json("Create a product listing for a wireless keyboard", Product)

    print(f"Generated output:\n{result}\n")

    # Extract and validate with Pydantic
    json_start = result.find("<json-output>") + len("<json-output>")
    json_end = result.find("</json-output>")
    json_str = result[json_start:json_end]

    product = Product.model_validate_json(json_str)
    print("Validated product:")
    print(f"  Name: {product.name}")
    print(f"  Price: ${product.price}")
    print(f"  In Stock: {product.in_stock}")
    print(f"  Tags: {', '.join(product.tags)}\n")


def example_regex_patterns():
    """Example using regex patterns for formatted output."""
    print("=== Regex Pattern Examples ===")

    # Phone number
    phone = generate_regex("Generate a US phone number:", r"\(\d{3}\) \d{3}-\d{4}")
    print(f"Phone: {phone}")

    # Date format
    date = generate_regex("Today's date in MM/DD/YYYY format:", r"\d{2}/\d{2}/\d{4}")
    print(f"Date: {date}")

    # Product code
    code = generate_regex(
        "Generate a product code (3 letters followed by 4 digits):", r"[A-Z]{3}\d{4}"
    )
    print(f"Product Code: {code}")

    # Email
    email = generate_regex("Contact email:", r"[a-z]+\.[a-z]+@[a-z]+\.(com|org|net)")
    print(f"Email: {email}\n")


def example_choices():
    """Example using choice constraints."""
    print("=== Choice Constraint Examples ===")

    # Simple yes/no
    answer = generate_choice("Is Python suitable for machine learning?", ["yes", "no"])
    print(f"Answer: {answer}")

    # Multiple choice
    sentiment = generate_choice(
        "The new product launch was spectacular! Sentiment:",
        ["positive", "negative", "neutral", "mixed"],
    )
    print(f"Sentiment: {sentiment}")

    # Priority selection
    priority = generate_choice(
        "Bug: Application crashes on startup. Priority level:",
        ["critical", "high", "medium", "low"],
    )
    print(f"Priority: {priority}")

    # Category selection
    category = generate_choice(
        "Article about new AI breakthroughs. Category:",
        ["technology", "science", "business", "health", "entertainment"],
    )
    print(f"Category: {category}\n")


def example_basic_types():
    """Example using basic Python types."""
    print("=== Basic Type Examples ===")

    # Integer
    count = generate_format("How many items are in stock?", int)
    print(f"Stock count: {count}")

    # Float
    rating = generate_format("Product rating (0-5):", float)
    print(f"Rating: {rating}")

    # Boolean
    available = generate_format("Is the product available for shipping?", bool)
    print(f"Available: {available}\n")


def example_complex_workflow():
    """Example of a complex workflow using multiple structured generations."""
    print("=== Complex Workflow Example ===")
    print("Creating a customer support ticket...\n")

    # Step 1: Classify the issue
    category = generate_choice(
        "Customer says: 'My order hasn't arrived yet and it's been 2 weeks!' Category:",
        ["shipping", "product", "billing", "technical", "other"],
    )
    print(f"1. Issue Category: {category}")

    # Step 2: Determine priority
    priority = generate_choice(
        f"Issue: Late delivery (2 weeks). Category: {category}. Priority:",
        ["low", "medium", "high", "urgent"],
    )
    print(f"2. Priority Level: {priority}")

    # Step 3: Generate ticket details
    ticket_schema = {
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string"},
            "summary": {"type": "string"},
            "category": {"type": "string"},
            "priority": {"type": "string"},
            "estimated_resolution": {"type": "string"},
            "next_steps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["ticket_id", "summary", "category", "priority", "next_steps"],
    }

    ticket_result = generate(
        f"Create support ticket. Issue: Order not delivered after 2 weeks. Category: {category}. Priority: {priority}.",
        schema=ticket_schema,
    )

    # Extract ticket data
    if ticket_result and isinstance(ticket_result, str):
        json_start = ticket_result.find("<json-output>") + len("<json-output>")
        json_end = ticket_result.find("</json-output>")
        ticket_data = json.loads(ticket_result[json_start:json_end])
    else:
        print("No ticket result generated")
        return

    print("\n3. Generated Ticket:")
    print(f"   ID: {ticket_data['ticket_id']}")
    print(f"   Summary: {ticket_data['summary']}")
    print("   Next Steps:")
    for step in ticket_data["next_steps"]:
        print(f"   - {step}")

    # Step 4: Generate customer response
    response_template = generate_regex(
        "Generate order number for reference:", r"ORD-\d{4}-[A-Z]{2}-\d{3}"
    )

    print(f"\n4. Reference Order: {response_template}")


def example_list_generation():
    """Example generating lists with specific constraints."""
    print("=== List Generation Example ===")

    # Todo list
    todos_result = generate_json(
        "Create a todo list for launching a new feature", schema=List[TodoItem]
    )

    if todos_result and isinstance(todos_result, str):
        json_start = todos_result.find("<json-output>") + len("<json-output>")
        json_end = todos_result.find("</json-output>")
        todos = json.loads(todos_result[json_start:json_end])
    else:
        print("No todos result generated")
        return

    print("Generated Todo List:")
    for i, todo in enumerate(todos, 1):
        # Validate that todo dict has required fields
        if not isinstance(todo, dict) or "task" not in todo or "priority" not in todo:
            print(f"{i}. Invalid todo item: {todo}")
            continue

        try:
            # Type annotation to help type checker understand todo has required fields
            todo_dict: Dict[str, Any] = todo
            # We've already validated that task and priority exist
            todo_item = TodoItem(
                task=todo_dict["task"],
                priority=todo_dict["priority"],
                completed=todo_dict.get("completed", False),
                due_date=todo_dict.get("due_date"),
            )
            print(f"{i}. [{todo_item.priority.upper()}] {todo_item.task}")
            if todo_item.due_date:
                print(f"   Due: {todo_item.due_date}")
        except Exception as e:
            print(f"{i}. Error creating todo item: {e}")


def example_with_seed():
    """Example showing deterministic generation with seeds."""
    print("=== Deterministic Generation with Seeds ===")

    schema = {"type": "string"}

    # Generate with same seed multiple times
    print("Generating with seed=42:")
    for i in range(3):
        result = generate_json("Say hello", schema, seed=42)
        json_start = result.find("<json-output>") + len("<json-output>")
        json_end = result.find("</json-output>")
        greeting = json.loads(result[json_start:json_end])
        print(f"  Attempt {i + 1}: {greeting}")

    print("\nGenerating with seed=123:")
    result = generate_json("Say hello", schema, seed=123)
    json_start = result.find("<json-output>") + len("<json-output>")
    json_end = result.find("</json-output>")
    greeting = json.loads(result[json_start:json_end])
    print(f"  Result: {greeting}")


def main():
    """Run all examples."""
    print("SteadyText Structured Generation Examples")
    print("=" * 50)
    print()

    # Run examples
    example_json_schema()
    example_pydantic_model()
    example_regex_patterns()
    example_choices()
    example_basic_types()
    example_complex_workflow()
    example_list_generation()
    example_with_seed()

    print("\nAll examples completed!")


if __name__ == "__main__":
    main()
