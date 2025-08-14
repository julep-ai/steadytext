#!/usr/bin/env python3
"""Example of using SteadyText with Pydantic model instantiation.

This example demonstrates the new return_pydantic parameter and generate_pydantic function
that allow you to get validated Pydantic model instances directly from generation.
"""

import steadytext
from pydantic import BaseModel
from typing import List, Optional


# Define Pydantic models
class User(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


class Product(BaseModel):
    name: str
    price: float
    description: str
    tags: List[str]
    in_stock: bool


class TodoItem(BaseModel):
    task: str
    completed: bool
    priority: Optional[int] = None


def example_basic_pydantic():
    """Basic example using return_pydantic parameter."""
    print("=== Basic Pydantic Generation ===\n")

    # Traditional way - returns string with XML tags
    result_str = steadytext.generate(
        "Create a user named Alice who is 30 years old", schema=User
    )
    print("Traditional output (string):")
    print(result_str)
    print()

    # New way - returns Pydantic model instance
    user = steadytext.generate(
        "Create a user named Alice who is 30 years old",
        schema=User,
        return_pydantic=True,
    )
    print("With return_pydantic=True:")
    print(f"Type: {type(user)}")
    # Type assertion to help type checker
    assert isinstance(user, User)
    print(f"Name: {user.name}")
    print(f"Age: {user.age}")
    print(f"Email: {user.email}")
    print()


def example_generate_pydantic():
    """Example using the generate_pydantic convenience function."""
    print("=== Using generate_pydantic() ===\n")

    # Direct function that always returns Pydantic instance
    product = steadytext.generate_pydantic(
        "Create a laptop product that costs $999, with gaming features", Product
    )

    print(f"Product Type: {type(product)}")
    # Type assertion to help type checker
    assert isinstance(product, Product)
    print(f"Name: {product.name}")
    print(f"Price: ${product.price}")
    print(f"Description: {product.description}")
    print(f"Tags: {product.tags}")
    print(f"In Stock: {product.in_stock}")
    print()


def example_deterministic_generation():
    """Show that Pydantic generation is deterministic."""
    print("=== Deterministic Pydantic Generation ===\n")

    # Generate the same user multiple times with same seed
    prompt = "Create a user for testing"

    user1 = steadytext.generate_pydantic(prompt, User, seed=42)
    user2 = steadytext.generate_pydantic(prompt, User, seed=42)
    user3 = steadytext.generate_pydantic(prompt, User, seed=123)

    print(f"User 1 (seed=42): {user1}")
    print(f"User 2 (seed=42): {user2}")
    print(f"User 3 (seed=123): {user3}")
    print()

    print(f"User1 == User2: {user1 == user2}")
    print(f"User1 == User3: {user1 == user3}")
    print()


def example_complex_model():
    """Example with a more complex model."""
    print("=== Complex Model Generation ===\n")

    class Company(BaseModel):
        name: str
        founded_year: int
        employees: List[User]
        products: List[Product]

    # Generate a complex nested structure
    company = steadytext.generate_pydantic(
        "Create a tech startup with 2 employees and 2 products", Company
    )

    print(f"Company: {company.name}")
    print(f"Founded: {company.founded_year}")
    print(f"Number of employees: {len(company.employees)}")
    print(f"Number of products: {len(company.products)}")

    for i, emp in enumerate(company.employees, 1):
        print(f"  Employee {i}: {emp.name}, Age: {emp.age}")

    for i, prod in enumerate(company.products, 1):
        print(f"  Product {i}: {prod.name}, Price: ${prod.price}")
    print()


def example_todo_list():
    """Example generating a todo list."""
    print("=== Todo List Generation ===\n")

    class TodoList(BaseModel):
        title: str
        items: List[TodoItem]

    todo_list = steadytext.generate_pydantic(
        "Create a todo list for launching a new feature", TodoList
    )

    # Type assertion to help type checker
    assert isinstance(todo_list, TodoList)
    print(f"Todo List: {todo_list.title}")
    print("Items:")
    for item in todo_list.items:
        status = "✓" if item.completed else "○"
        priority = f"[P{item.priority}]" if item.priority else ""
        print(f"  {status} {item.task} {priority}")
    print()


def example_validation():
    """Show that Pydantic validation is enforced."""
    print("=== Pydantic Validation ===\n")

    # The generated JSON will always conform to the schema
    # due to grammar-constrained generation
    user = steadytext.generate_pydantic(
        "Create a user with invalid data",  # Even with this prompt
        User,  # The output will still be valid according to User schema
    )

    # Type assertion to help type checker
    assert isinstance(user, User)
    print("Even with 'invalid data' prompt, result is valid:")
    print(f"Name (str): {user.name} - Type: {type(user.name)}")
    print(f"Age (int): {user.age} - Type: {type(user.age)}")
    print(f"Email (optional): {user.email} - Type: {type(user.email)}")
    print()


def main():
    """Run all examples."""
    print("SteadyText Pydantic Generation Examples")
    print("=" * 40)
    print()

    # Note: These examples will only work when models are available
    # Set STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true to download models

    try:
        example_basic_pydantic()
        example_generate_pydantic()
        example_deterministic_generation()
        example_complex_model()
        example_todo_list()
        example_validation()

        print("All examples completed successfully!")

    except Exception as e:
        print("Note: Examples require model to be available.")
        print("Set STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true to download models.")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
