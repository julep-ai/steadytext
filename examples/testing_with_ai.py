#!/usr/bin/env python3
"""
Examples of using SteadyText for testing with AI
"""

import steadytext


# Testing with reliable assertions
def test_ai_function():
    """Example test showing deterministic behavior"""

    # This function would normally call your AI-powered function
    def my_ai_function(input_text):
        # In reality, this might call GPT-4, Claude, etc.
        # For demo, we'll use steadytext
        return steadytext.generate(f"Process this input: {input_text}")

    result = my_ai_function("test input")
    expected = steadytext.generate("Process this input: test input")
    assert result == expected  # This will always pass!
    print("✓ Test passed - deterministic AI output")


# Mock AI for testing
class MockAI:
    """Mock AI service for testing"""

    def complete(self, prompt):
        return steadytext.generate(prompt)

    def embed(self, text):
        return steadytext.embed(text)


# Test fixtures with deterministic data
def generate_test_user(user_id):
    """Generate consistent test user data"""
    return {
        "id": user_id,
        "bio": steadytext.generate(f"Write bio for user {user_id}"),
        "interests": steadytext.generate(f"List hobbies for user {user_id}"),
    }


# Fuzz testing with deterministic inputs
def generate_fuzz_input(test_name, iteration):
    """Generate reproducible fuzz test inputs"""
    return steadytext.generate(f"Fuzz input for {test_name} iteration {iteration}")


if __name__ == "__main__":
    # Run the test
    test_ai_function()

    # Create mock AI
    ai = MockAI()
    response = ai.complete("Hello, how are you?")
    print(f"\nMock AI response: {response[:50]}...")

    # Generate test data
    user = generate_test_user(123)
    print(f"\nTest user bio: {user['bio'][:80]}...")

    # Generate fuzz inputs
    for i in range(3):
        fuzz = generate_fuzz_input("parser_test", i)
        print(f"\nFuzz input {i}: {fuzz[:60]}...")
