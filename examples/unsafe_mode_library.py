#!/usr/bin/env python3
"""Example demonstrating the unsafe_mode parameter at the library level.

This shows how to use remote models without setting environment variables.
"""

import os
import steadytext


def main():
    print("=== SteadyText unsafe_mode Parameter Example ===\n")

    # Ensure environment variable is not set for demonstration
    os.environ.pop("STEADYTEXT_UNSAFE_MODE", None)

    # Example 1: Try to use remote model without unsafe_mode (will fail)
    print("1. Trying remote model without unsafe_mode:")
    result = steadytext.generate(
        "What is 2+2?",
        model="openai:gpt-4o-mini",
        unsafe_mode=False,  # This is the default
    )
    print(f"   Result: {result}")  # Should be None

    # Example 2: Use remote model with unsafe_mode=True
    print("\n2. Using remote model with unsafe_mode=True:")
    # Note: This would work if you have OPENAI_API_KEY set
    try:
        result = steadytext.generate(
            "What is 2+2?", model="openai:gpt-4o-mini", unsafe_mode=True
        )
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Error: {e}")
        print("   (This is expected if OPENAI_API_KEY is not set)")

    # Example 3: Local models work regardless of unsafe_mode
    print("\n3. Local models work with any unsafe_mode setting:")
    # Using default local model
    result1 = steadytext.generate("Hello", unsafe_mode=False)
    result2 = steadytext.generate("Hello", unsafe_mode=True)
    print(f"   unsafe_mode=False: {result1}")
    print(f"   unsafe_mode=True:  {result2}")

    # Example 4: Streaming with unsafe_mode
    print("\n4. Streaming with unsafe_mode:")
    tokens = list(
        steadytext.generate_iter(
            "Count to 3",
            model="openai:gpt-4o-mini",
            unsafe_mode=True,
            max_new_tokens=10,
        )
    )
    if tokens:
        print(f"   Tokens: {tokens}")
    else:
        print("   No tokens (remote model not available)")

    print("\n=== Key Points ===")
    print("- unsafe_mode=True enables remote models for a single call")
    print("- No need to set STEADYTEXT_UNSAFE_MODE environment variable")
    print("- Local models work the same regardless of unsafe_mode")
    print("- The parameter is available on generate() and generate_iter()")


if __name__ == "__main__":
    main()
