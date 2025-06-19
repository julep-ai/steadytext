#!/usr/bin/env python3
"""Test script for the new eos_string functionality."""

import steadytext

# Test 1: Default [EOS] behavior
print("Test 1: Default [EOS] behavior")
result1 = steadytext.generate("Tell me a story", eos_string="[EOS]")
print(f"Length: {len(result1)} chars")
print(f"First 100 chars: {result1[:100]}...")
print()

# Test 2: Custom eos_string
print("Test 2: Custom eos_string 'END'")
result2 = steadytext.generate("List items END more items", eos_string="END")
print(f"Result: {result2}")
print(f"Contains 'END': {'END' in result2}")
print()

# Test 3: Streaming with custom eos_string
print("Test 3: Streaming with custom eos_string")
tokens = []
for token in steadytext.generate_iter("Generate until STOP appears", eos_string="STOP"):
    tokens.append(token)
    if len(tokens) > 10:  # Limit for demo
        break
print(f"First 10 tokens: {''.join(tokens[:10])}")
print()

# Test 4: With logprobs
print("Test 4: With logprobs and custom eos_string")
text, logprobs = steadytext.generate("Hello world", return_logprobs=True, eos_string="[EOS]")
print(f"Text length: {len(text)}")
print(f"Has logprobs: {logprobs is not None}")
print()

# Test 5: Test that different eos_strings produce different outputs
print("Test 5: Different eos_strings produce different cache keys")
result_default = steadytext.generate("Test prompt for caching")
result_custom = steadytext.generate("Test prompt for caching", eos_string="CUSTOM_END")
print(f"Same result: {result_default == result_custom}")
print(f"Default length: {len(result_default)}")
print(f"Custom length: {len(result_custom)}")