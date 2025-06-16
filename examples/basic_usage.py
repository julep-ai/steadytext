#!/usr/bin/env python3
"""
Basic usage examples for SteadyText
"""

import steadytext

# Deterministic text generation
code = steadytext.generate("implement binary search in Python")
print(code[:200] + "...")  # Show first 200 chars
print()

# With log probabilities
text, logprobs = steadytext.generate("Explain quantum computing", return_logprobs=True)
print(f"Generated text: {text[:100]}...")
print(f"Has logprobs: {logprobs is not None}")
print()

# Streaming generation
print("Streaming example:")
for token in steadytext.generate_iter("explain recursion"):
    print(token, end="", flush=True)
print("\n")

# Deterministic embeddings
vec = steadytext.embed("Hello world")  # 1024-dim numpy array
print(f"Embedding shape: {vec.shape}")
print(f"Embedding first 5 values: {vec[:5]}")
print(f"Embedding is L2 normalized: {abs(sum(vec**2) - 1.0) < 0.001}")
