# Core Module Guidelines

This directory contains the core text generation and embedding functionality.

## Key Components

- `generator.py`: Deterministic text generation with model-based and fallback mechanisms.
- `embedder.py`: L2-normalized embeddings with zero-vector fallbacks.

## Critical Design Patterns

### Never Fails Principle
All functions must return valid outputs even when models fail to load:
- Text generation returns deterministic hash-based text on model failure
- Embeddings return zero vectors on model failure

### Deterministic Behavior
- All randomness must be seeded consistently
- Hash-based fallbacks ensure identical outputs for identical inputs
- Model parameters configured for deterministic sampling

## AIDEV Anchor Guidelines

Add `AIDEV-NOTE:` comments for:
- Complex algorithms (especially the hash-based fallback generator)
- Error handling and fallback mechanisms
- Model interaction points
- Dimension validation logic

Add `AIDEV-TODO:` for:
- Performance optimization opportunities
- Additional validation that could be helpful
- Error message improvements

Add `AIDEV-QUESTION:` for:
- Unclear model behavior or edge cases
- Potential issues with determinism
- Simple frecency cache only caching successful outputs.
