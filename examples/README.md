# SteadyText Examples

This directory contains example code demonstrating various use cases for SteadyText.

## Examples

### basic_usage.py
Core functionality demonstration:
- Text generation with and without logprobs
- Streaming generation
- Creating embeddings

### testing_with_ai.py
Using SteadyText for testing:
- Deterministic test assertions
- Mock AI services
- Test fixture generation
- Fuzz testing with reproducible inputs

### cli_tools.py
Building command-line tools:
- Motivational quotes
- Error message explanations
- Git command generation
- Click-based CLI examples

### content_generation.py
Content and data generation:
- ASCII art
- Game NPC dialogue
- Product reviews and user bios
- Auto-documentation
- Story generation
- Semantic cache keys

## Running the Examples

Each example can be run directly:

```bash
python examples/basic_usage.py
python examples/testing_with_ai.py
python examples/cli_tools.py
python examples/content_generation.py
```

The CLI tools example also supports command-line arguments:

```bash
python examples/cli_tools.py quote
python examples/cli_tools.py error ECONNREFUSED
python examples/cli_tools.py git "undo last commit"
```

## Note

All examples use deterministic generation, so running them multiple times will produce identical outputs. This is the core feature of SteadyText - predictable, reproducible AI outputs.