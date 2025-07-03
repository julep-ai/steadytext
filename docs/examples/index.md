# Examples

Real-world usage patterns and code examples for SteadyText.

## Overview

This section demonstrates practical applications of SteadyText across different use cases:

- **[Testing with AI](testing.md)** - Reliable AI tests that never flake
- **[CLI Tools](tooling.md)** - Building deterministic command-line tools
- **[Caching Guide](caching.md)** - Configure and optimize caching
- **[Custom Seeds Guide](custom-seeds.md)** - Use custom seeds for reproducible variations
- **[Daemon Usage Guide](daemon-usage.md)** - Persistent model serving for faster responses
- **[Error Handling Guide](error-handling.md)** - Handle errors gracefully
- **[Performance Tuning Guide](performance-tuning.md)** - Optimize for speed and efficiency
- **[PostgreSQL Integration Examples](postgresql-integration.md)** - Integrate with PostgreSQL

All examples showcase SteadyText's core principle: **same input → same output, every time**.

## Quick Examples

### Basic Usage

```python
import steadytext

# Deterministic text generation
code = steadytext.generate("implement binary search in Python")
assert "def binary_search" in code  # Always passes!

# Streaming generation
for token in steadytext.generate_iter("explain quantum computing"):
    print(token, end="", flush=True)

# Deterministic embeddings  
vec = steadytext.embed("Hello world")  # 1024-dim numpy array
print(f"Shape: {vec.shape}, Norm: {np.linalg.norm(vec):.6f}")
```

### Testing Applications

```python
def test_ai_code_generation():
    """Test that never flakes - same input, same output."""
    prompt = "write a function to reverse a string"
    result = my_ai_function(prompt)
    expected = steadytext.generate(prompt)
    assert result == expected  # Deterministic comparison!

def test_embedding_similarity():
    """Reliable similarity testing."""
    vec1 = steadytext.embed("machine learning")
    vec2 = steadytext.embed("artificial intelligence")
    similarity = np.dot(vec1, vec2)  # Already normalized
    assert similarity > 0.7  # Always passes with same threshold
```

### CLI Tool Building

```python
import click
import steadytext

@click.command()
@click.argument('topic')
def motivate(topic):
    """Generate motivational quotes about any topic."""
    prompt = f"Write an inspiring quote about {topic}"
    quote = steadytext.generate(prompt)
    click.echo(f"💪 {quote}")

# Usage: python script.py "programming"
# Always generates the same motivational quote for "programming"
```

## Use Case Categories

### 🧪 Testing & Quality Assurance

Perfect for:
- Unit tests with AI components
- Integration testing with deterministic outputs
- Regression testing for AI features
- Mock AI services for development

### 🛠️ Developer Tools

Ideal for:
- Code generation tools
- Documentation generators  
- CLI utilities with AI features
- Build system integration

### 📊 Data & Content Generation

Great for:
- Synthetic data generation
- Content templates
- Data augmentation for testing
- Reproducible research datasets

### 🔍 Search & Similarity

Excellent for:
- Semantic search systems
- Document clustering
- Content recommendation
- Duplicate detection

## Getting Started

1. **Browse examples** - Check out [Testing](testing.md) and [CLI Tools](tooling.md)
2. **Run the code** - All examples are fully executable
3. **Adapt for your use case** - Copy and modify patterns that fit your needs

## Example Repository

All examples are available in the [examples/ directory](https://github.com/julep-ai/steadytext/tree/main/examples) of the SteadyText repository:

```bash
git clone https://github.com/julep-ai/steadytext.git
cd steadytext/examples
python basic_usage.py
python testing_with_ai.py  
python cli_tools.py
```

!!! tip "Deterministic Outputs"
    Remember: all examples produce identical outputs every time you run them. This predictability is SteadyText's core feature and what makes it perfect for testing and tooling applications.