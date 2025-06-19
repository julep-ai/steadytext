# SteadyText Documentation

Welcome to the SteadyText documentation. This directory contains comprehensive guides and references for using SteadyText.

## Documentation Index

### API Reference
- **[API Documentation](api.md)** - Complete Python API reference with function signatures, parameters, and examples

### Features and Implementation
- **[EOS String Implementation](eos-string-implementation.md)** - Detailed guide on custom end-of-sequence string functionality

## Quick Links

### Getting Started
```python
import steadytext

# Basic text generation
text = steadytext.generate("Write a Python function")

# Streaming generation
for token in steadytext.generate_iter("Tell me a story"):
    print(token, end="", flush=True)

# Embeddings
vec = steadytext.embed("Hello world")  # 1024-dim numpy array
```

### CLI Usage
```bash
# Basic usage
steadytext "Generate some text"

# With custom EOS string
steadytext "Generate until DONE" --eos-string "DONE"

# Streaming mode
steadytext "Stream text" --stream

# Quiet mode (no logs)
steadytext --quiet "Generate without logs"
```

## Key Features

- **Deterministic**: Same input always produces the same output
- **Zero Configuration**: Works out of the box with automatic model downloads
- **Never Fails**: Graceful fallbacks ensure your code never breaks
- **Concurrent Safe**: SQLite-based caching supports multi-process/thread usage
- **CLI Ready**: Full-featured command-line interface

## Documentation Structure

This documentation is organized to help you find information quickly:

1. **API Reference** - For developers integrating SteadyText into their projects
2. **Feature Guides** - Detailed explanations of specific functionality
3. **Examples** - Practical usage examples and patterns

## Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/julep-ai/steadytext/issues)
- **Project Home**: [SteadyText on GitHub](https://github.com/julep-ai/steadytext)