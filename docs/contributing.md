# Contributing to SteadyText

We welcome contributions to SteadyText! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**: `git clone https://github.com/your-username/steadytext.git`
3. **Create a feature branch**: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- Python 3.10+ (supports up to Python 3.13)
- Git
- Recommended: [uv](https://github.com/astral-sh/uv) for faster dependency management

### Installation

=== "With uv (Recommended)"

    ```bash
    # Clone the repository
    git clone https://github.com/julep-ai/steadytext.git
    cd steadytext

    # Install in development mode
    uv sync --dev

    # Activate the virtual environment
    source .venv/bin/activate  # Linux/Mac
    # or
    .venv\Scripts\activate     # Windows
    ```

=== "With pip"

    ```bash
    # Clone the repository
    git clone https://github.com/julep-ai/steadytext.git
    cd steadytext

    # Create virtual environment
    python -m venv .venv
    source .venv/bin/activate  # Linux/Mac
    # or
    .venv\Scripts\activate     # Windows

    # Install in development mode
    pip install -e .

    # Install development dependencies
    pip install -e .[dev]
    ```

### Development Commands

SteadyText uses [uv](https://github.com/astral-sh/uv) for task management:

```bash
# Run tests
uv run python -m pytest

# Run tests with coverage
uv run python -m pytest --cov=steadytext

# Run tests with model downloads (slower)
STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true uv run python -m pytest

# Run linting
uvx ruff check .

# Format code
uvx ruff format .

# Type checking
uvx mypy .

# Run pre-commit hooks
uvx pre-commit run --all-files
```

## Making Changes

### Code Style

- **Follow PEP 8**: Use `uvx ruff format .` to auto-format code
- **Use type hints**: Add type annotations for function parameters and returns
- **Add docstrings**: Document all public functions and classes
- **Keep functions focused**: Single responsibility principle

Example:

```python
def embed(text_input: Union[str, List[str]]) -> np.ndarray:
    """Create deterministic embeddings for text input.
    
    Args:
        text_input: String or list of strings to embed
        
    Returns:
        1024-dimensional L2-normalized float32 numpy array
    """
    # Implementation here
```

### Testing

SteadyText has comprehensive tests covering:

- **Deterministic behavior**: Same input → same output
- **Fallback functionality**: Works without models
- **Edge cases**: Empty inputs, invalid types
- **Performance**: Caching behavior

#### Writing Tests

```python
def test_your_feature():
    """Test your new feature."""
    # Test deterministic behavior
    result1 = your_function("test input")
    result2 = your_function("test input")
    assert result1 == result2  # Should be identical
    
    # Test edge cases
    result3 = your_function("")
    assert isinstance(result3, expected_type)
```

#### Running Tests

```bash
# Run all tests
uv run python -m pytest

# Run specific test file
uv run python -m pytest tests/test_your_feature.py

# Run with coverage
uv run python -m pytest --cov=steadytext

# Run tests that require model downloads
STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true uv run python -m pytest

# Run tests in parallel
uv run python -m pytest -n auto
```

### Documentation

- **Update API docs**: Modify files in `docs/api/` if adding new functions
- **Add examples**: Include usage examples in `docs/examples/`
- **Update README**: For major features, update the main README.md

### Architecture Guidelines

SteadyText follows a layered architecture:

```
steadytext/
├── core/          # Core generation and embedding logic
├── models/        # Model loading and caching
├── cli/           # Command-line interface
└── utils.py       # Shared utilities
```

#### Core Principles

1. **Never fail**: Functions should always return valid outputs
2. **Deterministic**: Same input always produces same output
3. **Thread-safe**: Support concurrent usage
4. **Cached**: Use frecency caching for performance

#### Adding New Features

1. **Core functionality**: Add to `steadytext/core/`
2. **Model support**: Modify `steadytext/models/`
3. **CLI commands**: Add to `steadytext/cli/commands/`
4. **Utilities**: Add to `steadytext/utils.py`

## Submitting Changes

### Before Submitting

1. **Run all tests**: `uv run python -m pytest`
2. **Check linting**: `uvx ruff check .`
3. **Format code**: `uvx ruff format .`
4. **Type check**: `uvx mypy .`
5. **Update documentation**: Add/update relevant docs

### Pull Request Process

1. **Create descriptive title**: "Add feature X" or "Fix bug Y"
2. **Write clear description**: Explain what changes and why
3. **Reference issues**: Link to related GitHub issues
4. **Add tests**: Include tests for new functionality
5. **Update changelog**: Add entry to CHANGELOG.md

### Pull Request Template

```markdown
## Description
Brief description of the changes

## Changes Made
- [ ] Added feature X
- [ ] Fixed bug Y
- [ ] Updated documentation

## Testing
- [ ] All tests pass
- [ ] Added tests for new functionality
- [ ] Manually tested edge cases

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changelog updated
```

## Development Workflow

### Typical Development Cycle

1. **Pick/create an issue**: Find something to work on
2. **Create feature branch**: `git checkout -b feature/issue-123`
3. **Make changes**: Implement your feature
4. **Test thoroughly**: Run tests and manual testing
5. **Commit changes**: Use descriptive commit messages
6. **Push and PR**: Create pull request

### Commit Messages

Follow conventional commits:

```bash
feat: add new embedding model support
fix: resolve caching issue with concurrent access
docs: update API documentation for generate()
test: add tests for edge cases
chore: update dependencies
```

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes  
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

## Release Process

SteadyText follows semantic versioning:

- **Major (1.0.0)**: Breaking changes, new model versions
- **Minor (0.1.0)**: New features, backward compatible
- **Patch (0.0.1)**: Bug fixes, small improvements

### Model Versioning

- Models are fixed per major version
- Only major version updates change model outputs
- This ensures deterministic behavior across patch/minor updates

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Discord**: Join our community chat (link in README)

### Common Issues

**Tests failing locally:**
```bash
# Clear caches
rm -rf ~/.cache/steadytext/

# Reinstall dependencies  
pip install -e .[dev]

# Run tests
poe test
```

**Import errors:**
```bash
# Make sure you're in the right directory
cd steadytext/

# Install in development mode
pip install -e .
```

**Model download issues:**
```bash
# Set environment variable
export STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true

# Run tests
poe test-models
```

## Code of Conduct

Please be respectful and constructive in all interactions. We want SteadyText to be a welcoming project for everyone.

## Recognition

Contributors are recognized in:
- **README.md**: Major contributors listed
- **CHANGELOG.md**: Contributions noted in releases
- **GitHub**: Contributor graphs and statistics

Thank you for contributing to SteadyText! 🚀