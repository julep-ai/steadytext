# Contributing to SteadyText

We welcome contributions to SteadyText! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/steadytext.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Using UV (Recommended)

UV is a modern, blazing-fast Python package manager that's 10-100x faster than pip. We recommend using UV for development:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/your-username/steadytext.git
cd steadytext

# Install in development mode with all dependencies
uv sync --all-extras --dev

# Run tests
uv run python -m pytest

# Run linting and formatting
uvx ruff check .
uvx ruff format .
uvx black .
```

### Legacy Method (pip)

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest ruff black pre-commit

# Run tests
python -m pytest

# Run linting
ruff check .
```

## Making Changes

1. Write your code following the existing style
2. Add tests for new functionality
3. Ensure all tests pass
4. Update documentation as needed
5. Follow the project's commit message conventions

## Submitting Changes

1. Push your changes to your fork
2. Create a pull request with a clear description
3. Wait for review and address any feedback

## Code Style

- Follow PEP 8
- Use type hints where appropriate
- Keep functions focused and well-documented
- Add docstrings to all public functions

## Testing

- Write tests for all new functionality
- Ensure tests are deterministic
- Use pytest for testing

### Running Tests

**Using UV (recommended):**
```bash
# Run all tests
uv run python -m pytest

# Run with coverage
uv run python -m pytest --cov=steadytext

# Run specific test file
uv run python -m pytest tests/test_steadytext.py

# Allow model downloads in tests
STEADYTEXT_ALLOW_MODEL_DOWNLOADS=true uv run python -m pytest
```

**Legacy method:**
```bash
python -m pytest
```

## Questions?

Feel free to open an issue if you have questions or need help!