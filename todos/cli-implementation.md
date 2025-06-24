# SteadyText CLI Implementation Plan

## Overview
Create a command-line interface for SteadyText that provides instant, deterministic text generation with persistent caching. The CLI will be available as `steadytext` with a shorter alias `st`.

## Core Features

### 1. Basic Text Generation
```bash
echo "prompt" | st           # Generate text from prompt
st -                        # Read prompt from stdin
echo "prompt" | st          # Pipeline support
```

### 2. Output Modes
```bash
echo "prompt" | st           # Raw output (default, streaming)
echo "prompt" | st --wait    # Wait for full output before displaying
echo "prompt" | st generate --json     # JSON output with metadata
echo "prompt" | st generate --logprobs # Include log probabilities
```

### 3. Embedding Generation
```bash
st embed "text"             # Generate embedding vector
st embed "text" --format numpy  # Output as numpy array
st embed "text" --format json   # Output as JSON array
st embed "text" --format hex    # Output as hex string
```

### 4. Cache Management
```bash
st cache stats              # Show cache statistics
st cache clear              # Clear all caches
st cache clear --generation # Clear only generation cache
st cache clear --embedding  # Clear only embedding cache
st cache export <file>      # Export cache to file
st cache import <file>      # Import cache from file
```

### 5. Utility Commands
```bash
st --version               # Show version
st --help                  # Show help
st models status           # Check model download status
st models download         # Pre-download models
st models path             # Show model cache directory
```

## Implementation Details

### Project Structure
```
steadytext/
├── cli/
│   ├── __init__.py
│   ├── __main__.py        # Entry point for `python -m steadytext.cli`
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── generate.py    # Text generation command
│   │   ├── embed.py       # Embedding command
│   │   ├── cache.py       # Cache management commands
│   │   └── models.py      # Model management commands
│   └── utils.py           # CLI utilities (formatting, etc.)
└── ...
```

### Entry Point Configuration
Update `pyproject.toml` or `setup.py`:
```toml
[project.scripts]
steadytext = "steadytext.cli:main"
st = "steadytext.cli:main"
```

### Dependencies
- `click` or `typer` for CLI framework
- `rich` for enhanced terminal output (optional)
- Existing steadytext core functionality

### Key Implementation Considerations

1. **Streaming Output**: Use `generate_iter()` for `--stream` mode
2. **Pipeline Support**: Detect stdin input for Unix pipeline compatibility
3. **JSON Output**: Include metadata like generation time, cache hit, model info
4. **Error Handling**: Graceful fallback messages, never crash
5. **Performance**: Minimize import time for snappy CLI feel

## Advanced Features (Phase 2)

### 1. Templates and Presets
```bash
st template list                    # List available templates
st template add <name> <prompt>     # Save a template
st template use <name> [args...]    # Use template with args
st @explain "chmod 755"             # Shorthand for common templates
```

### 2. Shell Integration
```bash
# Generate .bashrc/.zshrc functions
st shell-integration --shell bash >> ~/.bashrc
st shell-integration --shell zsh >> ~/.zshrc

# This would enable:
gitdo() { $(echo "git command to $*" | st --wait); }
howto() { echo "how to $*" | st; }
```

### 3. Configuration File
```bash
# ~/.config/steadytext/config.yaml
templates:
  explain: "Explain what {0} does in simple terms"
  gitdo: "Git command to {0}"
  
aliases:
  g: generate
  e: embed
  
defaults:
  format: raw
  stream: false
```

### 4. Batch Processing
```bash
st batch process <input.txt> --output <output.jsonl>
st batch embed <documents.txt> --output <embeddings.npy>
```

## Testing Strategy

1. **Unit Tests**: Test each command in isolation
2. **Integration Tests**: Test full CLI workflows
3. **Cache Tests**: Verify cache persistence across invocations
4. **Performance Tests**: Ensure instant response for cached queries
5. **Shell Tests**: Test pipeline integration and shell compatibility

## Documentation Plan

1. **README Update**: Add CLI section with examples
2. **Man Page**: Generate man page for `st` command
3. **Shell Completion**: Provide bash/zsh completion scripts
4. **Tutorial**: Step-by-step guide for common use cases

## Timeline Estimate

### Phase 1: Core Implementation (1-2 days)
- Basic command structure
- Generate and embed commands
- Cache management
- Basic tests

### Phase 2: Enhanced Features (2-3 days)
- Templates system
- Shell integration
- Configuration file
- Advanced output formats

### Phase 3: Polish (1 day)
- Documentation
- Shell completions
- Performance optimization
- Package release

## Success Metrics

1. **Instant Response**: Cached queries return in <10ms
2. **Zero Configuration**: Works immediately after `pip install`
3. **Unix Philosophy**: Plays well with pipes and scripts
4. **Reliability**: Never crashes, always returns something
5. **Developer Experience**: Intuitive commands, helpful errors

## Example Use Cases to Validate

```bash
# Command generation
echo "ls command to find .banana files recursively" | st

# Error explanation
systemctl status nginx | st  # Will use stdin as context

# Config generation
echo "nginx config for reverse proxy port 3000" | st > nginx.conf

# Batch processing
find . -name "*.py" | xargs -I {} sh -c 'echo "summarize {}" | st' > summaries.txt

# Interactive shell helper
alias '??'='st'
?? "how to check disk usage by directory"
```