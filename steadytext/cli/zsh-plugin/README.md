# SteadyText ZSH Plugin

This directory contains ZSH plugins that integrate SteadyText's AI capabilities with your shell for intelligent command suggestions and autocompletion.

## Available Plugins

### 1. Basic Shell Completion
Use the `st completion` command to install standard shell completions for bash, zsh, or fish.

```bash
# Install completions for your current shell
st completion --install

# Or manually for a specific shell
st completion --shell zsh
```

### 2. Context-Aware Suggestions (`steadytext-context.plugin.zsh`)
Provides AI-powered command suggestions based on your current shell context.

**Features:**
- Gathers context: pwd, git status, last command, environment
- Triggered manually with `Ctrl-X Ctrl-S`
- Configurable via environment variables

**Installation:**
```bash
# Add to your .zshrc
source /path/to/steadytext/cli/zsh-plugin/steadytext-context.plugin.zsh

# Or with a plugin manager (oh-my-zsh)
# Copy to: ~/.oh-my-zsh/custom/plugins/steadytext-context/
```

**Configuration:**
```bash
# Enable/disable suggestions
export STEADYTEXT_SUGGEST_ENABLED=1

# Change trigger key (default: Ctrl-X Ctrl-S)
export STEADYTEXT_SUGGEST_KEY="^X^A"

# Model size (small/large)
export STEADYTEXT_SUGGEST_MODEL_SIZE="small"
```

### 3. Autosuggestions (`steadytext-autosuggestions.zsh`)
Fish-like autosuggestions powered by SteadyText AI. Shows suggestions as you type.

**Features:**
- Non-blocking async suggestions
- Suggestion caching for performance
- Multiple strategies: context, history, mixed
- Integrates with zsh-autosuggestions if available

**Installation:**
```bash
# Standalone
source /path/to/steadytext/cli/zsh-plugin/steadytext-autosuggestions.zsh

# With zsh-autosuggestions (recommended)
# Install zsh-autosuggestions first, then:
source /path/to/zsh-autosuggestions/zsh-autosuggestions.zsh
source /path/to/steadytext/cli/zsh-plugin/steadytext-autosuggestions.zsh
```

**Configuration:**
```bash
# Suggestion appearance
export STEADYTEXT_SUGGEST_HIGHLIGHT_STYLE="fg=240"

# Strategy: context, history, or mixed
export STEADYTEXT_SUGGEST_STRATEGY="context"

# Enable async suggestions (recommended)
export STEADYTEXT_SUGGEST_ASYNC=1

# Cache size
export STEADYTEXT_SUGGEST_CACHE_SIZE=100
```

**Usage:**
- Type commands normally
- Suggestions appear in gray
- Press `→` (right arrow) or `Tab` to accept
- Use `steadytext-suggest-toggle` to enable/disable
- Use `steadytext-suggest-clear-cache` to clear cache

## How It Works

1. **Context Gathering**: The plugin collects relevant shell context:
   - Current directory
   - Git repository status
   - Previous commands and exit codes
   - Environment variables
   - System information

2. **AI Processing**: Context is sent to SteadyText's generation model to predict the most likely next command

3. **Caching**: Suggestions are cached to improve performance and reduce API calls

4. **Display**: Suggestions are shown either:
   - On-demand (context plugin)
   - As you type (autosuggestions)

## Performance Considerations

- The `small` model is recommended for faster suggestions
- Async mode prevents blocking your shell
- Suggestions are cached to avoid repeated API calls
- Minimal context is gathered to reduce latency

## Privacy Note

The plugins send your shell context to the local SteadyText model. No data is sent to external servers. The context includes:
- Current directory path
- Command history (limited)
- Git branch/status
- Environment variable names (not values)

## Troubleshooting

1. **No suggestions appearing**
   - Ensure SteadyText is installed: `which st`
   - Check if daemon is running: `st daemon status`
   - Try manual generation: `echo "test" | st`

2. **Slow suggestions**
   - Use the `small` model size
   - Enable async mode
   - Check daemon status
   - Clear cache if it's too large

3. **Integration issues**
   - Check ZSH version: `echo $ZSH_VERSION` (5.0+ recommended)
   - Ensure plugin is sourced after other plugins
   - Check for conflicting key bindings

## Advanced Usage

### Custom Context Providers

You can extend the context gathering:

```bash
# Add custom context function
_my_custom_context() {
    echo "Project: $(basename $PWD)"
    # Add more context
}

# Override the context function
_steadytext_gather_context() {
    $(_my_custom_context)
    # ... existing context
}
```

### Project-Specific Suggestions

Create `.steadytext-context` in your project:

```bash
# .steadytext-context
echo "This is a Python project using Django"
echo "Common commands: manage.py runserver, pytest"
```

The plugin will include this in the context when in that directory.