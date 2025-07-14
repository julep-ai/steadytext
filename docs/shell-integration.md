# Shell Integration

SteadyText provides powerful shell integration features that enhance your command-line experience with AI-powered suggestions and intelligent autocompletion. From basic tab completion to context-aware AI suggestions, these integrations make your shell smarter and more productive.

## Overview

SteadyText offers three levels of shell integration:

1. **Basic Shell Completion**: Standard tab completion for commands and options (bash, zsh, fish)
2. **Context-Aware Suggestions**: AI-powered command suggestions based on your current context (ZSH only)
3. **AI Autosuggestions**: Real-time, fish-like autosuggestions powered by AI (ZSH only)

## Basic Shell Completion

Basic shell completion provides tab completion for all SteadyText commands and options. It works with both `st` and `steadytext` command names.

### Supported Shells

- **Bash** (4.4+)
- **Zsh** (5.0+)
- **Fish** (3.0+)

### Quick Installation

The easiest way to install completions is using the `--install` flag:

```bash
# Auto-detect shell and install
st completion --install

# Install for specific shell
st completion --shell zsh --install
```

### Manual Installation

#### Bash

```bash
# Create completion directory
mkdir -p ~/.local/share/bash-completion/completions

# Generate completion script
eval "$(_ST_COMPLETE=bash_source st)" > ~/.local/share/bash-completion/completions/st
eval "$(_STEADYTEXT_COMPLETE=bash_source steadytext)" > ~/.local/share/bash-completion/completions/steadytext

# Reload shell
source ~/.bashrc
```

#### Zsh

Add to your `~/.zshrc`:

```bash
# SteadyText completions
eval "$(_ST_COMPLETE=zsh_source st)"
eval "$(_STEADYTEXT_COMPLETE=zsh_source steadytext)"
```

Then reload:
```bash
source ~/.zshrc
```

#### Fish

```bash
# Create completion directory
mkdir -p ~/.config/fish/completions

# Generate completion scripts
_ST_COMPLETE=fish_source st > ~/.config/fish/completions/st.fish
_STEADYTEXT_COMPLETE=fish_source steadytext > ~/.config/fish/completions/steadytext.fish
```

### Usage

Once installed, use Tab to complete commands:

```bash
st gene<Tab>          # Completes to: st generate
st generate --he<Tab> # Completes to: st generate --help
st embed --for<Tab>   # Shows format options
```

## ZSH Plugins (Advanced Features)

For ZSH users, SteadyText provides advanced AI-powered plugins that go beyond basic completion.

### Installation Methods

#### Quick Install (Interactive)

```bash
# Run the interactive installer
bash /path/to/steadytext/cli/zsh-plugin/install.sh
```

#### Oh My Zsh

```bash
# Clone to custom plugins directory
git clone https://github.com/julep-ai/steadytext.git ~/.oh-my-zsh/custom/plugins/steadytext-temp
cp -r ~/.oh-my-zsh/custom/plugins/steadytext-temp/steadytext/cli/zsh-plugin ~/.oh-my-zsh/custom/plugins/steadytext-context
cp -r ~/.oh-my-zsh/custom/plugins/steadytext-temp/steadytext/cli/zsh-plugin ~/.oh-my-zsh/custom/plugins/steadytext-autosuggestions

# Add to .zshrc
plugins=(... steadytext-context steadytext-autosuggestions)
```

#### Manual Installation

Add to your `~/.zshrc`:

```bash
# Context-aware suggestions
source /path/to/steadytext/cli/zsh-plugin/steadytext-context.plugin.zsh

# AI autosuggestions (optional)
source /path/to/steadytext/cli/zsh-plugin/steadytext-autosuggestions.zsh
```

### Context-Aware Suggestions Plugin

This plugin provides on-demand AI suggestions based on your shell context.

#### Features

- **Manual Trigger**: Press `Ctrl-X Ctrl-S` to get suggestions
- **Smart Context**: Considers pwd, git status, last command, environment
- **Non-intrusive**: Only activates when you request it
- **Project Awareness**: Reads `.steadytext-context` files

#### Usage

1. Navigate to any directory:
   ```bash
   cd ~/projects/my-app
   ```

2. Press `Ctrl-X Ctrl-S`

3. SteadyText analyzes your context and suggests relevant commands:
   ```bash
   # In a Python project with uncommitted changes:
   git add . && git commit -m "Update dependencies"
   pytest tests/
   python manage.py runserver
   ```

#### Configuration

```bash
# Enable/disable plugin
export STEADYTEXT_SUGGEST_ENABLED=1

# Change trigger key binding
export STEADYTEXT_SUGGEST_KEY="^X^A"  # Ctrl-X Ctrl-A

# Model size (small = faster, large = smarter)
export STEADYTEXT_SUGGEST_MODEL_SIZE=small

# Disable in specific directories
export STEADYTEXT_SUGGEST_IGNORE_DIRS="/tmp,/private"
```

### AI Autosuggestions Plugin

This plugin provides fish-like autosuggestions powered by AI, showing predictions as you type.

#### Features

- **Real-time Suggestions**: Shows AI predictions in gray as you type
- **Async Processing**: Non-blocking for smooth typing experience
- **Smart Caching**: Remembers previous suggestions for speed
- **Multiple Strategies**: Context-based, history-based, or mixed
- **Integration**: Works with zsh-autosuggestions if installed

#### Usage

1. Start typing any command:
   ```bash
   git st[atus]  # Gray suggestion appears
   ```

2. Accept suggestions:
   - Press `→` (right arrow) to accept the whole suggestion
   - Press `Tab` to accept the next word
   - Keep typing to ignore

3. Control suggestions:
   ```bash
   # Toggle on/off
   steadytext-suggest-toggle
   
   # Clear suggestion cache
   steadytext-suggest-clear-cache
   ```

#### Configuration

```bash
# Visual style
export STEADYTEXT_SUGGEST_HIGHLIGHT_STYLE="fg=240"  # Gray color

# Suggestion strategy
export STEADYTEXT_SUGGEST_STRATEGY=context  # context, history, or mixed

# Performance
export STEADYTEXT_SUGGEST_ASYNC=1           # Enable async mode
export STEADYTEXT_SUGGEST_CACHE_SIZE=100    # Number of cached suggestions

# Buffer settings
export STEADYTEXT_SUGGEST_BUFFER=         # Show immediate suggestions
export STEADYTEXT_SUGGEST_USE_ASYNC=1     # Use async fetching
```

## Project-Specific Context

Create a `.steadytext-context` file in your project root to provide project-specific hints:

```bash
# .steadytext-context
echo "Django project using Python 3.11"
echo "Database: PostgreSQL with PostGIS"
echo "Common commands: make test, make migrate, make run"
echo "Deployment: docker-compose up -d"
```

The context-aware plugin will use this information to provide more relevant suggestions.

## Configuration Reference

### Global Settings

```bash
# Enable/disable all shell integrations
export STEADYTEXT_SUGGEST_ENABLED=1

# Model configuration
export STEADYTEXT_SUGGEST_MODEL_SIZE=small  # small or large
export STEADYTEXT_MAX_CONTEXT_WINDOW=2048   # Context size for suggestions

# Performance
export STEADYTEXT_SUGGEST_TIMEOUT=2         # Timeout in seconds
export STEADYTEXT_DAEMON_HOST=localhost     # Daemon connection
export STEADYTEXT_DAEMON_PORT=5557
```

### Plugin-Specific Settings

#### Context Plugin
```bash
export STEADYTEXT_SUGGEST_KEY="^X^S"        # Trigger key
export STEADYTEXT_SUGGEST_SHOW_CONTEXT=0    # Show gathered context
```

#### Autosuggestions Plugin
```bash
export STEADYTEXT_SUGGEST_STRATEGY=context  # Suggestion strategy
export STEADYTEXT_SUGGEST_ASYNC=1           # Async processing
export STEADYTEXT_SUGGEST_HIGHLIGHT_STYLE="fg=240"
```

## Usage Examples

### Example 1: Development Workflow

```bash
# In a git repository with changes
$ git st<Ctrl-X Ctrl-S>
# Suggests: git status, git stash, git stage

# After viewing status
$ <Ctrl-X Ctrl-S>
# Suggests: git add -A, git diff --cached, git commit -m "..."

# In a Python project
$ py<Tab>
# Completes: pytest, python, pyenv

# With autosuggestions enabled
$ pytest t[ests/ -v --cov]  # Shows in gray
```

### Example 2: System Administration

```bash
# In /var/log
$ tail -f <Ctrl-X Ctrl-S>
# Suggests: tail -f syslog, tail -f nginx/error.log

# Service management
$ systemctl <Tab>
# Shows: start, stop, restart, status, enable, disable

# With context awareness
$ <Ctrl-X Ctrl-S>
# Suggests: systemctl status nginx, journalctl -u nginx -f
```

### Example 3: Data Science Workflow

```bash
# In Jupyter project
$ jupyter <Tab>
# Completes: notebook, lab, console

# With AI suggestions
$ <Ctrl-X Ctrl-S>
# Suggests: jupyter lab --no-browser, python analysis.py, git status
```

## Privacy and Security

### Local Processing

All AI processing happens locally using SteadyText's models:
- No data is sent to external servers
- Context gathering is minimal and privacy-conscious
- Suggestions are generated on your machine

### Context Collection

The plugins collect limited context:
- Current directory path (not file contents)
- Git branch and status (not commit messages)
- Last few commands (not their output)
- Environment variable names (not values)
- System type (Linux/Mac)

### Opting Out

Disable specific context collection:

```bash
# Disable git context
export STEADYTEXT_SUGGEST_NO_GIT=1

# Disable command history
export STEADYTEXT_SUGGEST_NO_HISTORY=1

# Disable entirely
export STEADYTEXT_SUGGEST_ENABLED=0
```

## Troubleshooting

### No Suggestions Appearing

1. **Check SteadyText installation**:
   ```bash
   which st
   st --version
   ```

2. **Verify daemon is running**:
   ```bash
   st daemon status
   st daemon start  # If not running
   ```

3. **Test basic generation**:
   ```bash
   echo "test" | st
   ```

4. **Check plugin is loaded**:
   ```bash
   echo $STEADYTEXT_SUGGEST_ENABLED
   type steadytext-suggest-widget  # Should show function
   ```

### Slow Suggestions

1. **Use small model**:
   ```bash
   export STEADYTEXT_SUGGEST_MODEL_SIZE=small
   ```

2. **Enable async mode**:
   ```bash
   export STEADYTEXT_SUGGEST_ASYNC=1
   ```

3. **Check daemon performance**:
   ```bash
   st daemon status
   ```

4. **Clear cache if too large**:
   ```bash
   steadytext-suggest-clear-cache
   ```

### Key Binding Conflicts

1. **Check existing bindings**:
   ```bash
   bindkey | grep "^X^S"
   ```

2. **Use alternative binding**:
   ```bash
   export STEADYTEXT_SUGGEST_KEY="^X^A"
   ```

3. **Unbind conflicts**:
   ```bash
   bindkey -r "^X^S"  # Remove existing
   ```

### Integration Issues

1. **ZSH version**:
   ```bash
   echo $ZSH_VERSION  # Need 5.0+
   ```

2. **Plugin load order**: Ensure SteadyText plugins load after other plugins in `.zshrc`

3. **Environment issues**:
   ```bash
   # Add to .zshrc before plugin source
   export PATH="$PATH:$(python3 -m site --user-base)/bin"
   ```

## Performance Tips

### Optimize Suggestion Speed

1. **Use the daemon**:
   ```bash
   st daemon start
   st daemon status  # Verify running
   ```

2. **Preload models**:
   ```bash
   st models preload
   ```

3. **Limit context gathering**:
   ```bash
   export STEADYTEXT_SUGGEST_MAX_HISTORY=10  # Limit history items
   export STEADYTEXT_SUGGEST_NO_GIT=1        # Skip git if not needed
   ```

### Reduce Memory Usage

1. **Use small model**:
   ```bash
   export STEADYTEXT_SUGGEST_MODEL_SIZE=small
   ```

2. **Limit cache size**:
   ```bash
   export STEADYTEXT_SUGGEST_CACHE_SIZE=50
   ```

3. **Clear cache periodically**:
   ```bash
   # Add to .zshrc
   alias stclear="steadytext-suggest-clear-cache"
   ```

## Advanced Customization

### Custom Context Providers

Extend context gathering with custom functions:

```bash
# Add to .zshrc after loading plugin
_my_custom_context() {
    # Add Kubernetes context
    if command -v kubectl &> /dev/null; then
        echo "k8s: $(kubectl config current-context 2>/dev/null || echo 'none')"
    fi
    
    # Add Docker info
    if command -v docker &> /dev/null; then
        echo "docker: $(docker ps -q | wc -l) containers running"
    fi
}

# Hook into existing context function
_steadytext_gather_context_custom() {
    _steadytext_gather_context_original
    _my_custom_context
}
alias _steadytext_gather_context_original=_steadytext_gather_context
alias _steadytext_gather_context=_steadytext_gather_context_custom
```

### Custom Suggestion Filtering

Filter or modify suggestions before display:

```bash
# Add to .zshrc
_steadytext_filter_suggestion() {
    local suggestion="$1"
    
    # Block dangerous commands
    if [[ "$suggestion" =~ "rm -rf /" ]]; then
        echo "# Command blocked for safety"
        return
    fi
    
    # Add sudo if needed
    if [[ "$PWD" == "/etc"* ]] && [[ ! "$suggestion" =~ ^sudo ]]; then
        echo "sudo $suggestion"
    else
        echo "$suggestion"
    fi
}
```

## See Also

- [CLI Reference](api/cli.md) - Complete command reference
- [Configuration](api/configuration.md) - All configuration options
- [Troubleshooting](troubleshooting.md) - General troubleshooting guide
- [ZSH Plugin README](https://github.com/julep-ai/steadytext/blob/main/steadytext/cli/zsh-plugin/README.md) - Detailed plugin documentation