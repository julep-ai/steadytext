#!/usr/bin/env zsh
# Example configuration for SteadyText ZSH plugins
# Copy relevant sections to your ~/.zshrc

# ============================================
# Basic Configuration
# ============================================

# Enable/disable AI suggestions globally
export STEADYTEXT_SUGGEST_ENABLED=1

# Model size: "small" (faster) or "large" (more accurate)
export STEADYTEXT_SUGGEST_MODEL_SIZE="small"

# ============================================
# Context-Aware Suggestions Plugin
# ============================================

# Key binding for manual suggestions (default: Ctrl-X Ctrl-S)
export STEADYTEXT_SUGGEST_KEY="^X^S"

# Maximum context size to send (in characters)
export STEADYTEXT_SUGGEST_MAX_CONTEXT=500

# Load the plugin
source /path/to/steadytext/cli/zsh-plugin/steadytext-context.plugin.zsh

# ============================================
# Autosuggestions Plugin
# ============================================

# Suggestion appearance (ANSI color codes)
# Common values: fg=240 (gray), fg=8 (bright black), fg=245 (light gray)
export STEADYTEXT_SUGGEST_HIGHLIGHT_STYLE="fg=240"

# Suggestion strategy:
# - "context": AI-based suggestions using current context
# - "history": Use command history only (faster)
# - "mixed": Try history first, fall back to AI
export STEADYTEXT_SUGGEST_STRATEGY="mixed"

# Enable async suggestions (recommended for performance)
export STEADYTEXT_SUGGEST_ASYNC=1

# Cache size (number of suggestions to remember)
export STEADYTEXT_SUGGEST_CACHE_SIZE=100

# Load the plugin
source /path/to/steadytext/cli/zsh-plugin/steadytext-autosuggestions.zsh

# ============================================
# Integration with Oh My Zsh
# ============================================

# If using Oh My Zsh, add to your plugins list instead:
# plugins=(
#     git
#     docker
#     steadytext-context
#     steadytext-autosuggestions
#     # ... other plugins
# )

# ============================================
# Custom Functions
# ============================================

# Toggle suggestions on/off quickly
alias st-toggle='steadytext-suggest-toggle'

# Clear suggestion cache
alias st-clear='steadytext-suggest-clear-cache'

# Function to get AI help for current command
ai-help() {
    local cmd="${1:-$BUFFER}"
    echo "Explain this command: $cmd" | st --quiet
}

# Function to fix the last failed command
ai-fix() {
    local last_cmd=$(fc -ln -1)
    echo "The command '$last_cmd' failed with exit code $?. Suggest a fix:" | st --quiet
}

# ============================================
# Project-Specific Contexts
# ============================================

# Automatically load project contexts
precmd_steadytext_context() {
    if [[ -f .steadytext-context ]]; then
        export STEADYTEXT_PROJECT_CONTEXT=$(<.steadytext-context)
    else
        unset STEADYTEXT_PROJECT_CONTEXT
    fi
}
add-zsh-hook precmd precmd_steadytext_context

# ============================================
# Performance Optimizations
# ============================================

# Disable suggestions in large git repos
precmd_steadytext_performance() {
    if [[ -d .git ]] && [[ $(git rev-list --count HEAD 2>/dev/null) -gt 10000 ]]; then
        export STEADYTEXT_SUGGEST_ENABLED=0
    fi
}
# add-zsh-hook precmd precmd_steadytext_performance  # Uncomment to enable

# ============================================
# Debugging
# ============================================

# Enable debug output (shows context being sent)
# export STEADYTEXT_DEBUG=1

# Log suggestions to file for analysis
# export STEADYTEXT_LOG_FILE="$HOME/.steadytext-suggestions.log"