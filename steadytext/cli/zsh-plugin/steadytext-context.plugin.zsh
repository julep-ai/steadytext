#!/usr/bin/env zsh
# SteadyText Context-Aware Autocomplete Plugin for ZSH
# AIDEV-NOTE: This plugin provides intelligent command suggestions using SteadyText
# based on the current shell context (pwd, env vars, previous commands, etc.)

# Configuration
: ${STEADYTEXT_SUGGEST_ENABLED:=1}
: ${STEADYTEXT_SUGGEST_KEY:="^X^S"}  # Ctrl-X Ctrl-S to trigger suggestion
: ${STEADYTEXT_SUGGEST_MAX_CONTEXT:=500}  # Max chars of context to send
: ${STEADYTEXT_SUGGEST_MODEL_SIZE:="small"}  # Model size: small or large

# Function to gather shell context
_steadytext_gather_context() {
    local context=""
    
    # Current directory
    context+="Current directory: $(pwd)\n"
    
    # Current date/time
    context+="Date: $(date '+%Y-%m-%d %H:%M:%S')\n"
    
    # Current user and host
    context+="User: ${USER}@$(hostname)\n"
    
    # Git status if in a git repo
    if git rev-parse --git-dir &>/dev/null; then
        local branch=$(git branch --show-current 2>/dev/null)
        context+="Git branch: ${branch}\n"
        local status=$(git status --porcelain 2>/dev/null | head -5)
        if [[ -n "$status" ]]; then
            context+="Git status (first 5):\n${status}\n"
        fi
    fi
    
    # Last command and its exit status
    local last_cmd=$(fc -ln -1 2>/dev/null | sed 's/^[ \t]*//')
    local last_exit=$?
    if [[ -n "$last_cmd" ]]; then
        context+="Last command: ${last_cmd}\n"
        context+="Last exit code: ${last_exit}\n"
    fi
    
    # Current command buffer
    context+="Current buffer: ${BUFFER}\n"
    
    # Relevant environment variables
    [[ -n "$VIRTUAL_ENV" ]] && context+="Virtual env: $(basename $VIRTUAL_ENV)\n"
    [[ -n "$CONDA_DEFAULT_ENV" ]] && context+="Conda env: $CONDA_DEFAULT_ENV\n"
    
    # System info
    context+="OS: $(uname -s)\n"
    
    # Network status (simplified)
    if command -v ip &>/dev/null; then
        local net_status=$(ip link | grep -E "^[0-9]+: " | grep -v "lo:" | grep "state UP" | wc -l)
        context+="Network interfaces up: ${net_status}\n"
    fi
    
    # Limit context size
    if [[ ${#context} -gt $STEADYTEXT_SUGGEST_MAX_CONTEXT ]]; then
        context="${context:0:$STEADYTEXT_SUGGEST_MAX_CONTEXT}..."
    fi
    
    echo "$context"
}

# Function to get command suggestion from SteadyText
_steadytext_get_suggestion() {
    local context="$1"
    local prompt="Based on this shell context, suggest the most likely next command the user wants to run. Only output the command, nothing else:\n\n${context}"
    
    # Ensure daemon is running for faster responses
    # AIDEV-NOTE: The daemon check is silent to avoid cluttering shell output
    st daemon status &>/dev/null || st daemon start &>/dev/null
    
    # Use steadytext to generate suggestion
    local suggestion=$(echo "$prompt" | st --size "$STEADYTEXT_SUGGEST_MODEL_SIZE" --quiet 2>/dev/null | head -1)
    
    # Clean up the suggestion (remove leading/trailing whitespace)
    suggestion=$(echo "$suggestion" | sed 's/^[ \t]*//;s/[ \t]*$//')
    
    echo "$suggestion"
}

# Widget to insert SteadyText suggestion
_steadytext_suggest_widget() {
    if [[ $STEADYTEXT_SUGGEST_ENABLED -eq 0 ]]; then
        return
    fi
    
    # Show thinking indicator
    zle -M "Thinking..."
    
    # Gather context
    local context=$(_steadytext_gather_context)
    
    # Get suggestion
    local suggestion=$(_steadytext_get_suggestion "$context")
    
    if [[ -n "$suggestion" ]]; then
        # Insert suggestion at cursor position
        LBUFFER+="$suggestion"
        zle -M "Suggestion inserted: $suggestion"
    else
        zle -M "No suggestion available"
    fi
}

# Function to show inline suggestion (non-intrusive)
_steadytext_show_inline_suggestion() {
    if [[ $STEADYTEXT_SUGGEST_ENABLED -eq 0 ]] || [[ -z "$BUFFER" ]]; then
        return
    fi
    
    # Only suggest if buffer ends with a space and has content
    if [[ "$BUFFER" =~ ' $' ]] && [[ ${#BUFFER} -gt 1 ]]; then
        local context=$(_steadytext_gather_context)
        local suggestion=$(_steadytext_get_suggestion "$context")
        
        if [[ -n "$suggestion" ]]; then
            # Store suggestion for potential acceptance
            _STEADYTEXT_LAST_SUGGESTION="$suggestion"
            
            # Show as grayed out text (using ZSH's completion system)
            # This is a simplified version - full implementation would use zle highlighting
            zle -M "→ $suggestion (Press Tab to accept)"
        fi
    fi
}

# Widget to accept the last suggestion
_steadytext_accept_suggestion() {
    if [[ -n "$_STEADYTEXT_LAST_SUGGESTION" ]]; then
        LBUFFER+="$_STEADYTEXT_LAST_SUGGESTION"
        _STEADYTEXT_LAST_SUGGESTION=""
        zle -M ""
    fi
}

# Register the widgets
zle -N steadytext-suggest _steadytext_suggest_widget
zle -N steadytext-accept-suggestion _steadytext_accept_suggestion

# Bind keys
bindkey "$STEADYTEXT_SUGGEST_KEY" steadytext-suggest

# Optional: Hook into the line editor to show suggestions automatically
# (This is commented out by default as it might be too intrusive)
# add-zsh-hook precmd _steadytext_show_inline_suggestion

# Helper function to enable/disable suggestions
steadytext-toggle() {
    if [[ $STEADYTEXT_SUGGEST_ENABLED -eq 1 ]]; then
        STEADYTEXT_SUGGEST_ENABLED=0
        echo "SteadyText suggestions disabled"
    else
        STEADYTEXT_SUGGEST_ENABLED=1
        echo "SteadyText suggestions enabled"
    fi
}

# Add to fpath for additional completion support
fpath=(${0:h}/completions $fpath)

# AIDEV-NOTE: This plugin can be extended with:
# - Caching of suggestions for similar contexts
# - Learning from accepted/rejected suggestions
# - Integration with shell history for better predictions
# - Custom prompts for different types of tasks