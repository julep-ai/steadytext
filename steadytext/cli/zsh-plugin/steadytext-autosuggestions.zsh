#!/usr/bin/env zsh
# SteadyText Autosuggestions for ZSH
# AIDEV-NOTE: Provides fish-like autosuggestions powered by SteadyText AI
# This integrates with zsh-autosuggestions if available, or works standalone

# Colors and styling
: ${STEADYTEXT_SUGGEST_HIGHLIGHT_STYLE:="fg=240"}  # Gray color for suggestions
: ${STEADYTEXT_SUGGEST_STRATEGY:="context"}  # context, history, or mixed
: ${STEADYTEXT_SUGGEST_ASYNC:=1}  # Use async suggestions
: ${STEADYTEXT_SUGGEST_CACHE_SIZE:=100}  # Number of suggestions to cache

# Initialize suggestion cache
typeset -gA _steadytext_suggestion_cache

# Check if zsh-autosuggestions is loaded
if [[ -n "$ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE" ]]; then
    _STEADYTEXT_ZSH_AUTOSUGGEST_INTEGRATED=1
else
    _STEADYTEXT_ZSH_AUTOSUGGEST_INTEGRATED=0
fi

# Context gathering for suggestions
_steadytext_get_context_minimal() {
    local context=""
    
    # Minimal context for speed
    context+="pwd: $(pwd)\n"
    context+="cmd: ${BUFFER}\n"
    
    # Last command if available
    local last_cmd=$(fc -ln -1 2>/dev/null | sed 's/^[ \t]*//')
    [[ -n "$last_cmd" ]] && context+="last: ${last_cmd}\n"
    
    # Git branch if in repo
    if [[ -d .git ]]; then
        local branch=$(git branch --show-current 2>/dev/null)
        [[ -n "$branch" ]] && context+="git: ${branch}\n"
    fi
    
    echo "$context"
}

# Generate suggestion using SteadyText
_steadytext_suggestion_strategy_context() {
    local prefix="$1"
    
    # Check cache first
    local cache_key="${PWD}:${prefix}"
    if [[ -n "${_steadytext_suggestion_cache[$cache_key]}" ]]; then
        echo "${_steadytext_suggestion_cache[$cache_key]}"
        return
    fi
    
    # Generate context
    local context=$(_steadytext_get_context_minimal)
    local prompt="Complete this shell command based on context. Output only the completion, nothing else:\n${context}"
    
    if [[ $STEADYTEXT_SUGGEST_ASYNC -eq 1 ]]; then
        # Async suggestion (non-blocking)
        local tempfile=$(mktemp)
        {
            local suggestion=$(echo "$prompt" | st --quiet 2>/dev/null | head -1)
            suggestion="${suggestion#$prefix}"  # Remove prefix
            echo "$suggestion" > "$tempfile"
            
            # Update cache
            _steadytext_suggestion_cache[$cache_key]="$suggestion"
            
            # Trigger redraw if still on same buffer
            if [[ "$BUFFER" == "$prefix" ]]; then
                kill -USR1 $$
            fi
        } &!
        
        # Return empty for now (will update async)
        return
    else
        # Sync suggestion (blocking)
        local suggestion=$(echo "$prompt" | st --quiet 2>/dev/null | head -1)
        suggestion="${suggestion#$prefix}"  # Remove prefix
        
        # Cache the suggestion
        _steadytext_suggestion_cache[$cache_key]="$suggestion"
        
        # Limit cache size
        if [[ ${#_steadytext_suggestion_cache} -gt $STEADYTEXT_SUGGEST_CACHE_SIZE ]]; then
            # Remove oldest entries (simplified)
            local keys=(${(k)_steadytext_suggestion_cache})
            unset "_steadytext_suggestion_cache[${keys[1]}]"
        fi
        
        echo "$suggestion"
    fi
}

# Strategy that mixes history and AI suggestions
_steadytext_suggestion_strategy_mixed() {
    local prefix="$1"
    
    # First try history
    local hist_suggestion=$(fc -ln 1 | grep "^${prefix}" | tail -1)
    if [[ -n "$hist_suggestion" ]] && [[ "$hist_suggestion" != "$prefix" ]]; then
        echo "${hist_suggestion#$prefix}"
        return
    fi
    
    # Fall back to AI context suggestion
    _steadytext_suggestion_strategy_context "$prefix"
}

# Main suggestion function
_steadytext_suggestion() {
    local prefix="$1"
    
    case "$STEADYTEXT_SUGGEST_STRATEGY" in
        context)
            _steadytext_suggestion_strategy_context "$prefix"
            ;;
        history)
            # Pure history-based (faster)
            fc -ln 1 | grep "^${prefix}" | tail -1 | sed "s/^${prefix}//"
            ;;
        mixed)
            _steadytext_suggestion_strategy_mixed "$prefix"
            ;;
        *)
            _steadytext_suggestion_strategy_context "$prefix"
            ;;
    esac
}

# Integration with zsh-autosuggestions
if [[ $_STEADYTEXT_ZSH_AUTOSUGGEST_INTEGRATED -eq 1 ]]; then
    # Add our strategy to zsh-autosuggestions
    ZSH_AUTOSUGGEST_STRATEGY=(steadytext $ZSH_AUTOSUGGEST_STRATEGY)
    
    # Define the strategy function for zsh-autosuggestions
    _zsh_autosuggest_strategy_steadytext() {
        typeset -g suggestion
        suggestion=$(_steadytext_suggestion "$1")
    }
else
    # Standalone implementation
    
    # Variables
    typeset -g _STEADYTEXT_SUGGESTION
    typeset -g _STEADYTEXT_LAST_BUFFER
    
    # Show suggestion below the prompt
    _steadytext_show_suggestion() {
        if [[ -z "$BUFFER" ]] || [[ "$BUFFER" == "$_STEADYTEXT_LAST_BUFFER" ]]; then
            return
        fi
        
        _STEADYTEXT_LAST_BUFFER="$BUFFER"
        _STEADYTEXT_SUGGESTION=$(_steadytext_suggestion "$BUFFER")
        
        if [[ -n "$_STEADYTEXT_SUGGESTION" ]]; then
            region_highlight+=("${#BUFFER} $((${#BUFFER} + ${#_STEADYTEXT_SUGGESTION})) $STEADYTEXT_SUGGEST_HIGHLIGHT_STYLE")
        fi
    }
    
    # Accept suggestion widget
    _steadytext_accept_suggestion() {
        if [[ -n "$_STEADYTEXT_SUGGESTION" ]]; then
            LBUFFER+="$_STEADYTEXT_SUGGESTION"
            _STEADYTEXT_SUGGESTION=""
            _STEADYTEXT_LAST_BUFFER=""
        else
            # Default behavior (e.g., complete-word)
            zle complete-word
        fi
    }
    
    # Clear suggestion
    _steadytext_clear_suggestion() {
        _STEADYTEXT_SUGGESTION=""
        region_highlight=()
    }
    
    # Modify buffer display to include suggestion
    _steadytext_modify_buffer_zle_line_pre_redraw() {
        if [[ -n "$_STEADYTEXT_SUGGESTION" ]]; then
            POSTDISPLAY="$_STEADYTEXT_SUGGESTION"
        else
            unset POSTDISPLAY
        fi
    }
    
    # Set up widgets
    zle -N steadytext-accept-suggestion _steadytext_accept_suggestion
    zle -N steadytext-clear-suggestion _steadytext_clear_suggestion
    
    # Bind keys
    bindkey '^[[C' steadytext-accept-suggestion  # Right arrow
    bindkey '^I' steadytext-accept-suggestion    # Tab
    
    # Hooks
    add-zsh-hook preexec _steadytext_clear_suggestion
    
    # Set up zle hooks if available (zsh 5.3+)
    if [[ ${+functions[add-zle-hook-widget]} -eq 1 ]]; then
        add-zle-hook-widget line-pre-redraw _steadytext_modify_buffer_zle_line_pre_redraw
        add-zle-hook-widget line-init _steadytext_show_suggestion
        add-zle-hook-widget keymap-select _steadytext_show_suggestion
    fi
fi

# Utility functions
steadytext-suggest-toggle() {
    if [[ "$STEADYTEXT_SUGGEST_STRATEGY" == "off" ]]; then
        STEADYTEXT_SUGGEST_STRATEGY="context"
        echo "SteadyText suggestions enabled (strategy: context)"
    else
        STEADYTEXT_SUGGEST_STRATEGY="off"
        echo "SteadyText suggestions disabled"
    fi
}

steadytext-suggest-clear-cache() {
    _steadytext_suggestion_cache=()
    echo "SteadyText suggestion cache cleared"
}

# AIDEV-NOTE: Advanced features that could be added:
# - Multi-line command completion
# - Suggestion explanations (show why a command was suggested)
# - Learning from accepted/rejected suggestions
# - Integration with shell aliases and functions
# - Project-specific suggestion contexts