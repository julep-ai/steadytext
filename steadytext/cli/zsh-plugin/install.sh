#!/usr/bin/env bash
# SteadyText ZSH Plugin Installer
# AIDEV-NOTE: This script helps users install the SteadyText ZSH plugins

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZSH_CUSTOM="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"

echo "SteadyText ZSH Plugin Installer"
echo "==============================="
echo

# Check if zsh is installed
if ! command -v zsh &> /dev/null; then
    echo "Error: ZSH is not installed. Please install ZSH first."
    exit 1
fi

# Check if SteadyText is installed
if ! command -v st &> /dev/null; then
    echo "Error: SteadyText CLI (st) is not found in PATH."
    echo "Please install SteadyText first: pip install steadytext"
    exit 1
fi

# Detect plugin framework
if [[ -d "$HOME/.oh-my-zsh" ]]; then
    echo "Detected: Oh My Zsh"
    FRAMEWORK="oh-my-zsh"
elif [[ -d "$HOME/.config/zsh/plugins" ]]; then
    echo "Detected: Custom ZSH plugin directory"
    FRAMEWORK="custom"
    ZSH_CUSTOM="$HOME/.config/zsh/plugins"
elif [[ -f "$HOME/.zshrc" ]]; then
    echo "Detected: Standalone ZSH configuration"
    FRAMEWORK="standalone"
else
    echo "No ZSH configuration found. Creating ~/.zshrc"
    touch "$HOME/.zshrc"
    FRAMEWORK="standalone"
fi

echo
echo "Choose plugins to install:"
echo "1. Basic completion only"
echo "2. Context-aware suggestions (Ctrl-X Ctrl-S)"
echo "3. Autosuggestions (as you type)"
echo "4. All plugins"
echo

read -p "Enter choice (1-4): " choice

install_completion() {
    echo "Installing basic completions..."
    st completion --install
}

install_context_plugin() {
    echo "Installing context-aware suggestions plugin..."
    
    if [[ "$FRAMEWORK" == "oh-my-zsh" ]]; then
        # Install as oh-my-zsh plugin
        local plugin_dir="$ZSH_CUSTOM/plugins/steadytext-context"
        mkdir -p "$plugin_dir"
        cp "$SCRIPT_DIR/steadytext-context.plugin.zsh" "$plugin_dir/"
        cp "$SCRIPT_DIR/_steadytext_async" "$plugin_dir/"
        
        echo
        echo "Add 'steadytext-context' to your plugins list in ~/.zshrc:"
        echo "plugins=(... steadytext-context)"
    else
        # Standalone installation
        echo "" >> "$HOME/.zshrc"
        echo "# SteadyText context-aware suggestions" >> "$HOME/.zshrc"
        echo "source $SCRIPT_DIR/steadytext-context.plugin.zsh" >> "$HOME/.zshrc"
    fi
}

install_autosuggestions() {
    echo "Installing autosuggestions plugin..."
    
    # Check if zsh-autosuggestions is installed
    local has_autosuggest=0
    if [[ -f "$ZSH_CUSTOM/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh" ]] || \
       [[ -f "/usr/share/zsh-autosuggestions/zsh-autosuggestions.zsh" ]]; then
        has_autosuggest=1
        echo "Found existing zsh-autosuggestions - will integrate"
    fi
    
    if [[ "$FRAMEWORK" == "oh-my-zsh" ]]; then
        # Install as oh-my-zsh plugin
        local plugin_dir="$ZSH_CUSTOM/plugins/steadytext-autosuggestions"
        mkdir -p "$plugin_dir"
        cp "$SCRIPT_DIR/steadytext-autosuggestions.zsh" "$plugin_dir/steadytext-autosuggestions.plugin.zsh"
        
        echo
        echo "Add 'steadytext-autosuggestions' to your plugins list in ~/.zshrc:"
        echo "plugins=(... steadytext-autosuggestions)"
        
        if [[ $has_autosuggest -eq 0 ]]; then
            echo
            echo "Note: For best results, also install zsh-autosuggestions:"
            echo "git clone https://github.com/zsh-users/zsh-autosuggestions \$ZSH_CUSTOM/plugins/zsh-autosuggestions"
        fi
    else
        # Standalone installation
        echo "" >> "$HOME/.zshrc"
        echo "# SteadyText autosuggestions" >> "$HOME/.zshrc"
        echo "source $SCRIPT_DIR/steadytext-autosuggestions.zsh" >> "$HOME/.zshrc"
    fi
}

# Install based on choice
case $choice in
    1)
        install_completion
        ;;
    2)
        install_completion
        install_context_plugin
        ;;
    3)
        install_completion
        install_autosuggestions
        ;;
    4)
        install_completion
        install_context_plugin
        install_autosuggestions
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo
echo "Installation complete!"
echo
echo "Configuration options can be set in your ~/.zshrc:"
echo "  export STEADYTEXT_SUGGEST_ENABLED=1"
echo "  export STEADYTEXT_SUGGEST_MODEL_SIZE=small"
echo "  export STEADYTEXT_SUGGEST_STRATEGY=context"
echo
echo "Restart your shell or run: source ~/.zshrc"
echo
echo "For more information, see: $SCRIPT_DIR/README.md"