#!/bin/bash
# AIDEV-NOTE: File watcher for auto-rebuilding pg_steadytext extension on changes
# Monitors SQL and Python files, triggers rebuild when changes detected

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WATCH_DIR="../pg_steadytext"
REBUILD_SCRIPT="$(dirname "$0")/rebuild_extension.sh"
DEBOUNCE_SECONDS=2

echo -e "${BLUE}👁️  Starting file watcher for pg_steadytext extension${NC}"
echo -e "${YELLOW}Watching: $WATCH_DIR${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Check if inotifywait is available
if ! command -v inotifywait &> /dev/null; then
    echo -e "${YELLOW}Installing inotify-tools...${NC}"
    sudo apt-get update && sudo apt-get install -y inotify-tools || {
        echo -e "${RED}Failed to install inotify-tools${NC}"
        echo "Falling back to polling mode..."
        USE_POLLING=true
    }
fi

# Function to rebuild
rebuild() {
    echo -e "${GREEN}📝 Changes detected, rebuilding...${NC}"
    if bash "$REBUILD_SCRIPT"; then
        echo -e "${GREEN}✅ Rebuild successful${NC}"
    else
        echo -e "${RED}❌ Rebuild failed${NC}"
    fi
    echo -e "${BLUE}👁️  Watching for changes...${NC}"
}

# Function to check if file should trigger rebuild
should_rebuild() {
    local file="$1"
    # Check if it's a SQL or Python file
    if [[ "$file" == *.sql ]] || [[ "$file" == *.py ]]; then
        # Exclude test output files and temp files
        if [[ "$file" != *"/expected/"* ]] && [[ "$file" != *".tmp"* ]] && [[ "$file" != *"__pycache__"* ]]; then
            return 0
        fi
    fi
    return 1
}

if [ "$USE_POLLING" = "true" ]; then
    # Fallback: polling mode
    echo -e "${YELLOW}Using polling mode (less efficient)${NC}"
    
    # Get initial checksums
    declare -A checksums
    while IFS= read -r -d '' file; do
        if should_rebuild "$file"; then
            checksums["$file"]=$(md5sum "$file" 2>/dev/null | cut -d' ' -f1)
        fi
    done < <(find "$WATCH_DIR" -type f \( -name "*.sql" -o -name "*.py" \) -print0)
    
    while true; do
        sleep 2
        changed=false
        
        while IFS= read -r -d '' file; do
            if should_rebuild "$file"; then
                current_checksum=$(md5sum "$file" 2>/dev/null | cut -d' ' -f1)
                if [ "${checksums["$file"]}" != "$current_checksum" ]; then
                    checksums["$file"]=$current_checksum
                    changed=true
                    echo -e "${YELLOW}Changed: ${file#$WATCH_DIR/}${NC}"
                fi
            fi
        done < <(find "$WATCH_DIR" -type f \( -name "*.sql" -o -name "*.py" \) -print0)
        
        if [ "$changed" = true ]; then
            rebuild
        fi
    done
else
    # Preferred: inotify mode
    echo -e "${GREEN}Using inotify mode (efficient)${NC}"
    echo -e "${BLUE}👁️  Watching for changes...${NC}"
    
    # Track last rebuild time to debounce
    last_rebuild=0
    
    inotifywait -m -r -e modify,create,delete,move \
        --exclude '(\.git|__pycache__|\.pyc|\.pyo|\.tmp|expected/)' \
        "$WATCH_DIR" |
    while read -r directory event filename; do
        current_time=$(date +%s)
        
        # Check if this file should trigger a rebuild
        filepath="${directory}${filename}"
        if should_rebuild "$filepath"; then
            # Debounce: only rebuild if enough time has passed
            if (( current_time - last_rebuild >= DEBOUNCE_SECONDS )); then
                echo -e "${YELLOW}Changed: ${filepath#$WATCH_DIR/}${NC}"
                last_rebuild=$current_time
                rebuild
            fi
        fi
    done
fi