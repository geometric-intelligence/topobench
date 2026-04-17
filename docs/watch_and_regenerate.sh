#!/bin/bash
# Watch for changes in the topobench package and automatically regenerate docs

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================"
echo "Documentation Auto-Regeneration Watcher"
echo -e "======================================${NC}"
echo ""
echo "This script will watch for changes in the topobench/ directory"
echo "and automatically regenerate the documentation."
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop watching${NC}"
echo ""

# Get directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check if inotify-tools is installed (for Linux)
if command -v inotifywait &> /dev/null; then
    WATCHER="inotifywait"
elif command -v fswatch &> /dev/null; then
    WATCHER="fswatch"
else
    echo -e "${YELLOW}Warning: Neither inotifywait nor fswatch found.${NC}"
    echo "Please install one of them:"
    echo "  - Linux: sudo apt-get install inotify-tools"
    echo "  - macOS: brew install fswatch"
    echo ""
    echo "Falling back to periodic check (every 30 seconds)..."
    WATCHER="poll"
fi

# Function to regenerate docs
regenerate_docs() {
    echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} Changes detected, regenerating docs..."
    cd "$SCRIPT_DIR"
    bash generate_api_docs.sh > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} Documentation regenerated successfully!"
    else
        echo -e "${YELLOW}[$(date '+%H:%M:%S')]${NC} Warning: Documentation generation had errors"
    fi
}

# Generate docs initially
echo -e "${GREEN}Generating initial documentation...${NC}"
regenerate_docs
echo ""

# Watch for changes
if [ "$WATCHER" = "inotifywait" ]; then
    echo -e "${BLUE}Using inotifywait for file watching...${NC}"
    echo ""
    while inotifywait -r -e modify,create,delete "$PROJECT_DIR/topobench" --exclude '__pycache__|\.pyc$' -q; do
        regenerate_docs
    done
elif [ "$WATCHER" = "fswatch" ]; then
    echo -e "${BLUE}Using fswatch for file watching...${NC}"
    echo ""
    fswatch -o "$PROJECT_DIR/topobench" --exclude='__pycache__|\.pyc$' | while read; do
        regenerate_docs
    done
else
    # Fallback: periodic polling
    echo -e "${BLUE}Using periodic polling (checking every 30 seconds)...${NC}"
    echo ""
    LAST_HASH=""
    while true; do
        # Calculate hash of all Python files
        CURRENT_HASH=$(find "$PROJECT_DIR/topobench" -name "*.py" -type f -exec md5sum {} \; 2>/dev/null | sort | md5sum)

        if [ "$LAST_HASH" != "$CURRENT_HASH" ] && [ -n "$LAST_HASH" ]; then
            regenerate_docs
        fi

        LAST_HASH=$CURRENT_HASH
        sleep 30
    done
fi
