#!/bin/bash
# One-command documentation builder - generates, builds, and optionally opens docs

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}"
echo "========================================"
echo "  TopoBench Documentation Builder"
echo "========================================"
echo -e "${NC}"

# Parse arguments
OPEN_BROWSER=false
CLEAN_BUILD=false
SERVE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--open)
            OPEN_BROWSER=true
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -s|--serve)
            SERVE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -o, --open    Open documentation in browser after building"
            echo "  -c, --clean   Clean all build artifacts before building"
            echo "  -s, --serve   Start a local HTTP server after building"
            echo "  -h, --help    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0              # Just build docs"
            echo "  $0 --open       # Build and open in browser"
            echo "  $0 --clean --open  # Clean, build, and open"
            echo "  $0 --serve      # Build and start local server"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

cd "$SCRIPT_DIR"

# Step 1: Clean if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}[1/4] Cleaning previous builds...${NC}"
    make clean > /dev/null 2>&1
    make clean-api > /dev/null 2>&1
    echo -e "${GREEN}✓ Cleaned${NC}"
    echo ""
else
    echo -e "${YELLOW}[1/4] Skipping clean (use --clean to clean first)${NC}"
    echo ""
fi

# Step 2: Generate API docs
echo -e "${YELLOW}[2/4] Generating API documentation...${NC}"
bash generate_api_docs.sh > /tmp/apidoc.log 2>&1
if [ $? -eq 0 ]; then
    MODULE_COUNT=$(grep -o "with [0-9]* modules" /tmp/apidoc.log | grep -o "[0-9]*")
    echo -e "${GREEN}✓ Generated documentation for $MODULE_COUNT modules${NC}"
else
    echo -e "${RED}✗ API generation failed. Check /tmp/apidoc.log${NC}"
    exit 1
fi
echo ""

# Step 3: Build HTML documentation
echo -e "${YELLOW}[3/4] Building HTML documentation with Sphinx...${NC}"
make html > /tmp/sphinx.log 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ HTML documentation built successfully${NC}"
else
    echo -e "${RED}✗ Sphinx build failed. Check /tmp/sphinx.log${NC}"
    tail -20 /tmp/sphinx.log
    exit 1
fi
echo ""

# Step 4: Post-build actions
BUILD_DIR="$SCRIPT_DIR/_build/html"
INDEX_FILE="$BUILD_DIR/index.html"

echo -e "${YELLOW}[4/4] Post-build actions...${NC}"

if [ "$SERVE" = true ]; then
    echo -e "${GREEN}✓ Starting local documentation server...${NC}"
    echo ""
    echo -e "${BLUE}Documentation available at:${NC}"
    echo -e "  ${GREEN}http://localhost:8000${NC}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
    echo ""
    cd "$BUILD_DIR"
    python -m http.server 8000
elif [ "$OPEN_BROWSER" = true ]; then
    if [ -f "$INDEX_FILE" ]; then
        echo -e "${GREEN}✓ Opening documentation in browser...${NC}"

        # Detect OS and open browser
        if command -v xdg-open > /dev/null; then
            xdg-open "$INDEX_FILE" > /dev/null 2>&1 &
        elif command -v open > /dev/null; then
            open "$INDEX_FILE" > /dev/null 2>&1 &
        elif command -v firefox > /dev/null; then
            firefox "$INDEX_FILE" > /dev/null 2>&1 &
        elif command -v google-chrome > /dev/null; then
            google-chrome "$INDEX_FILE" > /dev/null 2>&1 &
        else
            echo -e "${YELLOW}Could not detect browser. Please open manually:${NC}"
            echo "  $INDEX_FILE"
        fi
    else
        echo -e "${RED}✗ Index file not found: $INDEX_FILE${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Build complete${NC}"
    echo ""
    echo -e "${BLUE}Documentation location:${NC}"
    echo "  $BUILD_DIR"
    echo ""
    echo -e "${BLUE}To view:${NC}"
    echo "  $0 --open"
    echo -e "${BLUE}To serve:${NC}"
    echo "  $0 --serve"
fi

echo ""
echo -e "${GREEN}========================================"
echo "  Documentation Ready! 🎉"
echo "========================================${NC}"
