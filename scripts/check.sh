#!/usr/bin/env bash
# Pre-push check script - Run all quality checks before pushing to remote
set -e

echo "=================================================="
echo "Running pre-push checks..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED=0

# Check 1: Ruff linting
echo ""
echo "📋 Running Ruff linting..."
if uv run ruff check .; then
    echo -e "${GREEN}✓ Ruff passed${NC}"
else
    echo -e "${RED}✗ Ruff failed${NC}"
    FAILED=1
fi

# Check 2: MyPy type checking
echo ""
echo "🔍 Running MyPy type checking..."
uv run mypy src/maktaba --no-error-summary 2>&1 | tee mypy.log || true
ERROR_COUNT=$(grep -c "error:" mypy.log 2>/dev/null || echo 0)
echo "MyPy found $ERROR_COUNT errors"

if [ "$ERROR_COUNT" -gt 20 ]; then
    echo -e "${RED}✗ MyPy failed (threshold: 20 errors)${NC}"
    FAILED=1
else
    echo -e "${GREEN}✓ MyPy passed ($ERROR_COUNT errors, threshold: 20)${NC}"
fi

# Check 3: Pytest
echo ""
echo "🧪 Running tests..."
if uv run python -m pytest tests/ -v; then
    echo -e "${GREEN}✓ Tests passed${NC}"
else
    echo -e "${RED}✗ Tests failed${NC}"
    FAILED=1
fi

# Summary
echo ""
echo "=================================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Safe to push.${NC}"
    echo "=================================================="
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please fix before pushing.${NC}"
    echo "=================================================="
    exit 1
fi
