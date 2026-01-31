#!/bin/bash
# Quick verification that the repo is production-ready

echo "🔍 Hyper-Mnemosyne Production Readiness Check"
echo "=============================================="
echo ""

# Check for required files
echo "📋 Checking required files..."
REQUIRED_FILES=(
    "README.md"
    "LICENSE"
    "setup.py"
    "requirements.txt"
    ".gitignore"
    "CONTRIBUTING.md"
    "CHANGELOG.md"
    "config.py"
    "start_training.sh"
    "test_training.py"
    "training/train.py"
    "scripts/prepare_data.py"
    "model/backbone.py"
)

ALL_PRESENT=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (MISSING)"
        ALL_PRESENT=false
    fi
done

echo ""

# Check for artifacts that should NOT be present
echo "🧹 Checking for unwanted artifacts..."
UNWANTED=(
    "*.pt"
    "*.pth"
    "*.log"
    "data/"
    "__pycache__"
)

CLEAN=true
for pattern in "${UNWANTED[@]}"; do
    if ls $pattern 2>/dev/null | grep -q .; then
        echo "  ⚠️  Found: $pattern (should be gitignored)"
        CLEAN=false
    else
        echo "  ✅ Clean: $pattern"
    fi
done

echo ""

# Check package installation
echo "📦 Checking package installation..."
if pip show hyper-mnemosyne &>/dev/null; then
    echo "  ✅ Package installed"
else
    echo "  ⚠️  Package not installed (run: pip install -e .)"
fi

echo ""

# Check imports work
echo "🐍 Checking Python imports..."
python3 -c "
import sys
try:
    from config import HyperMnemosyneConfig
    from model.backbone import HyperMnemosyne
    from training.data_utils import create_dataloader
    print('  ✅ All imports work')
except ImportError as e:
    print(f'  ❌ Import failed: {e}')
    sys.exit(1)
"

echo ""

# Final verdict
echo "=============================================="
if $ALL_PRESENT && $CLEAN; then
    echo "✅ Repository is PRODUCTION READY!"
    echo ""
    echo "Next steps:"
    echo "  1. Review and commit all changes"
    echo "  2. Push to GitHub"
    echo "  3. Tag release: git tag v0.1.0"
else
    echo "⚠️  Some issues found. Please review above."
fi

echo ""
echo "Repository size: $(du -sh . | cut -f1)"
