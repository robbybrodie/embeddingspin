#!/bin/bash
# Temporal-Phase Spin Retrieval - Quick Start Script

set -e

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  TEMPORAL-PHASE SPIN RETRIEVAL SYSTEM - QUICK START"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "✓ Found Python $PYTHON_VERSION"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Run demo
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  RUNNING DEMO"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "This demo will:"
echo "  1. Ingest 10 IBM financial reports (2015-2024)"
echo "  2. Encode timestamps as 2D spin vectors on unit circle"
echo "  3. Demonstrate temporal zoom with β parameter"
echo "  4. Show multi-pass retrieval (coarse recall + re-ranking)"
echo ""
read -p "Press Enter to continue..."
echo ""

python demo.py

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  DEMO COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "  1. Run custom query:"
echo "     python demo.py --query \"IBM cloud\" --timestamp 2019-06-30 --beta 10.0"
echo ""
echo "  2. Show β parameter sweep:"
echo "     python demo.py --beta-sweep"
echo ""
echo "  3. Start API server:"
echo "     python api.py"
echo "     # Visit http://localhost:8080/docs"
echo ""
echo "  4. Read documentation:"
echo "     cat README.md"
echo "     cat USAGE_EXAMPLES.md"
echo "     cat DEPLOYMENT.md"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"

