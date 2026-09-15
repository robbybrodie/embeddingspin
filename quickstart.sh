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
echo "  1. Ingest 11 IBM financial documents (2015-2024)"
echo "  2. Encode each period as arcs on three circles (1y / 16y / 256y)"
echo "  3. Split a multi-year document at its year boundaries"
echo "  4. Show two-pass retrieval: hard overlap gate, then β-weighted ranking"
echo ""
read -p "Press Enter to continue..."
echo ""

export USE_OPENAI_EMBEDDINGS=false
export USE_MOCK_EMBEDDINGS=true
python demo.py

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "  DEMO COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "  1. Run a custom query (β is in [0, 1] now, not the thousands):"
echo "     python demo.py --query \"IBM cloud\" --start 2019-01-01 --end 2020-01-01 --beta 0.7"
echo ""
echo "  2. Show the β sweep:"
echo "     python demo.py --beta-sweep"
echo ""
echo "  3. Decompose a multi-period question:"
echo "     python demo.py --decompose \"Q1 impact on the full year for 2021, 2022 and 2023\""
echo ""
echo "  4. See the geometry on its own:"
echo "     python arc_demo.py"
echo ""
echo "  5. Run the tests:"
echo "     pytest"
echo ""
echo "  6. Start the API server:"
echo "     python api.py"
echo "     # Visit http://localhost:8080/docs"
echo ""
echo "  7. Read the documentation:"
echo "     cat README.md"
echo "     cat PATENT_ALIGNMENT.md"
echo "     cat CHANGELOG.md"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"

