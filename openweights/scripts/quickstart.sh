#!/bin/bash
# Quick setup script for openweights

set -e

echo "🚀 OpenWeights Quick Setup"
echo "=========================="
echo ""

# Check if openweights is installed
if ! python -c "import openweights" 2>/dev/null; then
    echo "📦 Installing openweights..."
    pip install openweights
else
    echo "✅ openweights already installed"
fi

# Check for API key
if [ -z "$OPENWEIGHTS_API_KEY" ]; then
    echo ""
    echo "⚠️  OPENWEIGHTS_API_KEY not set"
    echo ""
    echo "To get an API key, run:"
    echo "  ow signup"
    echo ""
    echo "Then set it with:"
    echo "  export OPENWEIGHTS_API_KEY=your_key_here"
    echo ""
    echo "Or add to your shell profile (~/.bashrc or ~/.zshrc):"
    echo "  echo 'export OPENWEIGHTS_API_KEY=your_key_here' >> ~/.bashrc"
    echo ""
else
    echo "✅ OPENWEIGHTS_API_KEY is set"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Validate setup: python scripts/validate_setup.py"
echo "  2. Try an example: python assets/examples/sft/lora_qwen3_4b.py"
