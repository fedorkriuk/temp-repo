#!/bin/bash

echo "🔧 SHEPHERD-GRID ENVIRONMENT SETUP"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✅ Virtual environment already exists"
else
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

echo ""
echo "📝 To activate the virtual environment, run:"
echo "   source venv/bin/activate"
echo ""
echo "Then install requirements with:"
echo "   pip install -r requirements.txt"
echo ""
echo "To run experiments after activation:"
echo "   python run_experiment.py"
echo ""

# Create a convenience script
cat > run_with_venv.sh << 'EOF'
#!/bin/bash
# Convenience script to run experiments with virtual environment

# Activate virtual environment
source venv/bin/activate

# Check if requirements are installed
if ! python -c "import flask" 2>/dev/null || ! python -c "import numpy" 2>/dev/null; then
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
fi

# Run the experiment runner
python run_experiment.py

# Deactivate when done
deactivate
EOF

chmod +x run_with_venv.sh

echo "💡 TIP: Created run_with_venv.sh for easy execution"
echo "   Just run: ./run_with_venv.sh"