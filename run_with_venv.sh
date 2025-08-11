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
