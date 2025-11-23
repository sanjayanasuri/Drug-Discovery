#!/bin/bash
# Helper script to run Streamlit app with the correct Python environment

# Activate conda environment if available
if command -v conda &> /dev/null; then
    # Try to activate chem-ml environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate chem-ml 2>/dev/null || echo "Warning: chem-ml environment not found, using current Python"
fi

# Use python -m streamlit to ensure correct Python interpreter
python -m streamlit run app.py "$@"
