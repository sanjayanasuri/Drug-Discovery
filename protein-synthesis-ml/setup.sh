#!/bin/bash
# Setup script for protein-synthesis-ml project
# Run this once to ensure all dependencies are installed

echo "=== Setting up protein-synthesis-ml environment ==="

# Activate conda base environment (if not already active)
# If you're using a different conda env, replace 'base' with your env name
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base

# Verify we're using conda Python
PYTHON_PATH=$(which python)
if [[ ! "$PYTHON_PATH" == *"anaconda"* ]] && [[ ! "$PYTHON_PATH" == *"conda"* ]]; then
    echo "WARNING: Not using conda Python. Current Python: $PYTHON_PATH"
    echo "Please activate conda environment first: conda activate base"
    exit 1
fi

echo "Using Python: $PYTHON_PATH"
echo "Python version: $(python --version)"

# Install/update packages using conda (preferred for conda environments)
echo ""
echo "=== Installing packages via conda ==="
conda install -y -c conda-forge \
    pandas>=2.1.0 \
    scikit-learn>=1.3.0 \
    numpy>=1.25.2,<2.0.0 \
    xgboost>=2.0.0 \
    matplotlib>=3.7.0 \
    || echo "Some packages may need pip installation"

# Install packages that might not be available via conda
echo ""
echo "=== Installing packages via pip ==="
python -m pip install --upgrade \
    rdkit-pypi>=2022.9.0 \
    chembl-webresource-client>=0.10.0

# Verify all imports work
echo ""
echo "=== Verifying installations ==="
python -c "
import sys
packages = ['pandas', 'numpy', 'sklearn', 'xgboost', 'matplotlib', 'rdkit', 'chembl_webresource_client']
failed = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError as e:
        print(f'✗ {pkg}: {e}')
        failed.append(pkg)

if failed:
    print(f'\nERROR: Failed to import: {failed}')
    sys.exit(1)
else:
    print('\n✓ All packages imported successfully!')
"

echo ""
echo "=== Setup complete! ==="
echo "To run scripts, use: python train_classification.py"
echo "Make sure you're using conda Python: $(which python)"

