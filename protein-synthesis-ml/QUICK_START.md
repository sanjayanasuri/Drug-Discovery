# Quick Start Guide - Exact Commands

## One-Time Setup (Run Once)

### Option 1: Automated Setup (Recommended)
```bash
cd /Users/sanjayanasuri/protein-synthesis-ml
./setup.sh
```

### Option 2: Manual Setup
```bash
# 1. Activate conda environment
conda activate base

# 2. Verify you're using conda Python (should show /opt/anaconda3/bin/python)
which python

# 3. Install packages via conda
conda install -y -c conda-forge pandas scikit-learn "numpy>=1.25.2,<2.0.0" xgboost matplotlib

# 4. Install packages via pip (for packages not in conda)
python -m pip install rdkit-pypi chembl-webresource-client

# 5. Verify everything works
python -c "import pandas, numpy, sklearn, xgboost, matplotlib, rdkit, chembl_webresource_client; print('All packages imported successfully!')"
```

## Daily Usage (Every Time You Work on This Project)

### Step 1: Activate Conda Environment
```bash
conda activate base
```

### Step 2: Navigate to Project Directory
```bash
cd /Users/sanjayanasuri/protein-synthesis-ml
```

### Step 3: Verify You're Using the Right Python
```bash
# This should show: /opt/anaconda3/bin/python
which python

# If it shows /usr/local/bin/python3, you're using the wrong Python!
# Fix it by running: conda activate base
```

### Step 4: Run Your Scripts
```bash
# Download data from ChEMBL
python get_protein.py

# Train classification models
python train_classification.py

# Train regression models
python train_regression.py

# Run main pipeline
python main.py

# Run Streamlit app (IMPORTANT: use python -m streamlit, not just streamlit)
python -m streamlit run app.py
# OR use the helper script:
./run_app.sh
```

## Important Notes

1. **Always use `python` (conda Python), NOT `python3` (system Python)**
   - ✅ Correct: `python train_classification.py`
   - ❌ Wrong: `python3 train_classification.py` or `/usr/local/bin/python3 train_classification.py`

2. **If you get import errors:**
   ```bash
   # Check which Python you're using
   which python
   
   # If it's not conda Python, activate conda:
   conda activate base
   
   # Verify packages are installed
   python -c "import sklearn; print('sklearn OK')"
   ```

3. **To check if packages are installed:**
   ```bash
   python -m pip list | grep -E "pandas|numpy|scikit-learn|xgboost|matplotlib|rdkit|chembl"
   ```

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'sklearn'"
**Solution:** You're using the wrong Python. Run:
```bash
conda activate base
which python  # Should show /opt/anaconda3/bin/python
```

### Problem: "externally-managed-environment" error
**Solution:** You're trying to use system pip. Use conda Python's pip:
```bash
python -m pip install <package>
# NOT: pip install <package>
```

### Problem: NumPy version conflicts
**Solution:** Ensure NumPy >= 1.25.2:
```bash
python -m pip install "numpy>=1.25.2,<2.0.0" --upgrade
```

### Problem: "ModuleNotFoundError: No module named 'rdkit'" when running Streamlit
**Solution:** You're using the wrong Python. The `streamlit` command uses system Python. Use:
```bash
# Use python -m streamlit instead of just streamlit
python -m streamlit run app.py

# OR use the helper script
./run_app.sh
```

