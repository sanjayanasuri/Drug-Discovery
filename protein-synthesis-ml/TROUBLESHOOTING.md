# Troubleshooting Guide

## ModuleNotFoundError: No module named 'rdkit'

If you see this error when running the Streamlit app, try these solutions:

### Solution 1: Clear Streamlit Cache

Streamlit caches imports. Clear the cache:

```bash
# Option A: Clear cache via command
rm -rf ~/.streamlit/cache

# Option B: Clear cache directory manually
# On macOS/Linux: ~/.streamlit/cache
# On Windows: %USERPROFILE%\.streamlit\cache
```

Then restart Streamlit:
```bash
streamlit run app.py
```

### Solution 2: Verify Environment

Check that you're using the correct Python environment:

```bash
# Run the environment check script
python check_environment.py

# Verify RDKit is installed
python -c "from rdkit import Chem; print('RDKit OK')"
```

### Solution 3: Reinstall Dependencies

If RDKit is missing, install it:

```bash
# Using pip
pip install rdkit-pypi

# Or install all requirements
pip install -r requirements.txt

# Using conda (recommended for RDKit)
conda install -c conda-forge rdkit
```

### Solution 4: Use Conda Environment

If you have a conda environment set up:

```bash
# Activate the environment
conda activate hiv-ml  # or your environment name

# Verify RDKit
python -c "from rdkit import Chem; print('RDKit OK')"

# Run Streamlit
streamlit run app.py
```

### Solution 5: Use `python -m streamlit` Instead

If `streamlit run app.py` uses the wrong Python, use:

```bash
# This ensures Streamlit uses the Python from your current environment
python -m streamlit run app.py

# Or use the helper script
./run_app.sh
```

This is more reliable than calling `streamlit` directly because it explicitly uses the Python interpreter from your current environment.

### Solution 6: Restart Everything

Sometimes a full restart helps:

1. Stop Streamlit (Ctrl+C)
2. Clear cache: `rm -rf ~/.streamlit/cache`
3. Restart terminal/shell
4. Activate environment (if using conda)
5. Run: `streamlit run app.py`

## Other Common Issues

### Import errors after code changes

Clear Streamlit cache and restart:
```bash
rm -rf ~/.streamlit/cache
streamlit run app.py
```

### Models not found

Make sure you've trained the models first:
```bash
# Train HIV-1 model (or use target selector in UI)
python main.py

# Train ADMET models
python train_all_admet.py
```

### ChEMBL API errors

If downloading ChEMBL data fails:
- Check internet connection
- Verify ChEMBL ID is correct
- Try again (API may be temporarily unavailable)

## Getting Help

If issues persist:
1. Run `python check_environment.py` and share output
2. Check Python/Streamlit versions match requirements
3. Verify all dependencies from `requirements.txt` are installed

