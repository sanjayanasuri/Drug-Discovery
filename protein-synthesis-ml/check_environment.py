#!/usr/bin/env python
"""
Quick environment check script.
Run this to verify all dependencies are installed correctly.
"""

import sys

print("=" * 60)
print("Environment Check")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")
print()

# Check critical imports
checks = [
    ("rdkit", "from rdkit import Chem"),
    ("pandas", "import pandas"),
    ("numpy", "import numpy"),
    ("sklearn", "import sklearn"),
    ("streamlit", "import streamlit"),
    ("chembl_webresource_client", "from chembl_webresource_client.new_client import new_client"),
    ("pytdc", "import tdc"),
    ("featurization", "from featurization import smiles_to_morgan_fp"),
    ("pipeline", "from pipeline import registry"),
    ("target_selector", "from target_selector import PREDEFINED_TARGETS"),
]

print("Checking imports...")
all_ok = True
for name, import_stmt in checks:
    try:
        exec(import_stmt)
        print(f"✅ {name}")
    except ImportError as e:
        print(f"❌ {name}: {e}")
        all_ok = False
    except Exception as e:
        print(f"⚠️  {name}: {e}")

print()
if all_ok:
    print("✅ All imports successful! Environment is ready.")
else:
    print("❌ Some imports failed. Please install missing packages:")
    print("   pip install -r requirements.txt")
    print("   or")
    print("   conda env create -f environment.yml")

print("=" * 60)

