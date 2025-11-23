# Drug Discovery Screening Pipeline

A comprehensive Streamlit application for HIV-1 protease drug discovery, featuring activity screening, ADMET evaluation, lead optimization, and ranking.

## ⚠️ Current Limitations & Roadmap

This system has been improved for production use, but several advanced features are still in development:

- **❌ Protein-aware ML**: Currently requires one model per target. Future: Universal binding model using protein embeddings (ESM2/ProtBERT).
- **❌ Advanced Docking**: Currently uses AutoDock Vina. Future: GNINA/DiffDock integration for better accuracy.
- **⚠️ ADMET Uncertainty**: Calibration added, but uncertainty estimates still needed.
- **⚠️ Synthetic Accessibility**: Medchem filters added, but SA Score and retrosynthesis checks still needed.
- **❌ GPU Scaling**: Currently CPU-based. Future: GPU batching, Faiss for similarity search.

See [PITFALLS_ANALYSIS.md](PITFALLS_ANALYSIS.md) for detailed roadmap.

## Features

- **Activity Screening**: Score molecules for HIV-1 protease activity using trained ML models
- **ADMET Evaluation**: Comprehensive ADMET property prediction (toxicity, absorption, distribution, metabolism, excretion)
- **Lead Optimization**: Generate new molecules by mutating top leads using RDKit transformations
- **Lead Ranking**: Filter, sort, and analyze top candidates with detailed reports
- **Clustering & Visualization**: Molecular similarity clustering with UMAP visualization

## Troubleshooting

If you encounter `ModuleNotFoundError: No module named 'rdkit'`:

1. **Clear Streamlit cache**: `rm -rf ~/.streamlit/cache`
2. **Verify environment**: Run `python check_environment.py`
3. **Restart Streamlit**: Stop and restart `streamlit run app.py`

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

## Quick Start

### Option 1: Using Conda (Recommended)

```bash
# Create conda environment
make setup
# or manually:
conda env create -f environment.yml
conda activate hiv-ml

# Train models
make train          # Train all models (HIV-1 + ADMET)
# or separately:
make train-hiv      # Train HIV-1 protease model
make train-admet    # Train all ADMET models

# Run app
make app
# or manually (recommended - ensures correct Python):
python -m streamlit run app.py
# or use the helper script:
./run_app.sh
```

### Option 2: Using Docker

```bash
# Build image
docker build -t hiv-ml-pipeline .

# Run container
docker run -p 8501:8501 hiv-ml-pipeline

# Access app at http://localhost:8501
```

### Option 3: Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python main.py              # Train HIV-1 model
python train_all_admet.py   # Train ADMET models

# Run app
streamlit run app.py
```

## Workflow

1. **Upload Library** → Upload a CSV file with SMILES strings
2. **Activity Screening** → Score molecules for HIV-1 protease activity
3. **ADMET Evaluation** → Evaluate ADMET properties and compute composite scores
4. **Lead Optimization** → Generate new molecules by mutating top leads
5. **Lead Ranking** → Filter, sort, and analyze top candidates

## Requirements

See `requirements.txt` or `environment.yml` for full dependency list.

Key dependencies:
- Python 3.10+
- RDKit (for molecular operations)
- scikit-learn (for ML models)
- Streamlit (for UI)
- pandas, numpy, matplotlib
- XGBoost (for models)
- UMAP (for visualization)

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── pipeline.py             # Core pipeline and model registry
├── featurization.py        # SMILES to fingerprint conversion
├── admet_loader.py         # ADMET data loading
├── lead_optimization.py    # Lead optimization module
├── clustering.py           # Molecular clustering utilities
├── main.py                 # Train HIV-1 protease model
├── train_all_admet.py      # Train all ADMET models
├── models/                 # Trained model files (generated)
│   ├── hiv_protease_rf.pkl
│   └── admet/
├── requirements.txt        # pip dependencies
├── environment.yml        # conda environment
├── Makefile               # Common commands
└── Dockerfile             # Docker container setup
```

## Makefile Commands

```bash
make help          # Show all available commands
make setup         # Create conda environment
make train         # Train all models
make train-hiv     # Train HIV-1 model only
make train-admet   # Train ADMET models
make app           # Run Streamlit app
make clean         # Clean Python cache files
```

## Notes

- Models must be trained before using the app (see training commands above)
- First run will download ADMET datasets from TDC (Therapeutics Data Commons)
- Training ADMET models takes ~20-40 minutes depending on hardware
- The app caches models and featurizations for faster subsequent runs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
