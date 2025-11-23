# Production-Ready Improvements

This document summarizes the improvements made to transform the pipeline from a single-target toy into a production-ready multi-target drug discovery platform.

## 1. Config-Driven Multi-Target Support

### Files Created/Modified:
- `config/targets.yaml` - Target configuration file
- `config/scoring_weights.yaml` - Configurable scoring weights
- `config_loader.py` - Configuration loader utilities
- `pipeline.py` - Updated to use config-driven targets

### Features:
- **Target Configuration**: All targets defined in YAML with ChEMBL IDs, model paths, and metadata
- **Dynamic Target Loading**: Registry automatically loads targets from config
- **Scoring Weights**: Composite scoring weights are configurable per disease area/use case
- **Backward Compatibility**: Legacy HIV-1 model still supported

### Usage:
```python
from config_loader import get_targets_config, get_scoring_weights

# Get available targets
targets = get_targets_config()
# {'HIV1_PROTEASE': {...}, 'SARS2_MPRO': {...}, ...}

# Get scoring weights
weights = get_scoring_weights()
# Adjust weights for different use cases
```

## 2. Enhanced ADMET Training

### Files Created:
- `train_admet_enhanced.py` - Enhanced training with model selection and calibration

### Features:
- **Model Selection**: Cross-validation to choose between RandomForest and XGBoost
- **Probability Calibration**: CalibratedClassifierCV for meaningful probabilities
- **Better Metrics**: ROC-AUC for classification, R² for regression
- **Improved Performance**: Better out-of-sample predictions

### Usage:
```bash
# Train a single ADMET model with enhanced features
python train_admet_enhanced.py --task hERG --cv-folds 5

# Available tasks: hERG, AMES, DILI, caco2_wang, HIA_Hou, BBB_Martins, 
#                  PPBR_AZ, CYP3A4_Veith, CYP2D6_Veith, Half_Life_Obach, LD50_Zhu
```

## 3. Medicinal Chemistry Filters

### Files Created:
- `medchem_filters.py` - PAINS filters, structural alerts, and drug-likeness checks

### Features:
- **PAINS Filtering**: Detects Pan Assay Interference Compounds
- **Structural Alerts**: Filters reactive groups, toxicophores
- **Atom/Ring Limits**: Configurable heavy atom and ring count constraints
- **Integration**: Seamlessly integrated into lead optimization pipeline

### Usage:
```python
from medchem_filters import passes_simple_medchem_filters, filter_molecules_by_medchem

# Check single molecule
if passes_simple_medchem_filters(smiles, check_pains=True):
    # Molecule passes filters
    pass

# Filter a list
filtered_smiles = filter_molecules_by_medchem(smiles_list)
```

## 4. Similarity Constraints

### Files Modified:
- `lead_optimization.py` - Added similarity filtering

### Features:
- **Tanimoto Similarity**: Computes similarity between parent and child molecules
- **Configurable Window**: Keep only mutations within similarity range (default: 0.4-0.9)
- **Prevents Drift**: Too low = random scaffold, too high = trivial changes
- **UI Controls**: Sliders in Streamlit app for easy adjustment

### Usage:
```python
from lead_optimization import optimize_leads, compute_similarity_to_parent

# Similarity is automatically computed and filtered
optimized_df = optimize_leads(
    hits_df=hits_df,
    hiv_model=model,
    admet_models=admet_models,
    min_similarity=0.4,  # Minimum Tanimoto similarity
    max_similarity=0.9    # Maximum Tanimoto similarity
)
```

## 5. CLI Pipeline Mode

### Files Created:
- `cli_pipeline.py` - Command-line interface for batch processing

### Features:
- **Headless Operation**: No UI, pure Python/pandas
- **Batch Processing**: Process entire libraries from CSV
- **Flexible Output**: CSV or Parquet format
- **Optional Optimization**: Can run lead optimization on top hits
- **Scalable**: Ready for cluster/cloud deployment

### Usage:
```bash
# Basic screening
python cli_pipeline.py \
  --input library.csv \
  --output results.parquet \
  --target HIV1_PROTEASE

# With lead optimization
python cli_pipeline.py \
  --input library.csv \
  --output results.parquet \
  --target HIV1_PROTEASE \
  --optimize \
  --n-optimization-parents 20 \
  --min-p-active 0.65 \
  --max-herg 0.7

# Skip ADMET (faster)
python cli_pipeline.py \
  --input library.csv \
  --output results.csv \
  --target HIV1_PROTEASE \
  --no-admet
```

## 6. Batch Docking Integration

### Files Created:
- `batch_docking.py` - Batch docking script for top N leads

### Features:
- **Top N Selection**: Dock only top-scoring leads (configurable)
- **CSV Integration**: Adds docking scores to existing DataFrames
- **Error Handling**: Gracefully handles docking failures
- **UI Integration**: Checkbox in Streamlit app for batch docking

### Usage:
```bash
# Batch dock top 20 leads
python batch_docking.py \
  --input results.csv \
  --output results_with_docking.csv \
  --receptor data/hiv1_protease/receptor.pdbqt \
  --top-n 20 \
  --output-dir docking_results
```

## Updated UI Features

### Streamlit App (`app.py`):
- **Medchem Filters Checkbox**: Enable/disable PAINS and structural alert filtering
- **Similarity Sliders**: Adjust min/max similarity to parent
- **Batch Docking**: Checkbox to dock top N leads automatically
- **Config-Driven Scoring**: Uses weights from `config/scoring_weights.yaml`

## Configuration Files

### `config/targets.yaml`
Defines all available targets:
```yaml
targets:
  HIV1_PROTEASE:
    chembl_id: "CHEMBL243"
    name: "HIV-1 Protease"
    model_path: "models/hiv1_protease_rf.pkl"
  SARS2_MPRO:
    chembl_id: "CHEMBL4301553"
    name: "SARS-CoV-2 Main Protease"
    model_path: "models/sars2_mpro_rf.pkl"
```

### `config/scoring_weights.yaml`
Configurable scoring weights:
```yaml
scoring:
  target_weight: 0.4
  tox_penalty:
    hERG: 0.3
    DILI: 0.2
    AMES: 0.2
  absorption_weight:
    HIA_Hou: 0.1
    caco2_wang: 0.1
```

## Next Steps (Future Enhancements)

1. **Additional ADMET Endpoints**: Solubility, LogP, P-gp efflux, more CYP isoforms
2. **Advanced Generation**: Graph neural networks, transformer-based generation
3. **Parallel Docking**: Multi-core/GPU docking acceleration
4. **Confidence Flags**: Distance from training distribution, calibration-based confidence
5. **Data Provenance**: Track dataset sources, assay types, publication dates
6. **Cloud Deployment**: AWS/GCP batch processing, distributed training

## Testing

All new modules have been tested for:
- ✅ Syntax correctness
- ✅ Import compatibility
- ✅ Integration with existing pipeline
- ✅ Backward compatibility

## Dependencies Added

- `pyyaml>=6.0` - For YAML configuration files

## Migration Guide

### For Existing Users:
1. Install new dependency: `pip install pyyaml`
2. Create `config/` directory: `mkdir config`
3. Configuration files are optional - defaults will be used if not present
4. All existing functionality remains unchanged

### For New Users:
1. Follow setup instructions in `README.md`
2. Train models using `make train` or individual scripts
3. Use CLI mode for batch processing: `python cli_pipeline.py --help`
4. Customize scoring weights in `config/scoring_weights.yaml`

