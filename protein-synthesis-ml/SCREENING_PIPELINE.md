# Drug Discovery Screening Pipeline

Complete implementation of a multi-step drug discovery screening pipeline with HIV-1 protease target activity and full ADMET panel.

## Overview

This pipeline evaluates small molecules across:
- **Target Activity**: HIV-1 protease binding (from ChEMBL)
- **ADMET Properties**: Absorption, Distribution, Metabolism, Excretion, and Toxicity

## File Structure

### Core Modules

- **`admet_loader.py`**: Loads all ADMET datasets from TDC with unified interface
- **`featurization.py`**: SMILES → Morgan fingerprint conversion
- **`pipeline.py`**: Main screening brain that evaluates molecules across all models
- **`metrics_and_plots.py`**: Evaluation and visualization utilities

### Training Scripts

- **`main.py`**: Trains HIV-1 protease model and saves to `models/hiv_protease_rf.pkl`
- **`train_admet.py`**: Generic ADMET model training (supports all tasks)

### Interfaces

- **`screen_cli.py`**: Command-line interface for single-molecule screening
- **`app.py`**: Streamlit web UI with full ADMET panel and batch processing
- **`nl_queries.py`**: Placeholder for natural-language query filtering

## ADMET Tasks Supported

### Absorption (A)
- `caco2_wang`: Caco-2 permeability (regression)
- `HIA_Hou`: Human Intestinal Absorption (classification)

### Distribution (D)
- `BBB_Martins`: Blood-Brain Barrier penetration (classification)
- `PPBR_AZ`: Plasma Protein Binding Rate (regression)

### Metabolism (M)
- `CYP3A4_Veith`: CYP3A4 inhibition (classification)
- `CYP2D6_Veith`: CYP2D6 inhibition (classification)

### Excretion (E)
- `Half_Life_Obach`: Half-life (regression)

### Toxicity (T)
- `hERG`: hERG toxicity (classification)
- `AMES`: AMES mutagenicity (classification)
- `DILI`: Drug-Induced Liver Injury (classification)
- `LD50_Zhu`: LD50 toxicity (regression)

## Usage

### 0. Batch Library Screening (Recommended Workflow)

The realistic screening workflow screens large libraries efficiently:

```bash
# Screen a library of molecules
python screen_library.py \
    --input library.csv \
    --output results.csv \
    --threshold 0.8

# With custom column names
python screen_library.py \
    --input library.csv \
    --output results.csv \
    --smiles-col "SMILES" \
    --id-col "compound_id" \
    --threshold 0.75
```

**Workflow:**
1. Loads all molecules from CSV
2. Computes HIV-1 p_active for all
3. Filters to hits with p_active >= threshold
4. Runs ADMET models only on the filtered subset (efficient!)
5. Computes composite scores
6. Saves comprehensive results table

**Input CSV format:**
```csv
smiles,compound_id
CCO,ethanol
Cc1ccccc1,toluene
...
```

**Output CSV includes:**
- Original columns (smiles, compound_id, etc.)
- `hiv1_p_active`: HIV-1 protease activity probability
- `{task}_prob` or `{task}_value`: ADMET predictions
- `composite_score`: Overall drug-likeness score

### 1. Train All Models

```bash
# Train HIV-1 protease target model
python main.py

# Train ADMET models (one at a time)
python train_admet.py --task hERG
python train_admet.py --task AMES
python train_admet.py --task DILI
python train_admet.py --task LD50_Zhu
python train_admet.py --task caco2_wang
python train_admet.py --task HIA_Hou
python train_admet.py --task BBB_Martins
python train_admet.py --task PPBR_AZ
python train_admet.py --task CYP3A4_Veith
python train_admet.py --task CYP2D6_Veith
python train_admet.py --task Half_Life_Obach
```

### 2. Run Screening

**Batch Library Screening (Recommended):**
```bash
python screen_library.py --input library.csv --output results.csv --threshold 0.8
```

**Single Molecule (CLI):**
```bash
python screen_cli.py --smiles "CC(C)(C)NC(=O)..."
```

**Web UI:**
```bash
streamlit run app.py
```

Then:
- Paste SMILES in the text input
- Click "Run ADMET Panel"
- View comprehensive results with traffic-light risk indicators

**Batch Processing:**
- Upload CSV with `smiles` column
- Click "Run Batch Screening"
- Download results as CSV

### 3. Natural Language Queries (Placeholder)

```python
from nl_queries import filter_reports_by_text_query
from pipeline import evaluate_single_smiles

reports = [evaluate_single_smiles(smi) for smi in smiles_list]
safe_compounds = filter_reports_by_text_query(reports, "safe")
```

## Model Storage

Models are saved in:
- `models/hiv_protease_rf.pkl` - HIV-1 protease classifier
- `models/admet/<task>_clf.pkl` - ADMET classification models
- `models/admet/<task>_reg.pkl` - ADMET regression models

## Composite Score

The pipeline computes a composite "drug-likeness" score that:
- Rewards high target potency (HIV p_active)
- Penalizes high toxicity probabilities
- Considers absorption/distribution properties

Score formula can be tuned in `pipeline.composite_score()`.

## Traffic-Light Risk Indicators

The UI uses color coding:
- 🟢 **Green**: Good/acceptable property
- 🟡 **Yellow**: Borderline/needs attention
- 🔴 **Red**: Poor/problematic property

Thresholds are defined in `app.get_risk_color()`.

## Future Extensions

- **LLM Integration**: Replace `nl_queries.py` with LLM-based query understanding
- **Additional Models**: Easy to add more ADMET tasks via `admet_loader.py`
- **Custom Scoring**: Modify `composite_score()` for different therapeutic areas
- **Model Ensembles**: Extend to support multiple models per task

## Notes

- All models use the same Morgan fingerprint featurization (2048 bits, radius=2)
- Models are lazy-loaded on first use
- Invalid SMILES are handled gracefully with clear error messages
- The pipeline is stateless and can be easily parallelized

