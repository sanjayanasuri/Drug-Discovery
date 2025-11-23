# Streamlit UI Guide

## New Workflow-Based Interface

The UI has been redesigned as a step-by-step drug discovery pipeline with clear workflow stages.

## Workflow Steps

### Step 1: Upload Library 📤
- Upload a CSV file with a `smiles` column
- Preview the uploaded data
- Library is stored in session state for subsequent steps

### Step 2: Activity Screening 🎯
- Automatically featurizes all SMILES (cached for speed)
- Runs HIV-1 protease activity model on entire library
- Displays results with p_active scores
- **Threshold slider** to filter active hits
- Shows filtered hits in sortable table
- Download activity results as CSV

### Step 3: ADMET Evaluation 🔬
- **Automatic mode**: Evaluates all active hits (default)
- **Manual mode**: Select specific molecules from the hits table
- Runs all available ADMET models on selected molecules
- Computes composite scores
- Displays results in two tabs:
  - **Results Table**: Full data with all ADMET predictions
  - **Summary Statistics**: Descriptive stats for all properties
- Download ADMET results as CSV

### Advanced Mode ⚙️
- Manual SMILES input (one per line)
- Quick single-molecule evaluation
- Useful for testing or ad-hoc queries

## Key Features

### Caching for Performance
- **Model loading**: Cached with `@st.cache_resource`
- **SMILES featurization**: Cached with `@st.cache_data`
- Significantly faster on repeated runs

### Session State Management
- Uploaded library persists across tabs
- Activity results stored after screening
- Selected hits maintained for ADMET step
- Clear button to reset workflow

### Automatic Hit Detection
- By default, all molecules passing the activity threshold are selected for ADMET
- User can override with manual selection
- Efficient: only runs ADMET on active compounds

### Composite Scoring
- Uses the same `composite_score()` function from `pipeline.py`
- Rewards high target potency
- Penalizes toxicity
- Considers absorption/distribution properties

## Usage Example

1. **Upload**: Upload `library.csv` with SMILES column
2. **Screen**: Click "Run Activity Screening" → wait for results
3. **Filter**: Adjust threshold slider (e.g., 0.8) → see filtered hits
4. **Evaluate**: Go to ADMET tab → click "Run ADMET Panel" → wait
5. **Review**: Check results table and summary statistics
6. **Download**: Download final CSV with all predictions and scores

## Input CSV Format

```csv
smiles,compound_id
CCO,ethanol
Cc1ccccc1,toluene
CC(C)NC(=O)OCc1ccccc1,example_1
```

## Output CSV Format

The final ADMET results CSV includes:
- Original columns (smiles, compound_id, etc.)
- `hiv1_p_active`: HIV-1 protease activity probability
- `{task}_prob`: ADMET classification probabilities
- `{task}_value`: ADMET regression values
- `composite_score`: Overall drug-likeness score

## Tips

- **Large libraries**: The caching makes re-runs much faster
- **Threshold tuning**: Start with 0.5, adjust based on hit rate
- **Manual selection**: Use when you want to evaluate specific compounds
- **Session persistence**: Results persist until you clear or refresh

