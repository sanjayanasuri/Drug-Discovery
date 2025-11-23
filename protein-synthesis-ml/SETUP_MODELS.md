# Model Training Guide

## Quick Start: Train All Models

### Option 1: Train All ADMET Models at Once (Recommended)

```bash
# Make sure you're in conda base environment
conda activate base
cd /Users/sanjayanasuri/protein-synthesis-ml

# Train all ADMET models automatically
python train_all_admet.py
```

This will train all 11 ADMET tasks sequentially. It may take 10-30 minutes depending on dataset sizes.

### Option 2: Train Models Individually

```bash
# Train HIV-1 protease model (required)
python main.py

# Train ADMET models one at a time
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

## Required Models

### Minimum (for basic functionality):
- **HIV-1 Protease**: `models/hiv_protease_rf.pkl`
  - Required for activity screening
  - Train with: `python main.py`

### Full Pipeline (for complete ADMET panel):
- All 11 ADMET models in `models/admet/`:
  - Classification: `{task}_clf.pkl`
  - Regression: `{task}_reg.pkl`

## Model Storage

After training, models are saved to:
```
models/
├── hiv_protease_rf.pkl          # HIV-1 protease classifier
└── admet/
    ├── hERG_clf.pkl
    ├── AMES_clf.pkl
    ├── DILI_clf.pkl
    ├── LD50_Zhu_reg.pkl
    ├── caco2_wang_reg.pkl
    ├── HIA_Hou_clf.pkl
    ├── BBB_Martins_clf.pkl
    ├── PPBR_AZ_reg.pkl
    ├── CYP3A4_Veith_clf.pkl
    ├── CYP2D6_Veith_clf.pkl
    └── Half_Life_Obach_reg.pkl
```

## Verification

Check which models are available:

```bash
# List all trained models
ls -la models/
ls -la models/admet/

# Or check in Python
python -c "from pipeline import registry; registry.load(); print('HIV model:', registry.has_model('HIV_protease')); print('ADMET models:', [k for k in ['hERG', 'AMES', 'DILI'] if registry.has_model(k)])"
```

## Troubleshooting

### Problem: "No module named 'tdc'"
**Solution:** Install PyTDC:
```bash
python -m pip install pytdc
```

### Problem: Dataset download is slow
**Solution:** This is normal. TDC datasets are downloaded on first use. Subsequent runs will be faster.

### Problem: Training fails for a specific task
**Solution:** 
- Check your internet connection (TDC needs to download data)
- Try training that task individually: `python train_admet.py --task <task_name>`
- Some tasks may have different dataset availability

### Problem: Out of memory during training
**Solution:**
- Train models one at a time instead of using `train_all_admet.py`
- Close other applications
- Consider training on a subset of data (modify train_admet.py temporarily)

## Time Estimates

- HIV-1 Protease: ~1-2 minutes
- Each ADMET model: ~2-5 minutes (depends on dataset size)
- All ADMET models: ~20-40 minutes total

## Next Steps

Once models are trained:
1. Launch Streamlit UI: `python -m streamlit run app.py`
2. Upload your molecular library
3. Run the full screening pipeline!

