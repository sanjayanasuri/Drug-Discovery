.PHONY: help setup train train-all train-hiv train-admet app clean

help:
	@echo "Drug Discovery Pipeline - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Create conda environment from environment.yml"
	@echo ""
	@echo "Training:"
	@echo "  make train          - Train all models (HIV-1 + ADMET)"
	@echo "  make train-hiv      - Train HIV-1 protease model only"
	@echo "  make train-admet    - Train all ADMET models"
	@echo ""
	@echo "Running:"
	@echo "  make app            - Run Streamlit app"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove __pycache__ and .pyc files"

setup:
	@echo "Creating conda environment..."
	conda env create -f environment.yml
	@echo "Environment created! Activate with: conda activate hiv-ml"

train: train-hiv train-admet
	@echo "✅ All models trained!"

train-hiv:
	@echo "Training HIV-1 protease model..."
	python main.py

train-admet:
	@echo "Training all ADMET models (this may take 20-40 minutes)..."
	python train_all_admet.py

train-admet-enhanced:
	@echo "Training ADMET models with enhanced features (model selection, calibration)..."
	@echo "This will train each model with cross-validation and calibration."
	@echo "Usage: python train_admet_enhanced.py --task <TASK_KEY>"
	@echo "Available tasks: hERG, AMES, DILI, caco2_wang, HIA_Hou, BBB_Martins, PPBR_AZ, CYP3A4_Veith, CYP2D6_Veith, Half_Life_Obach, LD50_Zhu"

app:
	@echo "Starting Streamlit app..."
	python -m streamlit run app.py --server.port 8501

clean:
	@echo "Cleaning Python cache files..."
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned!"

