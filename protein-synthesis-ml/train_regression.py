"""
train_regression.py

Regression on pIC50 for HIV-1 protease (CHEMBL243) compounds:
- RandomForestRegressor
- XGBRegressor
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

import xgboost as xgb  # pip install xgboost

from featurization import smiles_to_matrix
from data_utils import load_bioactivity
from metrics_and_plots import evaluate_regressor


def main():
    print("Loading data...")
    df = load_bioactivity("bioactivity.csv", ic50_threshold_nM=1000.0)

    # Use only rows with valid pIC50 (load_bioactivity already filters molar > 0)
    print("\nData summary (for regression):")
    print(df[["smiles", "activity_value", "pIC50"]].head())

    # Featurize SMILES and align pIC50 by index
    print("\nConverting SMILES to fingerprints...")
    X, idx = smiles_to_matrix(df["smiles"], radius=2, n_bits=2048)
    y_reg = df["pIC50"].values[idx]

    print(f"Feature matrix shape: {X.shape}")
    print(f"pIC50 labels shape: {y_reg.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_reg,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBRegressor": xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        ),
    }

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        evaluate_regressor(y_test, y_pred, model_name=name)


if __name__ == "__main__":
    main()
