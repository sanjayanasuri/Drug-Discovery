"""
train_admet.py

Generic ADMET model training script.
Trains classification or regression models on any ADMET task from TDC.
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from admet_loader import load_admet_dataset, ADMET_TASKS
from featurization import smiles_to_matrix
from metrics_and_plots import (
    evaluate_classifier,
    evaluate_regressor,
    plot_roc_pr,
    print_threshold_table,
)


def train_admet_model(task_key: str, save_model: bool = True):
    """
    Train a RandomForest model on an ADMET dataset.
    
    Parameters
    ----------
    task_key : str
        One of the keys in ADMET_TASKS
    save_model : bool
        Whether to save the trained model to disk
    """
    if task_key not in ADMET_TASKS:
        raise ValueError(
            f"Unknown task: {task_key}. Available tasks: {list(ADMET_TASKS.keys())}"
        )
    
    task_config = ADMET_TASKS[task_key]
    task_type = task_config["type"]
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Loading {task_key} dataset...")
    print(f"{'='*60}")
    
    df, detected_type = load_admet_dataset(task_key)
    
    if detected_type != task_type:
        print(f"Warning: Task type mismatch. Expected {task_type}, got {detected_type}")
    
    print(f"\nDataset loaded: {len(df)} compounds")
    
    # Handle label column name
    if task_type == "classification":
        label_col = "label"
        if label_col not in df.columns:
            raise ValueError(f"Classification task {task_key} missing 'label' column")
        print(f"Label distribution:")
        print(df[label_col].value_counts())
    else:  # regression
        label_col = "value"
        if label_col not in df.columns:
            raise ValueError(f"Regression task {task_key} missing 'value' column")
        print(f"Value statistics:")
        print(df[label_col].describe())
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Featurize SMILES
    print(f"\n{'='*60}")
    print("Converting SMILES to Morgan fingerprints...")
    print(f"{'='*60}")
    
    X, idx = smiles_to_matrix(df["smiles"], radius=2, n_bits=2048)
    
    if task_type == "classification":
        y = df[label_col].values[idx]
    else:  # regression
        y = df[label_col].values[idx]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Valid SMILES: {len(idx)} / {len(df)}")
    
    # Train/test split
    if task_type == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:  # regression
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Train model
    print(f"\n{'='*60}")
    print(f"Training RandomForest {'Classifier' if task_type == 'classification' else 'Regressor'}...")
    print(f"{'='*60}")
    
    if task_type == "classification":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
    else:  # regression
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    if task_type == "classification":
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = evaluate_classifier(
            y_test, y_proba, threshold=0.5, model_name=f"{task_key} RandomForest"
        )
        
        # Plot ROC and PR curves
        print(f"\n{'='*60}")
        print("Generating ROC and PR curves...")
        print(f"{'='*60}")
        plot_roc_pr(y_test, y_proba, title_prefix=task_key)
        
        # Threshold analysis
        print(f"\n{'='*60}")
        print("Threshold vs Precision/Recall Analysis")
        print(f"{'='*60}")
        print_threshold_table(y_test, y_proba)
    else:  # regression
        y_pred = model.predict(X_test)
        evaluate_regressor(y_test, y_pred, model_name=f"{task_key} RandomForest")
    
    # Save model
    if save_model:
        os.makedirs("models/admet", exist_ok=True)
        
        if task_type == "classification":
            model_path = os.path.join("models", "admet", f"{task_key}_clf.pkl")
        else:  # regression
            model_path = os.path.join("models", "admet", f"{task_key}_reg.pkl")
        
        joblib.dump(model, model_path)
        print(f"\n{'='*60}")
        print(f"Model saved to: {model_path}")
        print(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print(f"Training complete for {task_key}!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Train ADMET prediction models on TDC datasets"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(ADMET_TASKS.keys()),
        help=f"ADMET task to train on. Available: {', '.join(ADMET_TASKS.keys())}",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the trained model to disk",
    )
    
    args = parser.parse_args()
    train_admet_model(args.task, save_model=not args.no_save)


if __name__ == "__main__":
    main()
