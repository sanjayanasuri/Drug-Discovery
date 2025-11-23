"""
train_classification.py

Train multiple classifiers on HIV-1 protease (CHEMBL243) bioactivity:
- RandomForest
- Logistic Regression
- XGBoost

Outputs:
- ROC-AUC, classification reports
- ROC and PR curves for the best model
- Threshold vs precision/recall table
- Top 10 predicted actives on a sample subset
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

import xgboost as xgb  # pip install xgboost

from featurization import smiles_to_matrix
from data_utils import load_bioactivity
from metrics_and_plots import (
    evaluate_classifier,
    plot_roc_pr,
    print_threshold_table,
)


def main():
    print("Loading data...")
    df = load_bioactivity("bioactivity.csv", ic50_threshold_nM=1000.0)

    print("\nData summary:")
    print(df.head())
    print("\nActive/Inactive distribution:")
    print(df["active"].value_counts())

    # Featurize SMILES
    print("\nConverting SMILES to fingerprints...")
    X, idx = smiles_to_matrix(df["smiles"], radius=2, n_bits=2048)
    y = df["active"].values[idx]

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Define models
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = evaluate_classifier(y_test, y_proba, threshold=0.5, model_name=name)

        roc_auc = roc_auc_score(y_test, y_proba)
        roc = (
            name,
            y_proba,
            y_pred,
            model,
            roc_auc,  # store ROC-AUC for picking the best later
        )
        results[name] = roc

    # Select best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k][4])
    _, best_proba, _, best_model, best_auc = results[best_name]

    print(f"\nBest model by ROC-AUC: {best_name} (AUC = {best_auc:.4f})")

    # Plot ROC and PR for best model
    plot_roc_pr(y_test, best_proba, title_prefix=best_name)

    # Threshold analysis for best model
    print_threshold_table(y_test, best_proba)

    # Use best model to rank a subset of molecules
    print("\nMaking predictions on sample SMILES...")
    sampled_smiles = df["smiles"].sample(50, random_state=0).tolist()

    # Re-featurize the sample
    from featurization import smiles_to_morgan_fp

    fps = []
    val_smiles = []
    for smi in sampled_smiles:
        fp = smiles_to_morgan_fp(smi)
        if fp is not None:
            fps.append(fp)
            val_smiles.append(smi)

    X_new = np.array(fps)
    proba_new = best_model.predict_proba(X_new)[:, 1]
    ranked = pd.DataFrame({"smiles": val_smiles, "p_active": proba_new})
    ranked.sort_values("p_active", ascending=False, inplace=True)

    print("\nTop 10 predictions (by p_active):")
    print(ranked.head(10))


if __name__ == "__main__":
    main()
