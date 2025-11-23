"""
Protein Activity ML Pipeline

Main entry point for the drug discovery toolkit.
Supports two modes:
- Mode 1: Target activity prediction (HIV-1 protease, CHEMBL243)
- Mode 2: ADMET prediction (via train_admet.py)
"""

import argparse
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    r2_score,
    mean_squared_error,
)

from data_utils import load_bioactivity
from featurization import smiles_to_matrix, smiles_to_morgan_fp
from metrics_and_plots import plot_roc_pr, print_threshold_table


def predict_activity(
    smiles_list: list[str], model, radius: int = 2, n_bits: int = 2048
) -> pd.DataFrame:
    """
    Predict activity for a list of SMILES strings.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings
    model : sklearn model
        Trained classifier model
    radius : int
        Morgan fingerprint radius
    n_bits : int
        Number of fingerprint bits

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: smiles, p_active
    """
    fps = []
    valid_smiles = []
    for smi in smiles_list:
        fp = smiles_to_morgan_fp(smi, radius, n_bits)
        if fp is not None:
            fps.append(fp)
            valid_smiles.append(smi)
    X_new = np.array(fps)
    proba = model.predict_proba(X_new)[:, 1]
    return pd.DataFrame({"smiles": valid_smiles, "p_active": proba})


def run_target_activity_pipeline():
    """Run the HIV-1 protease target activity prediction pipeline."""
    print("=" * 60)
    print("Mode 1: Target Activity Prediction (HIV-1 Protease)")
    print("=" * 60)

    # --------------------------------------------------
    # Step 1: Load and preprocess data
    # --------------------------------------------------
    print("\nLoading data...")
    df = load_bioactivity("bioactivity.csv", ic50_threshold_nM=1000.0)

    print("\nData summary:")
    print(df.head())
    print("\nActive/Inactive distribution:")
    print(df["active"].value_counts())

    # --------------------------------------------------
    # Step 2: Convert SMILES to fingerprints
    # --------------------------------------------------
    print("\nConverting SMILES to fingerprints...")
    X, idx = smiles_to_matrix(df["smiles"], radius=2, n_bits=2048)
    # y and smiles are aligned with X via idx
    y = df["active"].values[idx]
    smiles_clean = df["smiles"].iloc[idx].tolist()

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Valid SMILES count: {len(smiles_clean)}")

    # --------------------------------------------------
    # Step 3: Train classification model
    # --------------------------------------------------
    print("\nTraining RandomForest classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )

    clf.fit(X_train, y_train)

    # Save HIV protease model
    os.makedirs("models", exist_ok=True)
    model_path = "models/hiv_protease_rf.pkl"
    joblib.dump(clf, model_path)
    print(f"\nModel saved to: {model_path}")

    # --------------------------------------------------
    # Step 4: Evaluate classification model
    # --------------------------------------------------
    print("\nModel evaluation:")
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --------------------------------------------------
    # Step 5: Make predictions on sample data
    # --------------------------------------------------
    print("\nMaking predictions on sample SMILES...")
    unlabeled_smiles = df["smiles"].sample(50, random_state=0).tolist()
    ranked = predict_activity(unlabeled_smiles, clf)
    ranked.sort_values("p_active", ascending=False, inplace=True)
    print("\nTop 10 predictions:")
    print(ranked.head(10))

    # --------------------------------------------------
    # Step 6: Plot ROC and PR curves
    # --------------------------------------------------
    print("\nGenerating ROC and PR curves...")
    plot_roc_pr(y_test, y_proba, title_prefix="HIV-1 Protease")

    # --------------------------------------------------
    # Step 7: Threshold analysis
    # --------------------------------------------------
    print("\nThreshold vs Precision/Recall Analysis")
    print_threshold_table(y_test, y_proba)

    # --------------------------------------------------
    # Step 8: Regression on pIC50
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("Training pIC50 Regression Model")
    print("=" * 60)

    # df_reg corresponds exactly to rows used in X (via idx)
    df_reg = df.iloc[idx].copy()
    df_reg["activity_value_molar"] = df_reg["activity_value"] * 1e-9
    # avoid log(0)
    mask = df_reg["activity_value_molar"] > 0
    df_reg = df_reg[mask].copy()
    df_reg["pIC50"] = -np.log10(df_reg["activity_value_molar"])

    # Align X_reg with filtered df_reg
    X_reg = X[mask.values]
    y_reg = df_reg["pIC50"].values

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=300,
        n_jobs=-1,
        random_state=42,
    )
    reg.fit(X_train_reg, y_train_reg)

    y_pred_reg = reg.predict(X_test_reg)
    print(f"\nRegression Metrics:")
    print(f"R^2:   {r2_score(y_test_reg, y_pred_reg):.4f}")
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    print(f"RMSE:  {rmse:.4f}")

    # --------------------------------------------------
    # Step 9: Generate hits.csv for downstream ADMET
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating hits.csv for ADMET screening")
    print("=" * 60)

    # Predict p_active for ALL valid molecules (X, y, smiles_clean)
    p_active_all = clf.predict_proba(X)[:, 1]

    results = pd.DataFrame(
        {
            "smiles": smiles_clean,
            "p_active": p_active_all,
            "label_active": y,  # original binary label (0/1)
        }
    )

    hit_threshold = 0.7  # you can tweak this
    active_hits = results[results["p_active"] >= hit_threshold].copy()

    print(f"Total molecules: {len(results)}")
    print(f"Hits with p_active >= {hit_threshold}: {len(active_hits)}")

    hits_path = "hits.csv"
    active_hits.to_csv(hits_path, index=False)
    print(f"\nSaved {len(active_hits)} hits to {hits_path}")
    print("\nSample of hits.csv:")
    print(active_hits.head())

    print("\n" + "=" * 60)
    print("Target Activity Pipeline Complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Drug Discovery Toolkit - Target Activity & ADMET Prediction"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="target",
        choices=["target", "admet"],
        help=(
            "Pipeline mode: 'target' for HIV-1 protease activity, "
            "'admet' for ADMET model training hints"
        ),
    )

    args = parser.parse_args()

    if args.mode == "target":
        run_target_activity_pipeline()
    elif args.mode == "admet":
        print("=" * 60)
        print("Mode 2: ADMET Prediction")
        print("=" * 60)
        print("\nTo train ADMET models, please run:")
        print("  python train_admet.py --dataset herg")
        print("  python train_admet.py --dataset caco2")
        print("  python train_admet.py --dataset ld50")
        print("\nAvailable datasets:")
        print("  - herg:  hERG toxicity prediction")
        print("  - caco2: Caco-2 permeability prediction")
        print("  - ld50:  LD50 toxicity prediction")
        print("=" * 60)


if __name__ == "__main__":
    main()
