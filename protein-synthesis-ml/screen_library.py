"""
screen_library.py

Batch screening pipeline for large molecular libraries.
Implements the realistic workflow:
1. Screen all molecules for HIV-1 activity
2. Filter to high-activity hits (p_active >= threshold)
3. Run ADMET on the filtered subset
4. Store comprehensive results in CSV
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from featurization import smiles_to_morgan_fp
from pipeline import registry, composite_score
from admet_loader import ADMET_TASKS


def screen_library(
    input_csv: str,
    output_csv: str,
    smiles_col: str = "smiles",
    id_col: Optional[str] = None,
    p_active_threshold: float = 0.8,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Batch screening pipeline for a library of molecules.
    
    Workflow:
    1. Load a library of molecules from CSV
    2. Compute HIV-1 p_active for all molecules
    3. Filter to those with p_active >= threshold
    4. Run ADMET models on the filtered subset
    5. Compute composite scores
    6. Save and return the final table
    
    Parameters
    ----------
    input_csv : str
        Path to input CSV file with SMILES column
    output_csv : str
        Path to output CSV file for results
    smiles_col : str
        Name of SMILES column in input CSV (default: "smiles")
    id_col : str, optional
        Name of ID/name column to preserve (if exists)
    p_active_threshold : float
        Minimum HIV-1 p_active to keep for ADMET screening (default: 0.8)
    verbose : bool
        Whether to print progress messages
        
    Returns
    -------
    pd.DataFrame
        Results dataframe with all predictions and scores
    """
    if verbose:
        print("=" * 70)
        print("BATCH SCREENING PIPELINE")
        print("=" * 70)
        print(f"\nLoading library from: {input_csv}")
    
    # Load library
    df = pd.read_csv(input_csv)
    
    if smiles_col not in df.columns:
        raise ValueError(f"Expected a '{smiles_col}' column in {input_csv}")
    
    if verbose:
        print(f"Loaded {len(df)} molecules")
    
    # Load all models once
    if verbose:
        print("\nLoading models...")
    registry.load()
    
    hiv_model = registry.get("HIV_protease")
    if hiv_model is None:
        raise ValueError("HIV protease model not found. Run 'python main.py' first.")
    
    # Featurize all SMILES
    if verbose:
        print("\nFeaturizing SMILES...")
    fps = []
    valid_idx = []
    invalid_smiles = []
    
    for i, smi in enumerate(df[smiles_col]):
        fp = smiles_to_morgan_fp(str(smi))
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
        else:
            invalid_smiles.append(i)
    
    if verbose:
        print(f"Valid SMILES: {len(fps)} / {len(df)}")
        if invalid_smiles:
            print(f"Invalid SMILES: {len(invalid_smiles)}")
    
    if not fps:
        raise ValueError("No valid SMILES found in input file")
    
    X = np.array(fps)
    sub = df.iloc[valid_idx].copy()
    
    # HIV-1 potency prediction
    if verbose:
        print("\nComputing HIV-1 protease activity...")
    p_active = hiv_model.predict_proba(X)[:, 1]
    sub["hiv1_p_active"] = p_active
    
    if verbose:
        print(f"HIV-1 p_active range: [{p_active.min():.3f}, {p_active.max():.3f}]")
        print(f"Mean p_active: {p_active.mean():.3f}")
    
    # Filter by potency threshold
    hits = sub[sub["hiv1_p_active"] >= p_active_threshold].copy()
    
    if verbose:
        print(f"\nFiltering to p_active >= {p_active_threshold}")
        print(f"Hits passing threshold: {len(hits)} / {len(sub)}")
    
    if hits.empty:
        if verbose:
            print("No molecules passed the HIV-1 activity threshold.")
        # Still save results with just HIV predictions
        sub.to_csv(output_csv, index=False)
        return sub
    
    # Get feature matrix for hits
    hit_mask = sub["hiv1_p_active"] >= p_active_threshold
    X_hits = X[hit_mask.values]
    
    # Run ADMET models on hits
    if verbose:
        print("\nRunning ADMET models on hits...")
    
    admet_tasks = list(ADMET_TASKS.keys())
    loaded_models = 0
    
    for task in admet_tasks:
        model = registry.get(task)
        if model is None:
            if verbose:
                print(f"  Skipping {task}: model not available")
            continue
        
        loaded_models += 1
        task_config = ADMET_TASKS[task]
        task_type = task_config["type"]
        
        if verbose:
            print(f"  Evaluating {task} ({task_type})...")
        
        if task_type == "classification":
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_hits)[:, 1]
                hits[f"{task}_prob"] = probs
            else:
                # Fallback to binary predictions
                preds = model.predict(X_hits)
                hits[f"{task}_pred"] = preds
        else:  # regression
            vals = model.predict(X_hits)
            hits[f"{task}_value"] = vals
    
    if verbose:
        print(f"\nEvaluated {loaded_models} ADMET models")
    
    # Compute composite score per molecule
    if verbose:
        print("\nComputing composite scores...")
    
    scores = []
    for _, row in hits.iterrows():
        admet_outputs: Dict[str, Any] = {}
        
        for task in admet_tasks:
            col_p = f"{task}_prob"
            col_v = f"{task}_value"
            col_pred = f"{task}_pred"
            
            if col_p in hits.columns:
                admet_outputs[task] = {"prob": row[col_p]}
            elif col_v in hits.columns:
                admet_outputs[task] = {"value": row[col_v]}
            elif col_pred in hits.columns:
                # Convert binary prediction to probability estimate
                admet_outputs[task] = {"prob": float(row[col_pred])}
        
        scores.append(
            composite_score(row["hiv1_p_active"], admet_outputs)
        )
    
    hits["composite_score"] = scores
    
    # Sort by composite score (descending)
    hits = hits.sort_values("composite_score", ascending=False)
    
    # Save results
    if verbose:
        print(f"\nSaving results to: {output_csv}")
    hits.to_csv(output_csv, index=False)
    
    if verbose:
        print("\n" + "=" * 70)
        print("SCREENING COMPLETE")
        print("=" * 70)
        print(f"\nResults summary:")
        print(f"  Total molecules screened: {len(df)}")
        print(f"  Valid SMILES: {len(sub)}")
        print(f"  Hits (p_active >= {p_active_threshold}): {len(hits)}")
        print(f"  ADMET models evaluated: {loaded_models}")
        print(f"\nTop 5 by composite score:")
        top_cols = ["hiv1_p_active", "composite_score"]
        if id_col and id_col in hits.columns:
            top_cols.insert(0, id_col)
        elif smiles_col in hits.columns:
            top_cols.insert(0, smiles_col)
        print(hits[top_cols].head().to_string())
    
    return hits


def main():
    """CLI entry point for batch screening."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch screening pipeline for molecular libraries"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file with SMILES column",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="smiles",
        help="Name of SMILES column (default: 'smiles')",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default=None,
        help="Name of ID/name column to preserve (optional)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum HIV-1 p_active threshold (default: 0.8)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    
    args = parser.parse_args()
    
    try:
        results = screen_library(
            input_csv=args.input,
            output_csv=args.output,
            smiles_col=args.smiles_col,
            id_col=args.id_col,
            p_active_threshold=args.threshold,
            verbose=not args.quiet,
        )
        print(f"\n✓ Screening complete. Results saved to {args.output}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

