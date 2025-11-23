"""
cli_pipeline.py

Command-line interface for batch processing of molecular libraries.
Runs the full pipeline without UI for large-scale screening.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from pipeline import registry, composite_score
from featurization import smiles_to_matrix
from admet_loader import ADMET_TASKS
from lead_optimization import optimize_leads
from config_loader import get_targets_config, get_scoring_weights


def predict_target_activity_batch(X: np.ndarray, target_model) -> np.ndarray:
    """Predict target activity for a batch of molecules."""
    if target_model is None:
        return np.zeros(len(X))
    if hasattr(target_model, "predict_proba"):
        return target_model.predict_proba(X)[:, 1]
    else:
        return np.zeros(len(X))


def predict_admet_batch(X: np.ndarray, task_key: str, model, task_type: str) -> np.ndarray:
    """Predict ADMET property for a batch of molecules."""
    if task_type == "classification":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X).astype(float)
    else:  # regression
        return model.predict(X)


def run_pipeline(
    input_csv: str,
    output_path: str,
    target_key: Optional[str] = None,
    run_admet: bool = True,
    run_optimization: bool = False,
    n_optimization_parents: int = 20,
    min_p_active: float = 0.65,
    max_herg: float = 0.7,
) -> pd.DataFrame:
    """
    Run the full screening pipeline on a CSV file.
    
    Parameters
    ----------
    input_csv : str
        Path to input CSV with 'smiles' column
    output_path : str
        Path to save results (CSV or Parquet)
    target_key : str, optional
        Target key from config. If None, uses first available target.
    run_admet : bool
        Whether to run ADMET evaluation
    run_optimization : bool
        Whether to run lead optimization
    n_optimization_parents : int
        Number of parents for optimization
    min_p_active : float
        Minimum p_active threshold
    max_herg : float
        Maximum hERG probability
        
    Returns
    -------
    pd.DataFrame
        Results DataFrame
    """
    print("=" * 60)
    print("Drug Discovery Pipeline - CLI Mode")
    print("=" * 60)
    
    # Load input
    print(f"\nLoading input: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if "smiles" not in df.columns:
        raise ValueError("Input CSV must contain 'smiles' column")
    
    print(f"Loaded {len(df)} molecules")
    
    # Load models
    print("\nLoading models...")
    registry.load()
    
    # Select target
    targets_config = get_targets_config()
    if target_key is None:
        # Use first available target
        target_key = list(targets_config.keys())[0]
        print(f"No target specified, using: {target_key}")
    
    if target_key not in targets_config:
        raise ValueError(f"Unknown target key: {target_key}. Available: {list(targets_config.keys())}")
    
    target_config = targets_config[target_key]
    target_name = target_config["name"]
    
    # Get target model
    target_model = registry.get_target_model(target_name)
    if target_model is None:
        raise ValueError(f"Target model not found for {target_name}. Please train it first.")
    
    print(f"Using target: {target_name} ({target_config['chembl_id']})")
    
    # Featurize
    print("\nFeaturizing SMILES...")
    smiles_list = df["smiles"].astype(str).tolist()
    X, idx = smiles_to_matrix(smiles_list)
    valid_smiles = [smiles_list[i] for i in idx]
    
    print(f"Valid SMILES: {len(valid_smiles)} / {len(smiles_list)}")
    
    # Predict target activity
    print("\nPredicting target activity...")
    p_active = predict_target_activity_batch(X, target_model)
    
    # Create results DataFrame
    results_df = df.iloc[idx].copy().reset_index(drop=True)
    results_df["p_active"] = p_active
    results_df["smiles"] = valid_smiles
    
    # ADMET evaluation
    if run_admet:
        print("\nRunning ADMET evaluation...")
        admet_models = {
            task: registry.get(task)
            for task in ADMET_TASKS.keys()
            if registry.has_model(task)
        }
        
        print(f"Evaluating {len(admet_models)} ADMET properties...")
        
        for task_key, model in admet_models.items():
            if model is None:
                continue
            
            task_config = ADMET_TASKS[task_key]
            task_type = task_config["type"]
            
            predictions = predict_admet_batch(X, task_key, model, task_type)
            
            if task_type == "classification":
                results_df[f"{task_key}_prob"] = predictions
            else:
                results_df[f"{task_key}_value"] = predictions
        
        # Compute composite scores
        print("\nComputing composite scores...")
        weights_config = get_scoring_weights()
        composite_scores = []
        
        for _, row in results_df.iterrows():
            admet_outputs = {}
            for task_key in ADMET_TASKS.keys():
                prob_col = f"{task_key}_prob"
                value_col = f"{task_key}_value"
                
                if prob_col in results_df.columns:
                    admet_outputs[task_key] = {"prob": row[prob_col]}
                elif value_col in results_df.columns:
                    admet_outputs[task_key] = {"value": row[value_col]}
            
            score = composite_score(row["p_active"], admet_outputs, weights_config)
            composite_scores.append(score)
        
        results_df["composite_score"] = composite_scores
        results_df = results_df.sort_values("composite_score", ascending=False)
    
    # Lead optimization (optional)
    if run_optimization and run_admet:
        print("\nRunning lead optimization...")
        
        # Filter to top hits
        hits_df = results_df[
            (results_df["p_active"] >= min_p_active)
        ].copy()
        
        if "hERG_prob" in hits_df.columns:
            hits_df = hits_df[hits_df["hERG_prob"] <= max_herg]
        
        if len(hits_df) > 0:
            admet_models = {
                task: registry.get(task)
                for task in ADMET_TASKS.keys()
                if registry.has_model(task)
            }
            
            optimized_df = optimize_leads(
                hits_df=hits_df,
                hiv_model=target_model,
                admet_models=admet_models,
                composite_fn=composite_score,
                n_parents=n_optimization_parents,
                n_children_per_parent=10,
                min_hiv1_p_active=min_p_active,
                max_hERG_prob=max_herg,
                enforce_lipinski=True,
                enforce_medchem_filters=True,
                min_similarity=0.4,
                max_similarity=0.9
            )
            
            if len(optimized_df) > 0:
                optimized_df["source"] = "optimized"
                results_df["source"] = "original"
                
                # Combine results
                results_df = pd.concat([results_df, optimized_df], ignore_index=True)
                results_df = results_df.sort_values("composite_score", ascending=False)
                print(f"Added {len(optimized_df)} optimized leads")
    
    # Save results
    print(f"\nSaving results to: {output_path}")
    output_path_obj = Path(output_path)
    
    if output_path_obj.suffix == ".parquet":
        results_df.to_parquet(output_path, index=False)
    else:
        results_df.to_csv(output_path, index=False)
    
    print(f"✅ Pipeline complete! Results saved: {len(results_df)} molecules")
    print(f"   Top composite score: {results_df['composite_score'].max():.3f}" if "composite_score" in results_df.columns else "")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Run drug discovery pipeline on a molecular library (CLI mode)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file with 'smiles' column"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (CSV or Parquet)"
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target key (e.g., HIV1_PROTEASE). If not specified, uses first available."
    )
    parser.add_argument(
        "--no-admet",
        action="store_true",
        help="Skip ADMET evaluation"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run lead optimization on top hits"
    )
    parser.add_argument(
        "--n-optimization-parents",
        type=int,
        default=20,
        help="Number of parents for optimization (default: 20)"
    )
    parser.add_argument(
        "--min-p-active",
        type=float,
        default=0.65,
        help="Minimum p_active threshold (default: 0.65)"
    )
    parser.add_argument(
        "--max-herg",
        type=float,
        default=0.7,
        help="Maximum hERG probability (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        input_csv=args.input,
        output_path=args.output,
        target_key=args.target,
        run_admet=not args.no_admet,
        run_optimization=args.optimize,
        n_optimization_parents=args.n_optimization_parents,
        min_p_active=args.min_p_active,
        max_herg=args.max_herg
    )


if __name__ == "__main__":
    main()

