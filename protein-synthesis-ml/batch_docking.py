"""
batch_docking.py

Batch docking script for top N leads.
Takes a CSV of SMILES, runs docking, and adds scores to the dataframe.
"""

import argparse
import pandas as pd
import os
from pathlib import Path
from typing import List, Optional

try:
    from docking import dock_smiles, check_vina_available, check_obabel_available
    DOCKING_AVAILABLE = True
except ImportError:
    DOCKING_AVAILABLE = False
    print("Warning: docking module not available")


def batch_dock_smiles(
    smiles_list: List[str],
    receptor_path: str,
    config_path: Optional[str] = None,
    output_dir: str = "docking_results",
    n_jobs: int = 1
) -> List[Optional[float]]:
    """
    Run docking on a list of SMILES strings.
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings to dock
    receptor_path : str
        Path to receptor PDBQT file
    config_path : str, optional
        Path to Vina config file
    output_dir : str
        Directory to save docking results
    n_jobs : int
        Number of parallel jobs (currently 1, can be extended)
        
    Returns
    -------
    List[Optional[float]]
        List of binding affinities (kcal/mol), None if docking failed
    """
    if not DOCKING_AVAILABLE:
        raise ImportError("Docking module not available. Please install AutoDock Vina and Open Babel.")
    
    vina_available, _ = check_vina_available()
    obabel_available, _ = check_obabel_available()
    
    if not vina_available or not obabel_available:
        raise RuntimeError("Docking tools not available. Please install AutoDock Vina and Open Babel.")
    
    if not os.path.exists(receptor_path):
        raise FileNotFoundError(f"Receptor file not found: {receptor_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    affinities = []
    
    print(f"Docking {len(smiles_list)} molecules...")
    for i, smiles in enumerate(smiles_list):
        try:
            ligand_id = f"ligand_{i}"
            affinity, ligand_path, mol_output_dir = dock_smiles(
                smiles=smiles,
                receptor_path=receptor_path,
                config_path=config_path,
                ligand_id=ligand_id
            )
            affinities.append(affinity)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(smiles_list)}")
        except Exception as e:
            print(f"  Warning: Docking failed for molecule {i}: {e}")
            affinities.append(None)
    
    return affinities


def add_docking_scores_to_dataframe(
    df: pd.DataFrame,
    receptor_path: str,
    config_path: Optional[str] = None,
    smiles_col: str = "smiles",
    top_n: Optional[int] = None,
    output_dir: str = "docking_results"
) -> pd.DataFrame:
    """
    Add docking scores to a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with SMILES
    receptor_path : str
        Path to receptor PDBQT file
    config_path : str, optional
        Path to Vina config file
    smiles_col : str
        Name of SMILES column
    top_n : int, optional
        Only dock top N molecules (by composite_score if available)
    output_dir : str
        Directory to save docking results
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'docking_score' column
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in DataFrame")
    
    # Select molecules to dock
    if top_n is not None and "composite_score" in df.columns:
        molecules_to_dock = df.nlargest(top_n, "composite_score")
        print(f"Docking top {top_n} molecules by composite score...")
    else:
        molecules_to_dock = df
        print(f"Docking all {len(df)} molecules...")
    
    smiles_list = molecules_to_dock[smiles_col].astype(str).tolist()
    
    # Run docking
    affinities = batch_dock_smiles(
        smiles_list=smiles_list,
        receptor_path=receptor_path,
        config_path=config_path,
        output_dir=output_dir
    )
    
    # Add scores to DataFrame
    result_df = df.copy()
    result_df["docking_score"] = None
    
    # Map scores back to original indices
    for idx, affinity in zip(molecules_to_dock.index, affinities):
        result_df.loc[idx, "docking_score"] = affinity
    
    print(f"\n✅ Docking complete!")
    print(f"   Successful: {sum(1 for a in affinities if a is not None)}/{len(affinities)}")
    if any(a is not None for a in affinities):
        valid_affinities = [a for a in affinities if a is not None]
        print(f"   Best affinity: {min(valid_affinities):.2f} kcal/mol")
        print(f"   Average affinity: {sum(valid_affinities)/len(valid_affinities):.2f} kcal/mol")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Run batch docking on molecules from a CSV file"
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
        help="Output CSV file with docking scores"
    )
    parser.add_argument(
        "--receptor",
        type=str,
        required=True,
        help="Path to receptor PDBQT file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to Vina config file (optional)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Only dock top N molecules (by composite_score if available)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docking_results",
        help="Directory to save docking output files"
    )
    
    args = parser.parse_args()
    
    # Load input
    df = pd.read_csv(args.input)
    
    # Add docking scores
    result_df = add_docking_scores_to_dataframe(
        df=df,
        receptor_path=args.receptor,
        config_path=args.config,
        top_n=args.top_n,
        output_dir=args.output_dir
    )
    
    # Save results
    result_df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

