"""
sar.py

Structure-Activity Relationship (SAR) optimization.
Generates molecular analogs and scores them using existing models.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from featurization import smiles_to_morgan_fp, smiles_to_matrix
from pipeline import registry, composite_score
from admet_loader import ADMET_TASKS


# Common R-group transformations for analog generation
R_GROUP_TRANSFORMATIONS = [
    # Methyl/ethyl variations
    ("[CH3]", "[CH2CH3]"),  # Methyl -> Ethyl
    ("[CH3]", "[CH2CH2CH3]"),  # Methyl -> Propyl
    ("[CH3]", "[CH(CH3)2]"),  # Methyl -> Isopropyl
    
    # Halogen swaps
    ("[Cl]", "[F]"),  # Chloro -> Fluoro
    ("[Cl]", "[Br]"),  # Chloro -> Bromo
    ("[Br]", "[Cl]"),  # Bromo -> Chloro
    ("[F]", "[Cl]"),  # Fluoro -> Chloro
    
    # Hydroxyl variations
    ("[OH]", "[OCH3]"),  # Hydroxyl -> Methoxy
    ("[OH]", "[NH2]"),  # Hydroxyl -> Amino
    ("[OH]", "[SH]"),  # Hydroxyl -> Thiol
    
    # Aromatic substitutions
    ("c1ccccc1", "c1ccc(F)cc1"),  # Phenyl -> Fluorophenyl
    ("c1ccccc1", "c1ccc(Cl)cc1"),  # Phenyl -> Chlorophenyl
    ("c1ccccc1", "c1ccc(O)cc1"),  # Phenyl -> Phenoxyphenyl
    
    # Common functional groups
    ("[NH2]", "[N(CH3)2]"),  # Amino -> Dimethylamino
    ("[NH2]", "[NHCH3]"),  # Amino -> Methylamino
    ("[C]=O", "[C]O"),  # Carbonyl -> Hydroxyl (simplified)
]


def generate_analogs(
    smiles: str,
    max_analogs: int = 20,
    transformations: Optional[List[tuple]] = None
) -> List[str]:
    """
    Generate structural analogs of a molecule using simple R-group transformations.
    
    Parameters
    ----------
    smiles : str
        Base SMILES string
    max_analogs : int
        Maximum number of analogs to generate (default: 20)
    transformations : Optional[List[tuple]]
        List of (pattern, replacement) tuples for substructure replacement.
        If None, uses default R_GROUP_TRANSFORMATIONS.
        
    Returns
    -------
    List[str]
        List of unique valid SMILES strings for analogs
        
    Notes
    -----
    Uses RDKit's ReplaceSubstructs to perform transformations.
    Filters out invalid SMILES and duplicates.
    """
    if transformations is None:
        transformations = R_GROUP_TRANSFORMATIONS
    
    # Parse base molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    
    analogs = set()
    analogs.add(smiles)  # Include original
    
    # Try each transformation
    for pattern_smi, replacement_smi in transformations:
        try:
            pattern = Chem.MolFromSmarts(pattern_smi)
            replacement = Chem.MolFromSmiles(replacement_smi)
            
            if pattern is None or replacement is None:
                continue
            
            # Replace all occurrences
            new_mols = Chem.ReplaceSubstructs(mol, pattern, replacement, replaceAll=True)
            
            for new_mol in new_mols:
                if new_mol is not None:
                    # Sanitize and get canonical SMILES
                    try:
                        Chem.SanitizeMol(new_mol)
                        analog_smi = Chem.MolToSmiles(new_mol, canonical=True)
                        if analog_smi and len(analog_smi) > 0:
                            analogs.add(analog_smi)
                    except:
                        continue
        except:
            continue
        
        # Stop if we have enough analogs
        if len(analogs) > max_analogs:
            break
    
    # Convert to list and remove original
    analog_list = list(analogs)
    if smiles in analog_list:
        analog_list.remove(smiles)
    
    # Limit to max_analogs
    return analog_list[:max_analogs]


def score_analogs(
    smiles_list: List[str],
    hiv_model,
    admet_models: Dict[str, Any],
    include_parent: bool = True,
    parent_smiles: Optional[str] = None
) -> pd.DataFrame:
    """
    Score a list of analog SMILES using HIV-1 and ADMET models.
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings (analogs)
    hiv_model
        Trained HIV-1 protease activity model
    admet_models : Dict[str, Any]
        Dictionary of ADMET models (keyed by task name)
    include_parent : bool
        Whether to include parent molecule in results (default: True)
    parent_smiles : Optional[str]
        SMILES of parent molecule (for comparison)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - smiles
        - hiv1_p_active
        - composite_score
        - ADMET columns (task_prob or task_value)
        - is_parent (bool)
        - better_than_parent (bool, if parent_smiles provided)
    """
    if len(smiles_list) == 0:
        return pd.DataFrame()
    
    # Featurize all SMILES
    X, valid_idx = smiles_to_matrix(smiles_list, radius=2, n_bits=2048)
    valid_smiles = [smiles_list[i] for i in valid_idx]
    
    if len(valid_smiles) == 0:
        return pd.DataFrame()
    
    # Initialize results
    results = {
        "smiles": valid_smiles,
        "hiv1_p_active": np.zeros(len(valid_smiles)),
        "composite_score": np.zeros(len(valid_smiles)),
    }
    
    # Predict HIV-1 activity
    if hiv_model is not None:
        if hasattr(hiv_model, "predict_proba"):
            p_active = hiv_model.predict_proba(X)[:, 1]
            results["hiv1_p_active"] = p_active
        else:
            results["hiv1_p_active"] = hiv_model.predict(X).astype(float)
    
    # Predict ADMET properties
    admet_outputs_list = []
    for task_key in ADMET_TASKS.keys():
        model = admet_models.get(task_key)
        if model is None:
            continue
        
        task_config = ADMET_TASKS[task_key]
        task_type = task_config["type"]
        
        if task_type == "classification":
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]
                results[f"{task_key}_prob"] = probs
            else:
                preds = model.predict(X).astype(float)
                results[f"{task_key}_pred"] = preds
        else:  # regression
            values = model.predict(X)
            results[f"{task_key}_value"] = values
    
    # Compute composite scores
    for i, smiles in enumerate(valid_smiles):
        admet_dict = {}
        for task_key in ADMET_TASKS.keys():
            prob_col = f"{task_key}_prob"
            value_col = f"{task_key}_value"
            pred_col = f"{task_key}_pred"
            
            if prob_col in results:
                admet_dict[task_key] = {"prob": results[prob_col][i]}
            elif value_col in results:
                admet_dict[task_key] = {"value": results[value_col][i]}
            elif pred_col in results:
                admet_dict[task_key] = {"prob": float(results[pred_col][i])}
        
        results["composite_score"][i] = composite_score(
            results["hiv1_p_active"][i],
            admet_dict
        )
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Mark parent molecule
    if parent_smiles:
        df["is_parent"] = df["smiles"] == parent_smiles
    else:
        df["is_parent"] = False
    
    # Compare to parent
    if parent_smiles and parent_smiles in valid_smiles:
        parent_idx = valid_smiles.index(parent_smiles)
        parent_composite = results["composite_score"][parent_idx]
        parent_p_active = results["hiv1_p_active"][parent_idx]
        
        df["better_than_parent"] = (
            (df["composite_score"] > parent_composite) |
            ((df["composite_score"] == parent_composite) & (df["hiv1_p_active"] > parent_p_active))
        )
    else:
        df["better_than_parent"] = False
    
    # Sort by composite score (descending)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    
    return df


def generate_and_score_analogs(
    parent_smiles: str,
    max_analogs: int = 20,
    hiv_model=None,
    admet_models: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Complete SAR workflow: generate analogs and score them.
    
    Parameters
    ----------
    parent_smiles : str
        SMILES of parent molecule
    max_analogs : int
        Maximum number of analogs to generate
    hiv_model
        HIV-1 protease model (if None, loads from registry)
    admet_models : Optional[Dict[str, Any]]
        ADMET models dict (if None, loads from registry)
        
    Returns
    -------
    pd.DataFrame
        Scored analogs DataFrame
    """
    # Load models if not provided
    if hiv_model is None:
        registry.load()
        hiv_model = registry.get("HIV_protease")
    
    if admet_models is None:
        if not registry._loaded:
            registry.load()
        admet_models = {}
        for task_key in ADMET_TASKS.keys():
            model = registry.get(task_key)
            if model is not None:
                admet_models[task_key] = model
    
    # Generate analogs
    analog_smiles = generate_analogs(parent_smiles, max_analogs=max_analogs)
    
    if len(analog_smiles) == 0:
        return pd.DataFrame(columns=["smiles", "hiv1_p_active", "composite_score"])
    
    # Score analogs
    results_df = score_analogs(
        analog_smiles,
        hiv_model=hiv_model,
        admet_models=admet_models,
        include_parent=True,
        parent_smiles=parent_smiles
    )
    
    return results_df

