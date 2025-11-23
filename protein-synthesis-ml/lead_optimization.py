"""
lead_optimization.py

Lead optimization module with generative proposals.
Takes ADMET-evaluated hits, generates new molecules by mutating top leads,
and rescores them with existing models.
"""

from typing import List, Dict, Any, Optional, Callable
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, BRICS
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

from featurization import smiles_to_morgan_fp, smiles_to_matrix
from admet_loader import ADMET_TASKS
from pipeline import composite_score
from medchem_filters import passes_simple_medchem_filters, filter_molecules_by_medchem
from rdkit import DataStructs
from rdkit.Chem import AllChem




def random_mutate_smiles(parent_smiles: str, n_trials: int = 10) -> List[str]:
    """
    Generate mutated variants of a parent SMILES using simple RDKit transformations.
    
    Uses a combination of:
    - Adding/removing methyl groups on aromatic rings
    - Halogen replacement (F/Cl/Br)
    - Simple BRICS fragmentation & recombination (if available)
    
    Parameters
    ----------
    parent_smiles : str
        Parent SMILES string to mutate
    n_trials : int
        Number of mutation attempts to make
        
    Returns
    -------
    List[str]
        List of valid, unique mutated SMILES (excluding the parent)
    """
    mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None:
        return []
    
    candidates = []
    seen = {parent_smiles}  # Track seen SMILES to avoid duplicates
    
    for _ in range(n_trials):
        try:
            # Strategy 1: Add/remove methyl on aromatic ring
            if np.random.random() < 0.4:
                # Find aromatic atoms
                aromatic_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsAromatic()]
                if len(aromatic_atoms) > 0:
                    # Try adding methyl
                    if np.random.random() < 0.5:
                        # Add methyl to random aromatic atom
                        target_idx = np.random.choice(aromatic_atoms)
                        new_mol = Chem.RWMol(mol)
                        new_mol.AddAtom(Chem.Atom("C"))
                        new_mol.AddBond(target_idx, new_mol.GetNumAtoms() - 1, Chem.BondType.SINGLE)
                        try:
                            new_mol = new_mol.GetMol()
                            Chem.SanitizeMol(new_mol)
                            new_smi = Chem.MolToSmiles(new_mol)
                            if new_smi not in seen:
                                candidates.append(new_smi)
                                seen.add(new_smi)
                        except:
                            pass
                    else:
                        # Try removing a methyl group
                        methyl_candidates = [
                            atom.GetIdx() for atom in mol.GetAtoms()
                            if atom.GetSymbol() == "C" and atom.GetDegree() == 1
                        ]
                        if len(methyl_candidates) > 0:
                            target_idx = np.random.choice(methyl_candidates)
                            new_mol = Chem.RWMol(mol)
                            new_mol.RemoveAtom(target_idx)
                            try:
                                new_mol = new_mol.GetMol()
                                Chem.SanitizeMol(new_mol)
                                new_smi = Chem.MolToSmiles(new_mol)
                                if new_smi not in seen:
                                    candidates.append(new_smi)
                                    seen.add(new_smi)
                            except:
                                pass
            
            # Strategy 2: Halogen replacement
            elif np.random.random() < 0.3:
                halogens = ["F", "Cl", "Br"]
                halogen_atoms = [
                    atom.GetIdx() for atom in mol.GetAtoms()
                    if atom.GetSymbol() in halogens
                ]
                if len(halogen_atoms) > 0:
                    target_idx = np.random.choice(halogen_atoms)
                    old_symbol = mol.GetAtomWithIdx(target_idx).GetSymbol()
                    alternative_halogens = [h for h in halogens if h != old_symbol]
                    if len(alternative_halogens) > 0:
                        new_symbol = np.random.choice(alternative_halogens)
                        
                        new_mol = Chem.RWMol(mol)
                        new_mol.GetAtomWithIdx(target_idx).SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new_symbol))
                        try:
                            new_mol = new_mol.GetMol()
                            Chem.SanitizeMol(new_mol)
                            new_smi = Chem.MolToSmiles(new_mol)
                            if new_smi not in seen:
                                candidates.append(new_smi)
                                seen.add(new_smi)
                        except:
                            pass
            
            # Strategy 3: Simple BRICS fragmentation & recombination
            else:
                try:
                    # Fragment molecule
                    frags = list(BRICS.BRICSDecompose(mol))
                    if len(frags) >= 2:
                        # Recombine fragments randomly
                        frag1 = np.random.choice(frags)
                        frag2 = np.random.choice([f for f in frags if f != frag1])
                        
                        # Try to combine fragments
                        frag1_mol = Chem.MolFromSmiles(frag1)
                        frag2_mol = Chem.MolFromSmiles(frag2)
                        if frag1_mol is not None and frag2_mol is not None:
                            combined = Chem.CombineMols(frag1_mol, frag2_mol)
                            # Add a bond between fragments (simplified)
                            try:
                                new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(combined))
                                if new_mol is not None:
                                    Chem.SanitizeMol(new_mol)
                                    new_smi = Chem.MolToSmiles(new_mol)
                                    if new_smi not in seen:
                                        candidates.append(new_smi)
                                        seen.add(new_smi)
                            except:
                                pass
                except:
                    pass
            
            # Strategy 4: Simple atom replacement (fallback)
            if len(candidates) == 0 and np.random.random() < 0.1:
                # Replace a non-aromatic carbon with nitrogen
                replaceable = [
                    atom.GetIdx() for atom in mol.GetAtoms()
                    if atom.GetSymbol() == "C" and not atom.GetIsAromatic() and atom.GetDegree() >= 2
                ]
                if len(replaceable) > 0:
                    target_idx = np.random.choice(replaceable)
                    new_mol = Chem.RWMol(mol)
                    new_mol.GetAtomWithIdx(target_idx).SetAtomicNum(7)  # N
                    try:
                        new_mol = new_mol.GetMol()
                        Chem.SanitizeMol(new_mol)
                        new_smi = Chem.MolToSmiles(new_mol)
                        if new_smi not in seen:
                            candidates.append(new_smi)
                            seen.add(new_smi)
                    except:
                        pass
        
        except Exception:
            # Skip failed mutations
            continue
    
    # Filter out invalid SMILES and ensure uniqueness
    valid_candidates = []
    for smi in candidates:
        if smi and smi != parent_smiles:
            mol_test = Chem.MolFromSmiles(smi)
            if mol_test is not None:
                valid_candidates.append(smi)
    
    # Remove duplicates while preserving order
    seen_valid = set()
    unique_candidates = []
    for smi in valid_candidates:
        if smi not in seen_valid:
            seen_valid.add(smi)
            unique_candidates.append(smi)
    
    return unique_candidates


def passes_lipinski(smiles: str) -> bool:
    """
    Check if a molecule passes Lipinski's Rule of Five.
    
    Criteria:
    - Molecular weight < 500 Da
    - logP <= 5
    - H-bond donors <= 5
    - H-bond acceptors <= 10
    - Rotatable bonds <= 10
    
    Parameters
    ----------
    smiles : str
        SMILES string to evaluate
        
    Returns
    -------
    bool
        True if molecule passes all Lipinski criteria, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # Molecular weight
        mw = Descriptors.MolWt(mol)
        if mw >= 500:
            return False
        
        # logP
        logp = Descriptors.MolLogP(mol)
        if logp > 5:
            return False
        
        # H-bond donors
        hbd = Descriptors.NumHDonors(mol)
        if hbd > 5:
            return False
        
        # H-bond acceptors
        hba = Descriptors.NumHAcceptors(mol)
        if hba > 10:
            return False
        
        # Rotatable bonds
        rot_bonds = CalcNumRotatableBonds(mol)
        if rot_bonds > 10:
            return False
        
        return True
    
    except Exception:
        return False


def score_molecules(
    smiles_list: List[str],
    hiv_model,
    admet_models: Dict[str, Any],
    composite_fn: Callable = composite_score,
) -> pd.DataFrame:
    """
    Score a list of molecules using HIV-1 and ADMET models.
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings to score
    hiv_model
        Trained HIV-1 protease activity model
    admet_models : Dict[str, Any]
        Dictionary of ADMET models (keyed by task_key from ADMET_TASKS)
    composite_fn : Callable
        Function to compute composite score (default: composite_score from pipeline)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - smiles
        - hiv1_p_active
        - {task_key}_prob or {task_key}_value for each ADMET task
        - composite_score
    """
    if len(smiles_list) == 0:
        return pd.DataFrame(columns=["smiles", "hiv1_p_active", "composite_score"])
    
    # Featurize SMILES and get valid indices
    X, valid_indices = smiles_to_matrix(smiles_list)
    
    if len(X) == 0:
        return pd.DataFrame(columns=["smiles", "hiv1_p_active", "composite_score"])
    
    # Get valid SMILES using the indices returned by smiles_to_matrix
    valid_smiles = [smiles_list[i] for i in valid_indices]
    
    if len(valid_smiles) == 0:
        return pd.DataFrame(columns=["smiles", "hiv1_p_active", "composite_score"])
    
    # Predict HIV-1 activity
    if hiv_model is not None and hasattr(hiv_model, "predict_proba"):
        hiv1_p_active = hiv_model.predict_proba(X)[:, 1]
    else:
        hiv1_p_active = np.zeros(len(X))
    
    # Initialize results DataFrame
    results = pd.DataFrame({
        "smiles": valid_smiles,
        "hiv1_p_active": hiv1_p_active
    })
    
    # Predict ADMET properties
    admet_cols = {}
    for task_key in ADMET_TASKS.keys():
        if task_key not in admet_models or admet_models[task_key] is None:
            continue
        
        model = admet_models[task_key]
        task_config = ADMET_TASKS[task_key]
        task_type = task_config["type"]
        
        if task_type == "classification":
            if hasattr(model, "predict_proba"):
                predictions = model.predict_proba(X)[:, 1]
            else:
                predictions = model.predict(X).astype(float)
            admet_cols[f"{task_key}_prob"] = predictions
        else:  # regression
            predictions = model.predict(X)
            admet_cols[f"{task_key}_value"] = predictions
    
    # Add ADMET columns
    for col, values in admet_cols.items():
        results[col] = values
    
    # Compute composite scores
    composite_scores = []
    for _, row in results.iterrows():
        admet_outputs = {}
        for task_key in ADMET_TASKS.keys():
            prob_col = f"{task_key}_prob"
            value_col = f"{task_key}_value"
            
            if prob_col in results.columns:
                admet_outputs[task_key] = {"prob": row[prob_col]}
            elif value_col in results.columns:
                admet_outputs[task_key] = {"value": row[value_col]}
        
        score = composite_fn(row["hiv1_p_active"], admet_outputs)
        composite_scores.append(score)
    
    results["composite_score"] = composite_scores
    
    return results


def compute_similarity_to_parent(parent_smiles: str, child_smiles: str) -> float:
    """
    Compute Tanimoto similarity between parent and child molecule.
    
    Parameters
    ----------
    parent_smiles : str
        Parent SMILES string
    child_smiles : str
        Child SMILES string
        
    Returns
    -------
    float
        Tanimoto similarity (0-1), or 0.0 if either molecule is invalid
    """
    try:
        parent_mol = Chem.MolFromSmiles(parent_smiles)
        child_mol = Chem.MolFromSmiles(child_smiles)
        
        if parent_mol is None or child_mol is None:
            return 0.0
        
        parent_fp = AllChem.GetMorganFingerprintAsBitVect(parent_mol, 2, nBits=2048)
        child_fp = AllChem.GetMorganFingerprintAsBitVect(child_mol, 2, nBits=2048)
        
        return DataStructs.TanimotoSimilarity(parent_fp, child_fp)
    except Exception:
        return 0.0


def optimize_leads(
    hits_df: pd.DataFrame,
    hiv_model,
    admet_models: Dict[str, Any],
    composite_fn: Callable = composite_score,
    n_parents: int = 20,
    n_children_per_parent: int = 20,
    min_hiv1_p_active: float = 0.7,
    max_hERG_prob: float = 0.7,
    enforce_lipinski: bool = True,
    enforce_medchem_filters: bool = True,
    check_sa_score: bool = False,
    max_sa_score: float = 6.0,
    min_similarity: float = 0.4,
    max_similarity: float = 0.9,
) -> pd.DataFrame:
    """
    Generate optimized lead candidates by mutating top hits.
    
    Parameters
    ----------
    hits_df : pd.DataFrame
        DataFrame of ADMET-evaluated hits with columns:
        - smiles
        - hiv1_p_active
        - composite_score
        - ADMET columns ({task_key}_prob or {task_key}_value)
        - optional: cluster_id
    hiv_model
        Trained HIV-1 protease activity model
    admet_models : Dict[str, Any]
        Dictionary of ADMET models
    composite_fn : Callable
        Function to compute composite score
    n_parents : int
        Number of top parent molecules to use (default: 20)
    n_children_per_parent : int
        Target number of children per parent (default: 20)
    min_hiv1_p_active : float
        Minimum HIV-1 p_active threshold (default: 0.7)
    max_hERG_prob : float
        Maximum hERG probability threshold (default: 0.7)
    enforce_lipinski : bool
        Whether to enforce Lipinski's Rule of Five (default: True)
        
    Returns
    -------
    pd.DataFrame
        DataFrame of optimized leads with columns:
        - smiles
        - hiv1_p_active
        - all ADMET columns
        - composite_score
        - parent_smiles
        - parent_index (if available)
    """
    if len(hits_df) == 0:
        return pd.DataFrame()
    
    # Sort by composite_score and select top parents
    sorted_hits = hits_df.sort_values("composite_score", ascending=False)
    n_parents_actual = min(n_parents, len(sorted_hits))
    parents_df = sorted_hits.head(n_parents_actual).copy()
    
    # Generate candidate SMILES from parents
    all_candidates = []
    parent_mapping = {}  # Map candidate SMILES to parent info
    
    for idx, (parent_idx, parent_row) in enumerate(parents_df.iterrows()):
        parent_smiles = parent_row["smiles"]
        
        # Generate mutations
        candidates = random_mutate_smiles(parent_smiles, n_trials=n_children_per_parent * 2)
        
        # Limit to n_children_per_parent
        candidates = candidates[:n_children_per_parent]
        
        # Store parent info for each candidate
        for cand_smi in candidates:
            all_candidates.append(cand_smi)
            parent_mapping[cand_smi] = {
                "parent_smiles": parent_smiles,
                "parent_index": parent_idx if isinstance(parent_idx, (int, np.integer)) else idx,
                "parent_cluster_id": parent_row.get("cluster_id", None)
            }
    
    if len(all_candidates) == 0:
        return pd.DataFrame()
    
    # Remove duplicates across parents
    unique_candidates = list(dict.fromkeys(all_candidates))  # Preserves order
    
    # Remove candidates that already exist in hits_df
    existing_smiles = set(hits_df["smiles"].astype(str))
    unique_candidates = [smi for smi in unique_candidates if smi not in existing_smiles]
    
    if len(unique_candidates) == 0:
        return pd.DataFrame()
    
    # Apply Lipinski filter if requested
    if enforce_lipinski:
        unique_candidates = [smi for smi in unique_candidates if passes_lipinski(smi)]
    
    # Apply medicinal chemistry filters if requested
    if enforce_medchem_filters:
        unique_candidates = filter_molecules_by_medchem(
            unique_candidates,
            check_sa_score=check_sa_score,
            max_sa_score=max_sa_score
        )
    
    # Apply similarity constraints (keep only candidates within similarity window)
    if min_similarity > 0.0 or max_similarity < 1.0:
        filtered_by_similarity = []
        for cand_smi in unique_candidates:
            parent_smiles = parent_mapping.get(cand_smi, {}).get("parent_smiles")
            if parent_smiles:
                similarity = compute_similarity_to_parent(parent_smiles, cand_smi)
                if min_similarity <= similarity <= max_similarity:
                    filtered_by_similarity.append(cand_smi)
                    # Update parent mapping with similarity
                    if cand_smi in parent_mapping:
                        parent_mapping[cand_smi]["similarity_to_parent"] = similarity
        unique_candidates = filtered_by_similarity
    
    if len(unique_candidates) == 0:
        return pd.DataFrame()
    
    # Score all candidates
    scored_df = score_molecules(unique_candidates, hiv_model, admet_models, composite_fn)
    
    if len(scored_df) == 0:
        return pd.DataFrame()
    
    # Add parent information
    scored_df["parent_smiles"] = scored_df["smiles"].map(lambda x: parent_mapping.get(x, {}).get("parent_smiles", ""))
    scored_df["parent_index"] = scored_df["smiles"].map(lambda x: parent_mapping.get(x, {}).get("parent_index", -1))
    
    # Add similarity to parent if computed
    if any("similarity_to_parent" in parent_mapping.get(smi, {}) for smi in scored_df["smiles"]):
        scored_df["similarity_to_parent"] = scored_df["smiles"].map(
            lambda x: parent_mapping.get(x, {}).get("similarity_to_parent", None)
        )
    
    if "cluster_id" in hits_df.columns:
        scored_df["parent_cluster_id"] = scored_df["smiles"].map(
            lambda x: parent_mapping.get(x, {}).get("parent_cluster_id", None)
        )
    
    # Apply soft constraints
    filtered_df = scored_df[
        (scored_df["hiv1_p_active"] >= min_hiv1_p_active)
    ].copy()
    
    if "hERG_prob" in scored_df.columns and "hERG" in admet_models:
        filtered_df = filtered_df[filtered_df["hERG_prob"] <= max_hERG_prob]
    
    # Sort by composite_score descending
    filtered_df = filtered_df.sort_values("composite_score", ascending=False)
    
    return filtered_df


def run_docking_stub(smiles_list: List[str]) -> None:
    """
    Placeholder for future docking integration.
    
    In the future this will:
    - convert SMILES to 3D
    - write PDBQT files
    - call AutoDock Vina or GNINA
    - attach docking scores back to the lead optimization dataframe.
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings to dock
        
    Raises
    ------
    NotImplementedError
        Always raises, as this is a placeholder
    """
    raise NotImplementedError("Docking integration not yet implemented.")

