"""
medchem_filters.py

Medicinal chemistry filters for generated molecules.
Includes PAINS filters, structural alerts, synthetic accessibility (SA Score), and basic drug-likeness checks.
"""

from typing import List, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


# PAINS (Pan Assay Interference Compounds) patterns
# Common problematic substructures that cause false positives
PAINS_SMARTS = [
    # Michael acceptors
    "[C,c]=[C,c]-[C,c]=[O,o]",
    "[C,c]=[C,c]-[C,c]=[N,n]",
    # Thiophenes with problematic substitutions
    "[S,s]1[C,c][C,c][C,c][C,c]1",
    # Quinones
    "[O,o]=[C,c]1[C,c][C,c][C,c][C,c][C,c]1=[O,o]",
    # Azo compounds
    "[N,n]=[N,n]",
    # Nitro groups on aromatics (can be mutagenic)
    "[c][N+](=O)[O-]",
    # Aldehydes (reactive)
    "[C,c][C,c]=[O,o]",
]


def passes_simple_medchem_filters(
    smiles: str,
    check_pains: bool = True,
    check_structural_alerts: bool = True,
    min_heavy_atoms: int = 5,
    max_heavy_atoms: int = 100,
    max_rings: int = 10,
    max_aromatic_rings: int = 6,
) -> bool:
    """
    Check if a molecule passes basic medicinal chemistry filters.
    
    Parameters
    ----------
    smiles : str
        SMILES string to check
    check_pains : bool
        Whether to check for PAINS patterns
    check_structural_alerts : bool
        Whether to check for structural alerts
    min_heavy_atoms : int
        Minimum number of heavy atoms
    max_heavy_atoms : int
        Maximum number of heavy atoms
    max_rings : int
        Maximum number of rings
    max_aromatic_rings : int
        Maximum number of aromatic rings
        
    Returns
    -------
    bool
        True if molecule passes all filters, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # Basic atom count checks
        num_heavy = mol.GetNumHeavyAtoms()
        if num_heavy < min_heavy_atoms or num_heavy > max_heavy_atoms:
            return False
        
        # Ring count checks
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        if num_rings > max_rings:
            return False
        
        # Aromatic ring count
        aromatic_rings = 0
        for ring in ring_info.AtomRings():
            is_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
            if is_aromatic:
                aromatic_rings += 1
        if aromatic_rings > max_aromatic_rings:
            return False
        
        # PAINS filter
        if check_pains:
            for smarts in PAINS_SMARTS:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is not None and mol.HasSubstructMatch(pattern):
                    return False
        
        # Structural alerts (reactive groups, toxicophores)
        if check_structural_alerts:
            # Epoxides (reactive)
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[C,c]1[C,c][O,o]1")):
                return False
            
            # Peroxides (unstable)
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[O,o][O,o]")):
                return False
            
            # Nitroso groups (mutagenic)
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[N,n]=[O,o]")):
                return False
            
            # Hydrazines (toxic)
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[N,n][N,n]")):
                return False
        
        return True
    
    except Exception:
        return False


def filter_molecules_by_medchem(
    smiles_list: List[str],
    check_pains: bool = True,
    check_structural_alerts: bool = True,
    **filter_kwargs
) -> List[str]:
    """
    Filter a list of SMILES strings by medicinal chemistry criteria.
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings to filter
    check_pains : bool
        Whether to check for PAINS patterns
    check_structural_alerts : bool
        Whether to check for structural alerts
    **filter_kwargs
        Additional arguments passed to passes_simple_medchem_filters
        
    Returns
    -------
    List[str]
        Filtered list of SMILES strings
    """
    filtered = []
    for smi in smiles_list:
        if passes_simple_medchem_filters(
            smi,
            check_pains=check_pains,
            check_structural_alerts=check_structural_alerts,
            **filter_kwargs
        ):
            filtered.append(smi)
    return filtered


def calculate_sa_score(smiles: str) -> float:
    """
    Calculate Synthetic Accessibility (SA) Score for a molecule.
    
    SA Score ranges from 1 (easy to synthesize) to 10 (difficult to synthesize).
    Based on fragment complexity, ring complexity, and molecular complexity.
    
    This is a simplified implementation. For production, consider using:
    - RDKit's SA Score (if available in your version)
    - External tools like ASKCOS
    
    Parameters
    ----------
    smiles : str
        SMILES string to evaluate
        
    Returns
    -------
    float
        SA Score (1-10, lower is better/easier to synthesize)
        Returns 10.0 if molecule is invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        
        # Fragment complexity (simplified)
        # More fragments = more complex
        num_fragments = len(Chem.GetMolFrags(mol))
        fragment_penalty = min(num_fragments * 0.5, 3.0)
        
        # Ring complexity
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        ring_penalty = min(num_rings * 0.3, 2.0)
        
        # Aromatic ring penalty (aromatic rings can be harder to synthesize)
        aromatic_rings = 0
        for ring in ring_info.AtomRings():
            is_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
            if is_aromatic:
                aromatic_rings += 1
        aromatic_penalty = min(aromatic_rings * 0.2, 1.5)
        
        # Stereocenter complexity
        num_stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        stereo_penalty = min(num_stereocenters * 0.3, 2.0)
        
        # Molecular weight penalty (larger molecules are harder)
        mw = Descriptors.MolWt(mol)
        mw_penalty = min((mw - 200) / 200 * 0.5, 1.5) if mw > 200 else 0.0
        
        # Base score (simple molecules start at 1)
        base_score = 1.0
        
        # Total score
        sa_score = base_score + fragment_penalty + ring_penalty + aromatic_penalty + stereo_penalty + mw_penalty
        
        # Clamp to 1-10 range
        return min(max(sa_score, 1.0), 10.0)
    
    except Exception:
        return 10.0


def passes_sa_score_threshold(smiles: str, max_sa_score: float = 6.0) -> bool:
    """
    Check if a molecule passes the SA Score threshold.
    
    Parameters
    ----------
    smiles : str
        SMILES string to check
    max_sa_score : float
        Maximum allowed SA Score (default: 6.0)
        Lower values = stricter (only very easy to synthesize)
        Recommended: 4.0-7.0
        
    Returns
    -------
    bool
        True if SA Score <= max_sa_score, False otherwise
    """
    sa_score = calculate_sa_score(smiles)
    return sa_score <= max_sa_score


def passes_simple_medchem_filters(
    smiles: str,
    check_pains: bool = True,
    check_structural_alerts: bool = True,
    check_sa_score: bool = False,
    max_sa_score: float = 6.0,
    min_heavy_atoms: int = 5,
    max_heavy_atoms: int = 100,
    max_rings: int = 10,
    max_aromatic_rings: int = 6,
) -> bool:
    """
    Check if a molecule passes basic medicinal chemistry filters.
    
    Parameters
    ----------
    smiles : str
        SMILES string to check
    check_pains : bool
        Whether to check for PAINS patterns
    check_structural_alerts : bool
        Whether to check for structural alerts
    check_sa_score : bool
        Whether to check synthetic accessibility score
    max_sa_score : float
        Maximum allowed SA Score (only used if check_sa_score=True)
    min_heavy_atoms : int
        Minimum number of heavy atoms
    max_heavy_atoms : int
        Maximum number of heavy atoms
    max_rings : int
        Maximum number of rings
    max_aromatic_rings : int
        Maximum number of aromatic rings
        
    Returns
    -------
    bool
        True if molecule passes all filters, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # Basic atom count checks
        num_heavy = mol.GetNumHeavyAtoms()
        if num_heavy < min_heavy_atoms or num_heavy > max_heavy_atoms:
            return False
        
        # Ring count checks
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        if num_rings > max_rings:
            return False
        
        # Aromatic ring count
        aromatic_rings = 0
        for ring in ring_info.AtomRings():
            is_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
            if is_aromatic:
                aromatic_rings += 1
        if aromatic_rings > max_aromatic_rings:
            return False
        
        # PAINS filter
        if check_pains:
            for smarts in PAINS_SMARTS:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is not None and mol.HasSubstructMatch(pattern):
                    return False
        
        # Structural alerts (reactive groups, toxicophores)
        if check_structural_alerts:
            # Epoxides (reactive)
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[C,c]1[C,c][O,o]1")):
                return False
            
            # Peroxides (unstable)
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[O,o][O,o]")):
                return False
            
            # Nitroso groups (mutagenic)
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[N,n]=[O,o]")):
                return False
            
            # Hydrazines (toxic)
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[N,n][N,n]")):
                return False
        
        # SA Score check
        if check_sa_score:
            if not passes_sa_score_threshold(smiles, max_sa_score):
                return False
        
        return True
    
    except Exception:
        return False


def filter_molecules_by_medchem(
    smiles_list: List[str],
    check_pains: bool = True,
    check_structural_alerts: bool = True,
    check_sa_score: bool = False,
    max_sa_score: float = 6.0,
    **filter_kwargs
) -> List[str]:
    """
    Filter a list of SMILES strings by medicinal chemistry criteria.
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings to filter
    check_pains : bool
        Whether to check for PAINS patterns
    check_structural_alerts : bool
        Whether to check for structural alerts
    check_sa_score : bool
        Whether to check synthetic accessibility score
    max_sa_score : float
        Maximum allowed SA Score (only used if check_sa_score=True)
    **filter_kwargs
        Additional arguments passed to passes_simple_medchem_filters
        
    Returns
    -------
    List[str]
        Filtered list of SMILES strings
    """
    filtered = []
    for smi in smiles_list:
        if passes_simple_medchem_filters(
            smi,
            check_pains=check_pains,
            check_structural_alerts=check_structural_alerts,
            check_sa_score=check_sa_score,
            max_sa_score=max_sa_score,
            **filter_kwargs
        ):
            filtered.append(smi)
    return filtered

