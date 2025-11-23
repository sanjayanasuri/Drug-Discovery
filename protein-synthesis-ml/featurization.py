"""
featurization.py

RDKit-based helpers to convert SMILES to Morgan fingerprints.
"""

from typing import Optional, Sequence, Tuple
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

# Silence RDKit logs
RDLogger.DisableLog("rdApp.*")


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """
    Convert a single SMILES string to a Morgan fingerprint bit vector.
    
    Parameters
    ----------
    smiles : str
        SMILES string
    radius : int
        Morgan fingerprint radius (default: 2)
    n_bits : int
        Number of bits in fingerprint (default: 2048)
        
    Returns
    -------
    np.ndarray | None
        Morgan fingerprint as numpy array, or None if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_matrix(
    smiles_list: Sequence[str], 
    radius: int = 2, 
    n_bits: int = 2048
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a sequence of SMILES strings into a matrix of fingerprints.
    
    Parameters
    ----------
    smiles_list : Sequence[str]
        List, pandas Series, or other sequence of SMILES strings
    radius : int
        Morgan fingerprint radius (default: 2)
    n_bits : int
        Number of bits in fingerprint (default: 2048)
        
    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_valid_mols, n_bits)
    idx : np.ndarray
        Array of indices indicating which SMILES were successfully converted
    """
    fps = []
    idx = []

    for i, smi in enumerate(smiles_list):
        fp = smiles_to_morgan_fp(smi, radius=radius, n_bits=n_bits)
        if fp is not None:
            fps.append(fp)
            idx.append(i)

    X = np.array(fps, dtype=int)
    idx = np.array(idx, dtype=int)
    return X, idx
