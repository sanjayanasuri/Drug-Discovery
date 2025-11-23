"""
clustering.py

Molecular similarity clustering and dimensionality reduction utilities.
Uses RDKit Butina clustering and UMAP for visualization.
"""

from typing import List, Tuple, Optional
import numpy as np
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
import umap

# Default Tanimoto distance threshold for Butina clustering
# Lower values = more clusters (stricter similarity requirement)
# Higher values = fewer clusters (looser similarity requirement)
DEFAULT_TANIMOTO_THRESHOLD = 0.25


def compute_fingerprints(smiles: List[str], radius: int = 2, n_bits: int = 2048) -> Tuple[np.ndarray, List[int]]:
    """
    Compute Morgan fingerprints for a list of SMILES strings.
    
    Reuses the existing featurization utilities to ensure consistency.
    
    Parameters
    ----------
    smiles : List[str]
        List of SMILES strings
    radius : int
        Morgan fingerprint radius (default: 2)
    n_bits : int
        Number of bits in fingerprint (default: 2048)
        
    Returns
    -------
    fps : np.ndarray
        Array of RDKit ExplicitBitVect fingerprints (for Tanimoto distance)
    valid_indices : List[int]
        Indices of successfully featurized SMILES
        
    Notes
    -----
    Returns RDKit ExplicitBitVect objects (not numpy arrays) for Butina clustering,
    which requires Tanimoto distance calculations.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    fps = []
    valid_indices = []
    
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(fp)
            valid_indices.append(i)
    
    return fps, valid_indices


def butina_clusters(
    fps: List, 
    threshold: float = DEFAULT_TANIMOTO_THRESHOLD
) -> List[int]:
    """
    Perform Butina clustering on molecular fingerprints using Tanimoto distance.
    
    Parameters
    ----------
    fps : List
        List of RDKit ExplicitBitVect fingerprints
    threshold : float
        Tanimoto distance threshold (default: 0.25)
        - Lower values = more clusters (stricter similarity)
        - Higher values = fewer clusters (looser similarity)
        - Typical range: 0.2-0.3
        
    Returns
    -------
    cluster_labels : List[int]
        Cluster assignment for each fingerprint (aligned with input order)
        Cluster IDs start at 0. Singleton clusters are assigned unique IDs.
        
    Notes
    -----
    Butina clustering is a leader-picker algorithm that groups molecules
    based on Tanimoto similarity. Molecules with Tanimoto distance <= threshold
    are grouped into the same cluster.
    """
    if len(fps) == 0:
        return []
    
    if len(fps) == 1:
        return [0]
    
    # Compute distance matrix (1 - Tanimoto similarity)
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    
    # Perform Butina clustering
    clusters = Butina.ClusterData(dists, nfps, threshold, isDistData=True)
    
    # Convert cluster list to label array
    cluster_labels = [-1] * nfps
    for cluster_id, cluster in enumerate(clusters):
        for mol_idx in cluster:
            cluster_labels[mol_idx] = cluster_id
    
    # Handle any unassigned molecules (shouldn't happen, but safety check)
    for i, label in enumerate(cluster_labels):
        if label == -1:
            cluster_labels[i] = max(cluster_labels) + 1 if cluster_labels else 0
    
    return cluster_labels


def umap_embedding(
    fps: List, 
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute UMAP embedding of molecular fingerprints for 2D visualization.
    
    Parameters
    ----------
    fps : List
        List of RDKit ExplicitBitVect fingerprints
    n_components : int
        Number of dimensions in embedding (default: 2)
    n_neighbors : int
        Number of neighbors for UMAP (default: 15)
        Lower values = more local structure, higher values = more global structure
    min_dist : float
        Minimum distance between points in embedding (default: 0.1)
        Lower values = tighter clusters, higher values = more spread out
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns
    -------
    embedding : np.ndarray
        UMAP embedding of shape (n_molecules, n_components)
        
    Notes
    -----
    Converts RDKit fingerprints to numpy arrays before UMAP computation.
    For very small datasets (< 4 samples), returns a simple PCA-like embedding.
    """
    if len(fps) == 0:
        return np.array([]).reshape(0, n_components)
    
    if len(fps) == 1:
        # UMAP needs at least 2 samples, return zero vector for single molecule
        return np.zeros((1, n_components))
    
    # Convert fingerprints to numpy arrays
    n_bits = len(fps[0])
    X = np.zeros((len(fps), n_bits), dtype=int)
    for i, fp in enumerate(fps):
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
    
    # UMAP requires at least 4 samples to work reliably
    # For smaller datasets, use simple PCA or random embedding
    if len(fps) < 4:
        # Use simple 2D projection based on first two principal components
        from sklearn.decomposition import PCA
        if len(fps) == 2:
            # For 2 samples, just place them at opposite ends
            embedding = np.array([[-1.0, 0.0], [1.0, 0.0]])
        else:  # 3 samples
            # Use PCA
            pca = PCA(n_components=min(n_components, len(fps) - 1))
            embedding = pca.fit_transform(X)
            # Pad if needed
            if embedding.shape[1] < n_components:
                padding = np.zeros((len(fps), n_components - embedding.shape[1]))
                embedding = np.hstack([embedding, padding])
        return embedding
    
    # Compute UMAP embedding for larger datasets
    # Adjust n_neighbors to be at most len(fps) - 1
    effective_n_neighbors = min(n_neighbors, len(fps) - 1, max(2, len(fps) // 4))
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric='euclidean'
    )
    
    embedding = reducer.fit_transform(X)
    
    return embedding


def cluster_molecules(
    smiles: List[str],
    tanimoto_threshold: float = DEFAULT_TANIMOTO_THRESHOLD,
    radius: int = 2,
    n_bits: int = 2048
) -> Tuple[List[int], np.ndarray, List[int]]:
    """
    Complete clustering pipeline: compute fingerprints, cluster, and embed.
    
    Parameters
    ----------
    smiles : List[str]
        List of SMILES strings
    tanimoto_threshold : float
        Tanimoto distance threshold for Butina clustering (default: 0.25)
    radius : int
        Morgan fingerprint radius (default: 2)
    n_bits : int
        Number of bits in fingerprint (default: 2048)
        
    Returns
    -------
    cluster_labels : List[int]
        Cluster assignment for each valid SMILES
    umap_coords : np.ndarray
        UMAP 2D embedding coordinates (n_valid, 2)
    valid_indices : List[int]
        Indices of successfully featurized SMILES in original list
        
    Notes
    -----
    This is a convenience function that combines fingerprint computation,
    clustering, and UMAP embedding in one call.
    """
    # Compute fingerprints
    fps, valid_indices = compute_fingerprints(smiles, radius=radius, n_bits=n_bits)
    
    if len(fps) == 0:
        return [], np.array([]).reshape(0, 2), []
    
    # Perform clustering
    cluster_labels = butina_clusters(fps, threshold=tanimoto_threshold)
    
    # Compute UMAP embedding
    umap_coords = umap_embedding(fps, n_components=2)
    
    return cluster_labels, umap_coords, valid_indices

