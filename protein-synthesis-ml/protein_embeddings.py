"""
protein_embeddings.py

Protein sequence embeddings using ESM2 or ProtBERT for universal binding prediction.
Enables protein-aware ML models that work across different targets.
"""

import os
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import warnings

# Try to import transformers for ESM2/ProtBERT
try:
    import torch
    from transformers import EsmModel, EsmTokenizer, AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None

# Fallback: simple one-hot encoding
def one_hot_encode_sequence(sequence: str, max_length: Optional[int] = None) -> np.ndarray:
    """
    Simple one-hot encoding of protein sequence (fallback).
    
    Parameters
    ----------
    sequence : str
        Protein sequence (amino acid letters)
    max_length : int, optional
        Maximum sequence length (pads or truncates)
        
    Returns
    -------
    np.ndarray
        One-hot encoded sequence
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    
    if max_length:
        sequence = sequence[:max_length]
    
    encoding = np.zeros((len(sequence), len(amino_acids)))
    for i, aa in enumerate(sequence):
        if aa in aa_to_idx:
            encoding[i, aa_to_idx[aa]] = 1.0
    
    return encoding


def get_esm2_embedding(
    sequence: str,
    model_name: str = "facebook/esm2_t6_8M_UR50D",
    device: Optional[str] = None,
    pool_method: str = "mean"
) -> Optional[np.ndarray]:
    """
    Get ESM2 embedding for a protein sequence.
    
    ESM2 (Evolutionary Scale Modeling) is a state-of-the-art protein language model.
    
    Parameters
    ----------
    sequence : str
        Protein sequence (amino acid letters)
    model_name : str
        ESM2 model name (default: "facebook/esm2_t6_8M_UR50D")
        Options:
        - "facebook/esm2_t6_8M_UR50D" (small, fast)
        - "facebook/esm2_t12_35M_UR50D" (medium)
        - "facebook/esm2_t33_650M_UR50D" (large, slow)
    device : str, optional
        Device to use ("cuda", "cpu", or None for auto-detect)
    pool_method : str
        How to pool sequence embeddings ("mean", "cls", "max")
        
    Returns
    -------
    np.ndarray or None
        Protein embedding vector, or None if model unavailable
    """
    if not TRANSFORMERS_AVAILABLE:
        warnings.warn("Transformers library not available. Install with: pip install transformers torch")
        return None
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load model and tokenizer
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        
        # Tokenize sequence
        inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state  # Shape: (1, seq_len, hidden_dim)
        
        # Pool embeddings
        if pool_method == "mean":
            embedding = embeddings.mean(dim=1).cpu().numpy()[0]
        elif pool_method == "cls":
            embedding = embeddings[:, 0, :].cpu().numpy()[0]  # CLS token
        elif pool_method == "max":
            embedding = embeddings.max(dim=1)[0].cpu().numpy()[0]
        else:
            embedding = embeddings.mean(dim=1).cpu().numpy()[0]
        
        return embedding
    
    except Exception as e:
        warnings.warn(f"Failed to get ESM2 embedding: {e}")
        return None


def get_protbert_embedding(
    sequence: str,
    model_name: str = "Rostlab/prot_bert",
    device: Optional[str] = None,
    pool_method: str = "mean"
) -> Optional[np.ndarray]:
    """
    Get ProtBERT embedding for a protein sequence.
    
    ProtBERT is a BERT model trained on protein sequences.
    
    Parameters
    ----------
    sequence : str
        Protein sequence
    model_name : str
        ProtBERT model name (default: "Rostlab/prot_bert")
    device : str, optional
        Device to use
    pool_method : str
        Pooling method ("mean", "cls", "max")
        
    Returns
    -------
    np.ndarray or None
        Protein embedding vector
    """
    if not TRANSFORMERS_AVAILABLE:
        warnings.warn("Transformers library not available")
        return None
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        
        # Tokenize
        inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
        
        # Pool
        if pool_method == "mean":
            embedding = embeddings.mean(dim=1).cpu().numpy()[0]
        elif pool_method == "cls":
            embedding = embeddings[:, 0, :].cpu().numpy()[0]
        elif pool_method == "max":
            embedding = embeddings.max(dim=1)[0].cpu().numpy()[0]
        else:
            embedding = embeddings.mean(dim=1).cpu().numpy()[0]
        
        return embedding
    
    except Exception as e:
        warnings.warn(f"Failed to get ProtBERT embedding: {e}")
        return None


def get_protein_embedding(
    sequence: str,
    method: str = "esm2",
    **kwargs
) -> Optional[np.ndarray]:
    """
    Get protein embedding using specified method.
    
    Parameters
    ----------
    sequence : str
        Protein sequence
    method : str
        Embedding method: "esm2", "protbert", or "onehot" (fallback)
    **kwargs
        Additional arguments passed to embedding function
        
    Returns
    -------
    np.ndarray or None
        Protein embedding vector
    """
    if method == "esm2":
        return get_esm2_embedding(sequence, **kwargs)
    elif method == "protbert":
        return get_protbert_embedding(sequence, **kwargs)
    elif method == "onehot":
        return one_hot_encode_sequence(sequence).flatten()
    else:
        warnings.warn(f"Unknown embedding method: {method}. Using one-hot encoding.")
        return one_hot_encode_sequence(sequence).flatten()


def check_protein_embeddings_available() -> Tuple[bool, str]:
    """
    Check if protein embedding models are available.
    
    Returns
    -------
    Tuple[bool, str]
        (is_available, status_message)
    """
    if not TRANSFORMERS_AVAILABLE:
        return False, "Transformers library not installed. Install with: pip install transformers torch"
    
    if torch is None:
        return False, "PyTorch not available"
    
    try:
        # Try to load a small ESM2 model
        from transformers import EsmModel, EsmTokenizer
        return True, f"Protein embeddings available (ESM2/ProtBERT). Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
    except Exception as e:
        return False, f"Protein embeddings not available: {e}"


def get_target_sequence_from_chembl(chembl_id: str) -> Optional[str]:
    """
    Get protein sequence from ChEMBL database.
    
    Parameters
    ----------
    chembl_id : str
        ChEMBL target ID (e.g., "CHEMBL243")
        
    Returns
    -------
    str or None
        Protein sequence, or None if not found
    """
    try:
        from chembl_webresource_client.new_client import new_client
        
        target = new_client.target
        res = target.filter(target_chembl_id=chembl_id)
        
        if len(res) == 0:
            return None
        
        # Try to get sequence
        target_data = res[0]
        
        # ChEMBL may have sequence in different fields
        if hasattr(target_data, "target_sequence"):
            return target_data.target_sequence
        elif hasattr(target_data, "sequence"):
            return target_data.sequence
        else:
            # Try to fetch from UniProt
            if hasattr(target_data, "target_components"):
                components = target_data.target_components
                if len(components) > 0:
                    # Try to get UniProt ID
                    uniprot_id = components[0].get("accession", None)
                    if uniprot_id:
                        # Would need biopython or requests to fetch from UniProt
                        warnings.warn(f"UniProt ID found ({uniprot_id}) but UniProt fetching not implemented")
        
        return None
    
    except Exception as e:
        warnings.warn(f"Failed to get sequence from ChEMBL: {e}")
        return None

