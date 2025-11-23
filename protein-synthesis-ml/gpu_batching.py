"""
gpu_batching.py

GPU-accelerated batch processing for molecular featurization and model inference.
Falls back to CPU if GPU is not available.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import warnings

# Try to import GPU libraries
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

from featurization import smiles_to_matrix


def batch_featurize_gpu(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048,
    batch_size: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Featurize SMILES strings in batches (GPU-accelerated if available).
    
    Currently uses CPU (RDKit doesn't have native GPU support),
    but batches efficiently for better performance.
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings
    radius : int
        Morgan fingerprint radius
    n_bits : int
        Number of bits in fingerprint
    batch_size : int
        Batch size for processing
        
    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_valid, n_bits)
    idx : np.ndarray
        Indices of valid SMILES
    """
    # For now, use CPU batching (RDKit doesn't support GPU)
    # But we batch efficiently
    all_fps = []
    all_idx = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        X_batch, idx_batch = smiles_to_matrix(batch, radius=radius, n_bits=n_bits)
        
        if len(X_batch) > 0:
            all_fps.append(X_batch)
            # Adjust indices to original list
            all_idx.append(idx_batch + i)
    
    if len(all_fps) == 0:
        return np.array([]), np.array([], dtype=int)
    
    X = np.vstack(all_fps)
    idx = np.concatenate(all_idx)
    
    return X, idx


def batch_predict_gpu(
    X: np.ndarray,
    models: Dict[str, Any],
    device: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Run batch predictions on multiple models (GPU-accelerated if available).
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    models : Dict[str, Any]
        Dictionary of models to run predictions on
    device : str, optional
        Device to use ('cuda', 'cpu', or None for auto-detect)
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of predictions keyed by model name
    """
    if device is None:
        device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    results = {}
    
    for model_name, model in models.items():
        try:
            # Check if model is PyTorch/TensorFlow model
            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                model.eval()
                model = model.to(device)
                X_tensor = torch.from_numpy(X).float().to(device)
                
                with torch.no_grad():
                    if hasattr(model, "forward"):
                        pred = model(X_tensor).cpu().numpy()
                    else:
                        pred = model(X_tensor).cpu().numpy()
                
                results[model_name] = pred
            else:
                # scikit-learn or other CPU models
                # Use batch prediction if available
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X)
                    if pred.ndim == 2 and pred.shape[1] == 2:
                        pred = pred[:, 1]  # Get probability of positive class
                    results[model_name] = pred
                elif hasattr(model, "predict"):
                    results[model_name] = model.predict(X)
                else:
                    warnings.warn(f"Model {model_name} doesn't support batch prediction")
                    results[model_name] = None
        except Exception as e:
            warnings.warn(f"Error predicting with {model_name}: {e}")
            results[model_name] = None
    
    return results


def vectorized_composite_score(
    target_p_active: np.ndarray,
    admet_predictions: Dict[str, np.ndarray],
    weights_config: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Compute composite scores for a batch of molecules using vectorized operations.
    
    Parameters
    ----------
    target_p_active : np.ndarray
        Array of target activity probabilities (n_samples,)
    admet_predictions : Dict[str, np.ndarray]
        Dictionary of ADMET predictions keyed by task name
        Values are arrays of shape (n_samples,)
    weights_config : Dict[str, Any], optional
        Scoring weights configuration
        
    Returns
    -------
    np.ndarray
        Array of composite scores (n_samples,)
    """
    if weights_config is None:
        try:
            from config_loader import get_scoring_weights
            weights_config = get_scoring_weights()
        except Exception:
            weights_config = {
                "target_weight": 0.4,
                "tox_penalty": {"hERG": 0.3, "DILI": 0.2, "AMES": 0.2},
                "absorption_weight": {"HIA_Hou": 0.1, "caco2_wang": 0.1},
            }
    
    n_samples = len(target_p_active)
    scores = np.zeros(n_samples)
    
    # Target activity
    target_weight = weights_config.get("target_weight", 0.4)
    scores += target_weight * target_p_active
    
    # Toxicity penalties
    tox_penalty = weights_config.get("tox_penalty", {})
    for tox_key, penalty_weight in tox_penalty.items():
        if tox_key in admet_predictions and admet_predictions[tox_key] is not None:
            scores -= penalty_weight * admet_predictions[tox_key]
    
    # Absorption rewards
    absorption_weight = weights_config.get("absorption_weight", {})
    for abs_key, weight in absorption_weight.items():
        if abs_key in admet_predictions and admet_predictions[abs_key] is not None:
            scores += weight * admet_predictions[abs_key]
    
    # Clamp to [0, 1]
    scores = np.clip(scores, 0.0, 1.0)
    
    return scores


def check_gpu_available() -> Tuple[bool, str]:
    """
    Check if GPU is available for computation.
    
    Returns
    -------
    bool
        True if GPU is available
    str
        Status message
    """
    if not TORCH_AVAILABLE:
        return False, "PyTorch not installed. Install with: pip install torch"
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return True, f"GPU available: {gpu_name} ({gpu_memory:.1f} GB)"
    else:
        return False, "No GPU detected. Using CPU."

