"""
pipeline.py

Main screening pipeline that evaluates molecules across target activity
and ADMET endpoints, returning a comprehensive MoleculeReport.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import os
import joblib
import numpy as np

from featurization import smiles_to_morgan_fp
from admet_loader import ADMET_TASKS


@dataclass
class MoleculeReport:
    """Comprehensive report for a single molecule."""
    smiles: str
    features: np.ndarray
    target: Dict[str, Any]
    admet: Dict[str, Any]
    score: float


class ModelRegistry:
    """Lazy-loading registry for all trained models."""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._target_models: Dict[str, str] = {}  # Maps target_name -> model_key
        self._loaded = False
    
    def load(self, target_name: Optional[str] = None):
        """
        Load all available models from disk.
        
        Parameters
        ----------
        target_name : str, optional
            Specific target to load. If None, loads all available targets.
        """
        if self._loaded and target_name is None:
            return
        
        # Load target models from config
        try:
            from config_loader import get_targets_config
            targets_config = get_targets_config()
        except Exception:
            targets_config = {}
        
        # Also load from target_selector for backward compatibility
        from target_selector import list_available_targets, PREDEFINED_TARGETS
        
        available_targets = list_available_targets()
        
        # Load targets from config
        for target_key, target_config in targets_config.items():
            model_path = target_config.get("model_path")
            legacy_path = target_config.get("legacy_path")
            target_name_key = target_config.get("name", target_key)
            
            # Try primary model path
            if model_path and os.path.exists(model_path):
                try:
                    self._models[target_key] = joblib.load(model_path)
                    self._target_models[target_name_key] = target_key
                    print(f"Loaded target model: {target_name_key} ({target_key})")
                except Exception as e:
                    print(f"Warning: Could not load {model_path}: {e}")
            # Try legacy path
            elif legacy_path and os.path.exists(legacy_path):
                try:
                    self._models[target_key] = joblib.load(legacy_path)
                    self._target_models[target_name_key] = target_key
                    print(f"Loaded target model (legacy): {target_name_key} ({target_key})")
                except Exception as e:
                    print(f"Warning: Could not load {legacy_path}: {e}")
        
        # Also check legacy HIV model for backward compatibility
        hiv_path = "models/hiv_protease_rf.pkl"
        if os.path.exists(hiv_path) and "HIV_protease" not in self._models and "HIV1_PROTEASE" not in self._models:
            self._models["HIV_protease"] = joblib.load(hiv_path)
            self._target_models["HIV-1 Protease"] = "HIV_protease"
            print(f"Loaded: {hiv_path}")
        
        # Load available targets from target_selector (for dynamically trained targets)
        for target_info in available_targets:
            target_name_key = target_info["target_name"]
            model_path = target_info["model_path"]
            
            # Skip if already loaded from config
            if target_name_key in self._target_models:
                continue
            
            # Create model key
            safe_name = target_name_key.replace(" ", "_").replace("-", "_").lower()
            model_key = f"{safe_name}_target"
            
            if model_key not in self._models:
                try:
                    self._models[model_key] = joblib.load(model_path)
                    self._target_models[target_name_key] = model_key
                    print(f"Loaded target model: {target_name_key} -> {model_key}")
                except Exception as e:
                    print(f"Warning: Could not load {model_path}: {e}")
        
        # Load ADMET models
        for task_key in ADMET_TASKS.keys():
            if task_key in self._models:
                continue
                
            task_config = ADMET_TASKS[task_key]
            task_type = task_config["type"]
            
            if task_type == "classification":
                model_path = os.path.join("models", "admet", f"{task_key}_clf.pkl")
            else:  # regression
                model_path = os.path.join("models", "admet", f"{task_key}_reg.pkl")
            
            if os.path.exists(model_path):
                self._models[task_key] = joblib.load(model_path)
                print(f"Loaded: {model_path}")
        
        self._loaded = True
    
    def get_target_model(self, target_name: str) -> Optional[Any]:
        """
        Get a target activity model by target name.
        
        Parameters
        ----------
        target_name : str
            Target name (e.g., "HIV-1 Protease")
            
        Returns
        -------
        Model or None
        """
        if not self._loaded:
            self.load()
        
        model_key = self._target_models.get(target_name)
        if model_key:
            return self._models.get(model_key)
        
        # Fallback: try legacy HIV_protease key
        if target_name == "HIV-1 Protease":
            return self._models.get("HIV_protease")
        
        return None
    
    def has_target_model(self, target_name: str) -> bool:
        """Check if a target model is available."""
        if not self._loaded:
            self.load()
        return target_name in self._target_models or (target_name == "HIV-1 Protease" and "HIV_protease" in self._models)
    
    def list_targets(self) -> List[str]:
        """
        List all available target keys from config.
        
        Returns
        -------
        List[str]
            List of target keys (e.g., ["HIV1_PROTEASE", "SARS2_MPRO", ...])
        """
        try:
            from config_loader import get_targets_config
            targets_config = get_targets_config()
            return list(targets_config.keys())
        except Exception:
            return []
    
    def get(self, name: str) -> Optional[Any]:
        """Get a model by name, loading if necessary."""
        if not self._loaded:
            self.load()
        return self._models.get(name)
    
    def has_model(self, name: str) -> bool:
        """Check if a model is available."""
        if not self._loaded:
            self.load()
        return name in self._models


# Global registry instance
registry = ModelRegistry()


def composite_score(
    target_p_active: float, 
    admet_outputs: Dict[str, Any],
    weights_config: Optional[Dict[str, Any]] = None
) -> float:
    """
    Compute a composite drug-likeness score using configurable weights.
    
    Parameters
    ----------
    target_p_active : float
        Probability of target activity
    admet_outputs : Dict[str, Any]
        Dictionary of ADMET predictions
    weights_config : Dict[str, Any], optional
        Scoring weights configuration. If None, loads from config file.
        
    Returns
    -------
    float
        Composite score (higher is better, typically 0-1 range)
    """
    # Load weights from config if not provided
    if weights_config is None:
        try:
            from config_loader import get_scoring_weights
            weights_config = get_scoring_weights()
        except Exception:
            # Fallback to default weights
            weights_config = {
                "target_weight": 0.4,
                "tox_penalty": {"hERG": 0.3, "DILI": 0.2, "AMES": 0.2},
                "absorption_weight": {"HIA_Hou": 0.1, "caco2_wang": 0.1},
                "other": {"LD50_Zhu": 0.05}
            }
    
    # Start with target activity
    target_weight = weights_config.get("target_weight", 0.4)
    score = target_weight * target_p_active
    
    # Penalize high toxicity probabilities
    tox_penalty = weights_config.get("tox_penalty", {})
    for tox_key, penalty_weight in tox_penalty.items():
        if tox_key in admet_outputs:
            if "prob" in admet_outputs[tox_key]:
                p_tox = admet_outputs[tox_key]["prob"]
                score -= penalty_weight * p_tox
    
    # Reward/penalize absorption properties
    absorption_weight = weights_config.get("absorption_weight", {})
    
    if "caco2_wang" in admet_outputs and "value" in admet_outputs["caco2_wang"]:
        caco2 = admet_outputs["caco2_wang"]["value"]
        weight = absorption_weight.get("caco2_wang", 0.1)
        # Low Caco-2 is bad (typical threshold ~ -5.15 log units)
        if caco2 < -5.15:
            score -= weight
    
    if "HIA_Hou" in admet_outputs and "prob" in admet_outputs["HIA_Hou"]:
        p_hia = admet_outputs["HIA_Hou"]["prob"]
        weight = absorption_weight.get("HIA_Hou", 0.1)
        score += weight * p_hia  # Reward good absorption
    
    # Distribution properties
    distribution = weights_config.get("distribution", {})
    if "PPBR_AZ" in admet_outputs and "value" in admet_outputs["PPBR_AZ"]:
        ppbr = admet_outputs["PPBR_AZ"]["value"]
        # Moderate values (0.3-0.9) are ideal
        if ppbr < 0.3 or ppbr > 0.9:
            score -= 0.05
    
    # Other properties
    other = weights_config.get("other", {})
    if "LD50_Zhu" in admet_outputs and "value" in admet_outputs["LD50_Zhu"]:
        ld50 = admet_outputs["LD50_Zhu"]["value"]
        weight = other.get("LD50_Zhu", 0.05)
        # Very high LD50 (> 5) might indicate poor compound properties
        if ld50 > 5.0:
            score -= weight
    
    # Docking score (if available)
    docking_config = weights_config.get("docking", {})
    if docking_config.get("enabled", False) and "docking_score" in admet_outputs:
        docking_score = admet_outputs["docking_score"]
        weight = docking_config.get("weight", 0.1)
        normalize_range = docking_config.get("normalize_range", [-15.0, -5.0])
        # Normalize docking score (lower/better binding = higher normalized score)
        min_score, max_score = normalize_range
        if min_score <= docking_score <= max_score:
            normalized = (docking_score - min_score) / (max_score - min_score)
            score += weight * normalized
    
    # Ensure score stays in reasonable range
    score = max(0.0, min(1.0, score))
    
    return float(score)


def evaluate_single_smiles(smiles: str) -> MoleculeReport:
    """
    Evaluate a single SMILES string across all models.
    
    Parameters
    ----------
    smiles : str
        SMILES string to evaluate
        
    Returns
    -------
    MoleculeReport
        Comprehensive report with target and ADMET predictions
        
    Raises
    ------
    ValueError
        If SMILES cannot be featurized
    """
    # Featurize SMILES
    fp = smiles_to_morgan_fp(smiles)
    if fp is None:
        raise ValueError(f"Could not featurize SMILES: {smiles}")
    
    X = fp.reshape(1, -1)
    
    # Evaluate HIV protease target model
    target_info: Dict[str, Any] = {}
    hiv_model = registry.get("HIV_protease")
    if hiv_model is not None:
        if hasattr(hiv_model, "predict_proba"):
            p_active = hiv_model.predict_proba(X)[0, 1]
            target_info["p_active"] = float(p_active)
        else:
            target_info["p_active"] = 0.0
    else:
        target_info["p_active"] = None
        print("Warning: HIV protease model not loaded")
    
    # Evaluate all ADMET models
    admet_info: Dict[str, Any] = {}
    
    for task_key in ADMET_TASKS.keys():
        model = registry.get(task_key)
        if model is None:
            continue
        
        task_config = ADMET_TASKS[task_key]
        task_type = task_config["type"]
        
        if task_type == "classification":
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[0, 1]
                admet_info[task_key] = {"prob": float(prob)}
            else:
                pred = model.predict(X)[0]
                admet_info[task_key] = {"pred": int(pred)}
        else:  # regression
            value = model.predict(X)[0]
            admet_info[task_key] = {"value": float(value)}
    
    # Compute composite score
    target_p = target_info.get("p_active", 0.0) or 0.0
    score = composite_score(target_p, admet_info)
    
    return MoleculeReport(
        smiles=smiles,
        features=fp,
        target=target_info,
        admet=admet_info,
        score=score,
    )

