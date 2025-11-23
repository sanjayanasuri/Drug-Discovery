"""
config_loader.py

Load configuration files for targets, scoring weights, etc.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML file
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def get_targets_config() -> Dict[str, Dict[str, Any]]:
    """
    Load targets configuration.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping target keys to target configs
    """
    config_path = os.path.join("config", "targets.yaml")
    if not os.path.exists(config_path):
        # Return default if config doesn't exist
        return {
            "HIV1_PROTEASE": {
                "chembl_id": "CHEMBL243",
                "name": "HIV-1 Protease",
                "description": "HIV-1 protease inhibitor screening",
                "model_path": "models/hiv1_protease_rf.pkl",
                "legacy_path": "models/hiv_protease_rf.pkl"
            }
        }
    
    config = load_yaml_config(config_path)
    return config.get("targets", {})


def get_scoring_weights() -> Dict[str, Any]:
    """
    Load scoring weights configuration.
    
    Returns
    -------
    Dict[str, Any]
        Scoring weights configuration
    """
    config_path = os.path.join("config", "scoring_weights.yaml")
    if not os.path.exists(config_path):
        # Return default weights
        return {
            "target_weight": 0.4,
            "tox_penalty": {
                "hERG": 0.3,
                "DILI": 0.2,
                "AMES": 0.2,
                "CYP3A4_Veith": 0.1,
                "CYP2D6_Veith": 0.1
            },
            "absorption_weight": {
                "HIA_Hou": 0.1,
                "caco2_wang": 0.1
            },
            "distribution": {
                "BBB_Martins": 0.05,
                "PPBR_AZ": 0.0
            },
            "other": {
                "LD50_Zhu": 0.05,
                "Half_Life_Obach": 0.0
            },
            "docking": {
                "enabled": False,
                "weight": 0.1,
                "normalize_range": [-15.0, -5.0]
            }
        }
    
    config = load_yaml_config(config_path)
    return config.get("scoring", {})

