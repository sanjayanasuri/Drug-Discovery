"""
admet_loader.py

Load ADMET datasets from PyTDC (Therapeutics Data Commons).
Provides loaders for all ADMET tasks in the screening panel.
"""

import pandas as pd
from typing import Tuple
from tdc.single_pred import Tox, ADME


# ADMET task configuration mapping
ADMET_TASKS = {
    # Toxicity (T)
    "hERG": {"group": "tox", "name": "hERG", "type": "classification"},
    "AMES": {"group": "tox", "name": "AMES", "type": "classification"},
    "DILI": {"group": "tox", "name": "DILI", "type": "classification"},
    "LD50_Zhu": {"group": "tox", "name": "LD50_Zhu", "type": "regression"},
    
    # Absorption (A)
    "caco2_wang": {"group": "adme", "name": "Caco2_Wang", "type": "regression"},
    "HIA_Hou": {"group": "adme", "name": "HIA_Hou", "type": "classification"},
    
    # Distribution (D)
    "BBB_Martins": {"group": "adme", "name": "BBB_Martins", "type": "classification"},
    "PPBR_AZ": {"group": "adme", "name": "PPBR_AZ", "type": "regression"},
    
    # Metabolism (M)
    "CYP3A4_Veith": {"group": "adme", "name": "CYP3A4_Veith", "type": "classification"},
    "CYP2D6_Veith": {"group": "adme", "name": "CYP2D6_Veith", "type": "classification"},
    
    # Excretion (E)
    "Half_Life_Obach": {"group": "adme", "name": "Half_Life_Obach", "type": "regression"},
}


def load_admet_dataset(task_key: str) -> Tuple[pd.DataFrame, str]:
    """
    Load an ADMET dataset from TDC by task key.
    
    Parameters
    ----------
    task_key : str
        One of the keys in ADMET_TASKS (e.g., "hERG", "caco2_wang", etc.)
        
    Returns
    -------
    Tuple[pd.DataFrame, str]
        DataFrame with at least 'smiles' and 'label' (or 'value' for regression) columns,
        and task type string ("classification" or "regression")
    """
    if task_key not in ADMET_TASKS:
        raise ValueError(
            f"Unknown task: {task_key}. Available tasks: {list(ADMET_TASKS.keys())}"
        )
    
    task_config = ADMET_TASKS[task_key]
    task_name = task_config["name"]
    task_type = task_config["type"]
    task_group = task_config["group"]
    
    # Load from TDC
    if task_group == "tox":
        data = Tox(name=task_name)
    else:  # adme
        data = ADME(name=task_name)
    
    df = data.get_data()
    
    # Standardize column names
    if "Drug" in df.columns:
        df = df.rename(columns={"Drug": "smiles"})
    
    if "Y" in df.columns:
        if task_type == "classification":
            df = df.rename(columns={"Y": "label"})
        else:  # regression
            df = df.rename(columns={"Y": "value"})
    
    return df, task_type


# Legacy functions for backward compatibility
def load_herg() -> pd.DataFrame:
    """Load hERG toxicity dataset from TDC."""
    df, _ = load_admet_dataset("hERG")
    return df


def load_caco2() -> pd.DataFrame:
    """Load Caco-2 permeability dataset from TDC."""
    df, _ = load_admet_dataset("caco2_wang")
    return df


def load_ld50() -> pd.DataFrame:
    """Load LD50 toxicity dataset from TDC."""
    df, _ = load_admet_dataset("LD50_Zhu")
    return df
