"""
target_selector.py

Target selector and model training for multi-target drug discovery.
Supports downloading ChEMBL data and training activity models for different targets.
"""

import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple
from chembl_webresource_client.new_client import new_client
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

from data_utils import load_bioactivity
from featurization import smiles_to_matrix


# Predefined targets with their ChEMBL IDs and display names
PREDEFINED_TARGETS = {
    "HIV-1 Protease": {
        "chembl_id": "CHEMBL243",
        "description": "HIV-1 protease inhibitor screening"
    },
    "SARS-CoV-2 Mpro": {
        "chembl_id": "CHEMBL4301553",
        "description": "SARS-CoV-2 main protease (3CLpro) inhibitor screening"
    },
    "Influenza Neuraminidase": {
        "chembl_id": "CHEMBL3404044",
        "description": "Influenza neuraminidase inhibitor screening"
    },
    "HCV NS5B": {
        "chembl_id": "CHEMBL1615002",
        "description": "Hepatitis C virus NS5B polymerase inhibitor screening"
    },
}


def get_target_info(target_name: str, chembl_id: Optional[str] = None) -> Dict[str, str]:
    """
    Get target information.
    
    Parameters
    ----------
    target_name : str
        Target name (from predefined or custom)
    chembl_id : str, optional
        ChEMBL ID for custom targets
        
    Returns
    -------
    Dict[str, str]
        Dictionary with chembl_id and description
    """
    if target_name in PREDEFINED_TARGETS:
        return PREDEFINED_TARGETS[target_name]
    elif chembl_id:
        return {
            "chembl_id": chembl_id,
            "description": f"Custom target: {chembl_id}"
        }
    else:
        raise ValueError(f"Unknown target: {target_name}")


def download_chembl_data(
    chembl_id: str,
    output_path: str = "bioactivity.csv",
    max_records: int = 2000,
    ic50_threshold_nM: float = 1000.0
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Download bioactivity data from ChEMBL for a given target.
    
    Parameters
    ----------
    chembl_id : str
        ChEMBL target ID (e.g., "CHEMBL243")
    output_path : str
        Path to save the downloaded CSV
    max_records : int
        Maximum number of records to download
    ic50_threshold_nM : float
        IC50 threshold in nM for active/inactive classification
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        DataFrame with bioactivity data and metadata dictionary
    """
    print(f"Downloading ChEMBL data for {chembl_id}...")
    
    activity = new_client.activity
    
    # Filter and fetch IC50 data
    qs = activity.filter(
        target_chembl_id=chembl_id,
        standard_type="IC50",
        standard_units="nM"
    ).only(
        ['canonical_smiles', 'standard_value', 'standard_flag']
    )
    
    # Limit results
    print(f"Fetching up to {max_records} records...")
    res = list(qs[:max_records])
    
    if len(res) == 0:
        raise ValueError(f"No IC50 data found for {chembl_id}. Check the ChEMBL ID.")
    
    df = pd.DataFrame(res)
    
    # Keep only what we need
    cols = ["canonical_smiles", "standard_value", "standard_flag"]
    df = df[cols].dropna(subset=["canonical_smiles", "standard_value"])
    
    # Rename to match data_utils format
    df = df.rename(columns={
        "canonical_smiles": "smiles",
        "standard_value": "activity_value"
    })
    
    # Convert to numeric
    df["activity_value"] = pd.to_numeric(df["activity_value"], errors='coerce')
    df = df.dropna(subset=["activity_value"])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=["smiles", "activity_value"])
    
    # Get target metadata
    metadata = {
        "chembl_id": chembl_id,
        "target_name": chembl_id,
        "organism": "Unknown",
        "target_type": "Unknown",
        "n_records": len(df),
        "n_active": len(df[df["activity_value"] < ic50_threshold_nM]),
        "n_inactive": len(df[df["activity_value"] >= ic50_threshold_nM]),
    }
    
    try:
        target = new_client.target
        target_info_list = list(target.filter(chembl_id=chembl_id).only(['pref_name', 'organism', 'target_type']))
        if len(target_info_list) > 0:
            target_info = target_info_list[0]
            metadata["target_name"] = target_info.get("pref_name", chembl_id)
            metadata["organism"] = target_info.get("organism", "Unknown")
            metadata["target_type"] = target_info.get("target_type", "Unknown")
    except Exception as e:
        print(f"Warning: Could not fetch target metadata from ChEMBL: {e}")
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Downloaded {len(df)} records")
    print(f"   Active (< {ic50_threshold_nM} nM): {metadata['n_active']}")
    print(f"   Inactive (>= {ic50_threshold_nM} nM): {metadata['n_inactive']}")
    print(f"   Saved to: {output_path}")
    
    return df, metadata


def train_target_model(
    bioactivity_path: str,
    target_name: str,
    chembl_id: str,
    ic50_threshold_nM: float = 1000.0,
    model_dir: str = "models"
) -> Tuple[str, Dict[str, any]]:
    """
    Train a RandomForest classifier for target activity prediction.
    
    Parameters
    ----------
    bioactivity_path : str
        Path to bioactivity CSV file
    target_name : str
        Target name (for model filename)
    chembl_id : str
        ChEMBL ID (for metadata)
    ic50_threshold_nM : float
        IC50 threshold for active/inactive classification
    model_dir : str
        Directory to save the model
        
    Returns
    -------
    Tuple[str, Dict]
        Path to saved model and training metrics
    """
    print(f"\nTraining model for {target_name} ({chembl_id})...")
    
    # Load and preprocess data
    df = load_bioactivity(bioactivity_path, ic50_threshold_nM=ic50_threshold_nM)
    
    print(f"Data summary:")
    print(f"  Total molecules: {len(df)}")
    print(f"  Active: {df['active'].sum()}")
    print(f"  Inactive: {(~df['active'].astype(bool)).sum()}")
    
    # Featurize SMILES
    print("\nConverting SMILES to fingerprints...")
    X, idx = smiles_to_matrix(df["smiles"], radius=2, n_bits=2048)
    y = df["active"].values[idx]
    
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Valid SMILES: {len(idx)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("\nTraining RandomForest classifier...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\nModel Performance:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    
    # Create safe filename from target name
    safe_name = target_name.replace(" ", "_").replace("-", "_").lower()
    model_filename = f"{safe_name}_rf.pkl"
    model_path = os.path.join(model_dir, model_filename)
    
    joblib.dump(clf, model_path)
    print(f"\n✅ Model saved to: {model_path}")
    
    # Save metadata
    metadata = {
        "target_name": target_name,
        "chembl_id": chembl_id,
        "model_path": model_path,
        "roc_auc": float(roc_auc),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_active_train": int(y_train.sum()),
        "n_inactive_train": int((~y_train.astype(bool)).sum()),
    }
    
    metadata_path = os.path.join(model_dir, f"{safe_name}_metadata.json")
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_path, metadata


def download_and_train_target(
    target_name: str,
    chembl_id: Optional[str] = None,
    max_records: int = 2000,
    ic50_threshold_nM: float = 1000.0
) -> Tuple[str, Dict[str, any]]:
    """
    Complete pipeline: download ChEMBL data and train model for a target.
    
    Parameters
    ----------
    target_name : str
        Target name (from predefined or custom)
    chembl_id : str, optional
        ChEMBL ID for custom targets (required if target_name not in predefined)
    max_records : int
        Maximum number of records to download
    ic50_threshold_nM : float
        IC50 threshold for active/inactive classification
        
    Returns
    -------
    Tuple[str, Dict]
        Path to saved model and metadata dictionary
    """
    # Get target info
    if target_name in PREDEFINED_TARGETS:
        target_info = PREDEFINED_TARGETS[target_name]
        chembl_id = target_info["chembl_id"]
    elif chembl_id:
        target_info = {"chembl_id": chembl_id, "description": f"Custom: {chembl_id}"}
    else:
        raise ValueError(f"Must provide chembl_id for custom target: {target_name}")
    
    # Create unique bioactivity file for this target
    safe_name = target_name.replace(" ", "_").replace("-", "_").lower()
    bioactivity_path = f"bioactivity_{safe_name}.csv"
    
    # Download data
    df, download_metadata = download_chembl_data(
        chembl_id=chembl_id,
        output_path=bioactivity_path,
        max_records=max_records,
        ic50_threshold_nM=ic50_threshold_nM
    )
    
    # Train model
    model_path, train_metadata = train_target_model(
        bioactivity_path=bioactivity_path,
        target_name=target_name,
        chembl_id=chembl_id,
        ic50_threshold_nM=ic50_threshold_nM
    )
    
    # Combine metadata
    full_metadata = {
        **download_metadata,
        **train_metadata,
        "bioactivity_path": bioactivity_path
    }
    
    return model_path, full_metadata


def list_available_targets() -> List[Dict[str, str]]:
    """
    List all available trained target models.
    
    Returns
    -------
    List[Dict]
        List of target information dictionaries
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    
    available = []
    
    # Check predefined targets
    for target_name, target_info in PREDEFINED_TARGETS.items():
        safe_name = target_name.replace(" ", "_").replace("-", "_").lower()
        model_path = os.path.join(models_dir, f"{safe_name}_rf.pkl")
        if os.path.exists(model_path):
            available.append({
                "target_name": target_name,
                "chembl_id": target_info["chembl_id"],
                "model_path": model_path,
                "description": target_info["description"]
            })
    
    # Check for other models (custom targets)
    for filename in os.listdir(models_dir):
        if filename.endswith("_rf.pkl") and not filename.startswith("hiv_"):
            # Try to extract target name
            safe_name = filename.replace("_rf.pkl", "")
            # Try to load metadata
            metadata_path = os.path.join(models_dir, f"{safe_name}_metadata.json")
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                available.append({
                    "target_name": metadata.get("target_name", safe_name),
                    "chembl_id": metadata.get("chembl_id", "Unknown"),
                    "model_path": os.path.join(models_dir, filename),
                    "description": f"Custom target: {metadata.get('chembl_id', 'Unknown')}"
                })
    
    return available

