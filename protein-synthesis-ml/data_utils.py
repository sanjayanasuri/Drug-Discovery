"""
data_utils.py

Loading and labeling ChEMBL bioactivity data.
"""

import pandas as pd
import numpy as np


def load_bioactivity(
    path: str = "bioactivity.csv",
    ic50_threshold_nM: float = 1000.0,
):
    """
    Load bioactivity CSV and create classification/regression labels.

    Returns
    -------
    df : pandas.DataFrame
        Columns:
        - smiles
        - activity_value (nM)
        - active (binary label)
        - activity_value_molar
        - pIC50
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["smiles", "activity_value"])

    # Classification label: 1 = active, 0 = inactive
    df["active"] = (df["activity_value"] < ic50_threshold_nM).astype(int)

    # Regression label: pIC50
    df["activity_value_molar"] = df["activity_value"] * 1e-9
    df = df[df["activity_value_molar"] > 0]  # avoid log(0)
    df["pIC50"] = -np.log10(df["activity_value_molar"])

    return df
