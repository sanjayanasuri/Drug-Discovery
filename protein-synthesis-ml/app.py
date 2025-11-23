"""
app.py

Streamlit UI for the Drug Discovery Screening Pipeline.
Step-by-step workflow: Upload → Activity Screening → Select Hits → ADMET Evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any, Optional

from pipeline import registry, composite_score
from featurization import smiles_to_morgan_fp, smiles_to_matrix
from admet_loader import ADMET_TASKS
from clustering import cluster_molecules, DEFAULT_TANIMOTO_THRESHOLD
from docking import dock_smiles, check_vina_available, check_obabel_available, DEFAULT_RECEPTOR_PATH, DEFAULT_CONFIG_PATH
from sar import generate_and_score_analogs
from nl_queries import apply_nl_query_to_dataframe
from lead_optimization import optimize_leads
from target_selector import (
    PREDEFINED_TARGETS,
    download_and_train_target,
    list_available_targets,
    get_target_info
)


@st.cache_resource
def load_models():
    """Load all models into registry (cached)."""
    registry.load()
    return registry


@st.cache_data
def featurize_smiles_batch(smiles_list: List[str]) -> tuple:
    """
    Featurize a batch of SMILES strings (cached).
    
    Returns
    -------
    tuple: (X array, valid_indices list, valid_smiles list)
    """
    X, idx = smiles_to_matrix(smiles_list, radius=2, n_bits=2048)
    valid_smiles = [smiles_list[i] for i in idx]
    return X, idx, valid_smiles


def predict_target_activity_batch(X: np.ndarray, target_model) -> np.ndarray:
    """Predict target activity for a batch of molecules."""
    if target_model is None:
        return np.zeros(len(X))
    if hasattr(target_model, "predict_proba"):
        return target_model.predict_proba(X)[:, 1]
    else:
        return np.zeros(len(X))


def predict_admet_batch(X: np.ndarray, task_key: str, model, task_type: str) -> np.ndarray:
    """Predict ADMET property for a batch of molecules."""
    if task_type == "classification":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X).astype(float)
    else:  # regression
        return model.predict(X)


def analyze_lead_factors(row: pd.Series) -> Dict[str, Any]:
    """
    Analyze a lead molecule to identify key factors contributing to its ranking.
    
    Parameters
    ----------
    row : pd.Series
        Row from leads DataFrame with all ADMET predictions
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with 'strengths', 'weaknesses', and 'evidence'
    """
    strengths = []
    weaknesses = []
    evidence = {}
    
    # Target activity (primary factor)
    p_active = row.get("hiv1_p_active", 0.0)
    target_name = st.session_state.get("selected_target", "Target")
    if p_active > 0.8:
        strengths.append(f"High {target_name} activity (p_active = {p_active:.3f})")
    elif p_active > 0.6:
        strengths.append(f"Moderate {target_name} activity (p_active = {p_active:.3f})")
    elif p_active < 0.4:
        weaknesses.append(f"Low {target_name} activity (p_active = {p_active:.3f})")
    evidence[f"{target_name} p_active"] = f"{p_active:.3f}"
    
    # Toxicity factors (lower is better)
    tox_props = {
        "hERG_prob": ("hERG", "cardiac risk"),
        "AMES_prob": ("AMES", "mutagenicity"),
        "DILI_prob": ("DILI", "liver toxicity"),
    }
    
    for col, (name, desc) in tox_props.items():
        if col in row and pd.notna(row[col]):
            val = row[col]
            if val > 0.7:
                weaknesses.append(f"High {name} risk ({desc}): {val:.3f}")
            elif val > 0.3:
                weaknesses.append(f"Moderate {name} risk ({desc}): {val:.3f}")
            elif val < 0.2:
                strengths.append(f"Low {name} risk ({desc}): {val:.3f}")
            evidence[name] = f"{val:.3f}"
    
    # Absorption factors (higher is better)
    if "HIA_Hou_prob" in row and pd.notna(row["HIA_Hou_prob"]):
        hia = row["HIA_Hou_prob"]
        if hia > 0.7:
            strengths.append(f"Good intestinal absorption (HIA = {hia:.3f})")
        elif hia < 0.3:
            weaknesses.append(f"Poor intestinal absorption (HIA = {hia:.3f})")
        evidence["HIA"] = f"{hia:.3f}"
    
    if "caco2_wang_value" in row and pd.notna(row["caco2_wang_value"]):
        caco2 = row["caco2_wang_value"]
        if caco2 > -5.15:
            strengths.append(f"Good permeability (Caco-2 = {caco2:.2f})")
        elif caco2 < -6.0:
            weaknesses.append(f"Poor permeability (Caco-2 = {caco2:.2f})")
        evidence["Caco-2"] = f"{caco2:.2f}"
    
    # BBB (higher is better, but context-dependent)
    if "BBB_Martins_prob" in row and pd.notna(row["BBB_Martins_prob"]):
        bbb = row["BBB_Martins_prob"]
        if bbb > 0.7:
            strengths.append(f"Good BBB penetration: {bbb:.3f}")
        elif bbb < 0.3:
            weaknesses.append(f"Poor BBB penetration: {bbb:.3f}")
        evidence["BBB"] = f"{bbb:.3f}"
    
    # CYP inhibition (lower is better)
    for cyp_col in ["CYP3A4_Veith_prob", "CYP2D6_Veith_prob"]:
        if cyp_col in row and pd.notna(row[cyp_col]):
            cyp_val = row[cyp_col]
            cyp_name = "CYP3A4" if "3A4" in cyp_col else "CYP2D6"
            if cyp_val > 0.7:
                weaknesses.append(f"High {cyp_name} inhibition risk: {cyp_val:.3f}")
            elif cyp_val < 0.2:
                strengths.append(f"Low {cyp_name} inhibition risk: {cyp_val:.3f}")
            evidence[cyp_name] = f"{cyp_val:.3f}"
    
    # LD50 (higher is better, but very high might indicate issues)
    if "LD50_Zhu_value" in row and pd.notna(row["LD50_Zhu_value"]):
        ld50 = row["LD50_Zhu_value"]
        if 2.0 < ld50 <= 5.0:
            strengths.append(f"Reasonable toxicity profile (LD50 = {ld50:.2f})")
        elif ld50 < 2.0:
            weaknesses.append(f"Low LD50 (more toxic): {ld50:.2f}")
        elif ld50 > 5.0:
            weaknesses.append(f"Very high LD50 (may indicate poor bioavailability): {ld50:.2f}")
        evidence["LD50"] = f"{ld50:.2f}"
    
    # Composite score
    composite = row.get("composite_score", 0.0)
    evidence["Composite Score"] = f"{composite:.3f}"
    
    return {
        "strengths": strengths if strengths else ["No major strengths identified"],
        "weaknesses": weaknesses if weaknesses else ["No major weaknesses identified"],
        "evidence": evidence
    }


def get_admet_property_description(task_key: str) -> str:
    """
    Get a human-readable description of an ADMET property.
    
    Parameters
    ----------
    task_key : str
        ADMET task key (e.g., "hERG", "caco2_wang", etc.)
        
    Returns
    -------
    str
        Description of the property and whether higher/lower is better
    """
    descriptions = {
        "hERG": "Probability that the molecule blocks the hERG ion channel. **Lower is better** (cardiac risk).",
        "AMES": "Probability that the compound is mutagenic (causes DNA mutations). **Lower is better** (safer).",
        "DILI": "Probability of drug-induced liver injury (liver toxicity risk). **Lower is better**.",
        "LD50_Zhu": "Predicted toxicity dose (LD50). **Higher is better** (less toxic, requires higher dose for toxicity).",
        "caco2_wang": "Predicted Caco-2 cell permeability (absorption indicator). **Higher is better** (better oral absorption).",
        "HIA_Hou": "Probability of human intestinal absorption. **Higher is better** (better absorption).",
        "BBB_Martins": "Probability of blood-brain barrier penetration. **Higher is better** (for CNS drugs).",
        "PPBR_AZ": "Plasma protein binding rate. Moderate values (0.3-0.9) are ideal (too high = less free drug, too low = rapid clearance).",
        "CYP3A4_Veith": "Probability of CYP3A4 enzyme inhibition. **Lower is better** (reduces drug-drug interaction risk).",
        "CYP2D6_Veith": "Probability of CYP2D6 enzyme inhibition. **Lower is better** (reduces drug-drug interaction risk).",
        "Half_Life_Obach": "Predicted half-life in hours. Moderate values (1-24h) are ideal (too short = frequent dosing, too long = accumulation risk).",
    }
    return descriptions.get(task_key, f"{task_key}: ADMET property prediction.")


def get_risk_color(value: float, task_key: str, task_type: str) -> str:
    """Determine risk color (green/yellow/red) for an ADMET property."""
    if task_type == "classification":
        if task_key in ["hERG", "AMES", "DILI", "CYP3A4_Veith", "CYP2D6_Veith"]:
            if value > 0.7:
                return "red"
            elif value > 0.3:
                return "yellow"
            else:
                return "green"
        elif task_key in ["HIA_Hou", "BBB_Martins"]:
            if value > 0.7:
                return "green"
            elif value > 0.3:
                return "yellow"
            else:
                return "red"
        else:
            return "gray"
    else:  # regression
        if task_key == "caco2_wang":
            if value > -5.15:
                return "green"
            elif value > -6.0:
                return "yellow"
            else:
                return "red"
        elif task_key == "PPBR_AZ":
            if 0.3 < value < 0.9:
                return "green"
            elif 0.1 < value < 0.95:
                return "yellow"
            else:
                return "red"
        elif task_key == "Half_Life_Obach":
            if 1.0 < value < 24.0:
                return "green"
            elif 0.5 < value < 48.0:
                return "yellow"
            else:
                return "red"
        elif task_key == "LD50_Zhu":
            if value > 3.0:
                return "green"
            elif value > 2.0:
                return "yellow"
            else:
                return "red"
        else:
            return "gray"


def auto_detect_csv_format(df: pd.DataFrame) -> Dict[str, str]:
    """
    Auto-detect column names for SMILES and activity values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping 'smiles' and 'activity_value' to column names
    """
    column_mapping = {}
    
    # Common SMILES column names
    smiles_candidates = ["smiles", "SMILES", "canonical_smiles", "mol", "molecule", "structure"]
    for col in df.columns:
        if col.lower() in [s.lower() for s in smiles_candidates]:
            column_mapping["smiles"] = col
            break
    
    # Common activity column names
    activity_candidates = ["ic50", "IC50", "activity_value", "value", "pIC50", "ki", "KI", "activity", "potency"]
    for col in df.columns:
        if col.lower() in [a.lower() for a in activity_candidates]:
            column_mapping["activity_value"] = col
            break
    
    return column_mapping


def parameter_explainer(param_name: str, value: float) -> str:
    """
    Explain what a parameter does with examples.
    
    Parameters
    ----------
    param_name : str
        Name of the parameter
    value : float
        Current value
        
    Returns
    -------
    str
        Markdown-formatted explanation
    """
    explanations = {
        "p_active_threshold": {
            "description": "Minimum probability of target activity to be considered a 'hit'",
            "examples": {
                0.5: "Very permissive - includes many molecules, some may be false positives",
                0.65: "Balanced - good trade-off between finding hits and avoiding false positives",
                0.8: "Strict - only high-confidence hits, may miss some active molecules"
            }
        },
        "herg_threshold": {
            "description": "Maximum hERG probability (cardiac toxicity risk)",
            "examples": {
                0.3: "Very safe - excludes molecules with any cardiac risk",
                0.5: "Moderate safety - allows some risk for promising leads",
                0.7: "Permissive - may include molecules with cardiac concerns"
            }
        },
        "sa_score": {
            "description": "Maximum synthetic accessibility score (lower = easier to synthesize)",
            "examples": {
                4.0: "Very easy to synthesize - simple molecules only",
                6.0: "Moderate complexity - most drug-like molecules",
                8.0: "Complex - includes challenging synthetic targets"
            }
        },
        "ic50_threshold": {
            "description": "IC50 threshold in nM (molecules below this are considered 'active')",
            "examples": {
                100.0: "Very strict - only highly potent molecules",
                1000.0: "Standard - typical drug discovery threshold",
                10000.0: "Permissive - includes weakly active molecules"
            }
        }
    }
    
    exp = explanations.get(param_name, {})
    desc = exp.get("description", "Parameter description")
    examples = exp.get("examples", {})
    
    # Find closest example
    if examples:
        closest_value = min(examples.keys(), key=lambda x: abs(x - value))
        example_text = examples[closest_value]
        return f"**{desc}**\n\n💡 **At {value:.2f}:** {example_text}"
    else:
        return f"**{desc}**"


def show_threshold_impact(df: pd.DataFrame, threshold_col: str, threshold_value: float, target_col: str = "hiv1_p_active"):
    """
    Show visual impact of changing threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame
    threshold_col : str
        Name of threshold parameter
    threshold_value : float
        Current threshold value
    target_col : str
        Column to filter on
    """
    if target_col not in df.columns:
        return
    
    # Calculate how many molecules pass at different thresholds
    thresholds = np.linspace(
        max(0.1, df[target_col].min()),
        min(0.99, df[target_col].max()),
        30
    )
    passing_counts = [(df[target_col] >= t).sum() for t in thresholds]
    passing_pct = [100 * count / len(df) for count in passing_counts]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, passing_counts, linewidth=2, label="Passing molecules")
    ax.axvline(x=threshold_value, color="red", linestyle="--", linewidth=2, label=f"Current: {threshold_value:.2f}")
    
    current_count = (df[target_col] >= threshold_value).sum()
    ax.scatter([threshold_value], [current_count], color="red", s=100, zorder=5)
    
    ax.set_xlabel(f"{threshold_col} Threshold", fontsize=12)
    ax.set_ylabel("Number of Passing Molecules", fontsize=12)
    ax.set_title(f"Impact of {threshold_col} Threshold", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Show summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Threshold", f"{threshold_value:.2f}")
    with col2:
        st.metric("Passing Molecules", f"{current_count:,}")
    with col3:
        st.metric("Pass Rate", f"{100*current_count/len(df):.1f}%")


def main():
    st.set_page_config(
        page_title="AI-Driven Drug Discovery Pipeline",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Card styling for sections */
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-trained {
        background-color: #10b981;
        color: white;
    }
    
    .status-untrained {
        background-color: #f59e0b;
        color: white;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Info boxes */
    .info-box {
        background: #e0f2fe;
        border-left: 4px solid #0284c7;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Warning boxes */
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Success boxes */
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Header with gradient title
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-title">🧬 AI-Driven Drug Discovery Screening Pipeline</h1>
        <p style="font-size: 1.1rem; color: #64748b; margin-top: 0.5rem;">
            High-throughput virtual screening with ADMET prediction and lead optimization
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None
    if "activity_results" not in st.session_state:
        st.session_state.activity_results = None
    if "selected_hits" not in st.session_state:
        st.session_state.selected_hits = None
    if "admet_results" not in st.session_state:
        st.session_state.admet_results = None
    if "cluster_data" not in st.session_state:
        st.session_state.cluster_data = None
    if "docking_results" not in st.session_state:
        st.session_state.docking_results = {}
    if "sar_results" not in st.session_state:
        st.session_state.sar_results = None
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = None
    if "selected_target" not in st.session_state:
        st.session_state.selected_target = "HIV-1 Protease"  # Default
    
    # Load models
    with st.spinner("Loading models..."):
        reg = load_models()
    
    # ========== SCREEN 1: TARGET SELECTOR UI ==========
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0;">🎯 Screen 1: Select Target Protein</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Define the biological target (enzyme or receptor) to screen molecules against. 
            The target determines which activity model will be used for screening.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # List available targets
    available_targets = list_available_targets()
    available_target_names = [t["target_name"] for t in available_targets]
    
    # Add predefined targets that aren't trained yet
    predefined_names = list(PREDEFINED_TARGETS.keys())
    all_target_options = list(set(available_target_names + predefined_names))
    all_target_options.sort()
    
    # Target selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_selection_mode = st.radio(
            "Target selection:",
            ["Predefined target", "Custom CHEMBL target"],
            horizontal=True,
            key="target_mode"
        )
        
        if target_selection_mode == "Predefined target":
            if len(available_target_names) > 0:
                selected_target = st.selectbox(
                    "Select target:",
                    options=all_target_options,
                    index=0 if "HIV-1 Protease" in all_target_options else 0,
                    help="Select a target protein for activity screening"
                )
            else:
                st.warning("⚠️ No trained target models found. Please download and train a target first.")
                selected_target = st.selectbox(
                    "Select target to train:",
                    options=predefined_names,
                    help="Select a target to download and train"
                )
        else:
            custom_chembl_id = st.text_input(
                "Enter CHEMBL target ID:",
                placeholder="CHEMBL4301553",
                help="Enter a ChEMBL target ID (e.g., CHEMBL4301553 for SARS-CoV-2 Mpro)"
            )
            if custom_chembl_id:
                selected_target = f"Custom: {custom_chembl_id}"
            else:
                selected_target = None
    
    with col2:
        st.write("")  # Spacing
        download_train_button = st.button(
            "📥 Download Dataset & Train Model",
            type="primary",
            use_container_width=True
        )
    
    # Check if target is available with visual status badges
    target_available = False
    if selected_target:
        if selected_target in available_target_names:
            target_available = True
            st.session_state.selected_target = selected_target
            target_info = next(t for t in available_targets if t["target_name"] == selected_target)
            st.markdown(f"""
            <div style="background: #d1fae5; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981; margin: 1rem 0;">
                <span class="status-badge status-trained">✓ Model Trained</span>
                <p style="margin: 0.5rem 0 0 0; color: #065f46;">
                    <strong>{selected_target}</strong> model is loaded and ready ({target_info['chembl_id']})
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif selected_target.startswith("Custom: "):
            chembl_id = selected_target.replace("Custom: ", "")
            st.markdown(f"""
            <div style="background: #e0f2fe; padding: 1rem; border-radius: 8px; border-left: 4px solid #0284c7; margin: 1rem 0;">
                <span class="status-badge status-untrained">⚠ Not Trained</span>
                <p style="margin: 0.5rem 0 0 0; color: #0c4a6e;">
                    Ready to download and train model for <strong>{chembl_id}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Predefined but not trained
            target_info = PREDEFINED_TARGETS.get(selected_target)
            if target_info:
                st.markdown(f"""
                <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; border-left: 4px solid #f59e0b; margin: 1rem 0;">
                    <span class="status-badge status-untrained">⚠ Not Trained</span>
                    <p style="margin: 0.5rem 0 0 0; color: #92400e;">
                        <strong>{selected_target}</strong> model not yet trained. Click 'Download Dataset & Train Model' to train it.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Download and train target
    if download_train_button:
        if not selected_target:
            st.error("❌ Please select a target first.")
        else:
            with st.spinner("Downloading ChEMBL data and training model (this may take a few minutes)..."):
                try:
                    if selected_target.startswith("Custom: "):
                        chembl_id = selected_target.replace("Custom: ", "")
                        target_name = f"Custom {chembl_id}"
                    else:
                        target_info = PREDEFINED_TARGETS.get(selected_target)
                        if not target_info:
                            st.error(f"❌ Unknown target: {selected_target}")
                        else:
                            chembl_id = target_info["chembl_id"]
                            target_name = selected_target
                    
                    model_path, metadata = download_and_train_target(
                        target_name=target_name,
                        chembl_id=chembl_id if selected_target.startswith("Custom: ") else None,
                        max_records=2000,
                        ic50_threshold_nM=1000.0
                    )
                    
                    st.success(f"✅ Model trained successfully!")
                    st.json(metadata)
                    
                    # Reload models
                    reg.load()
                    st.session_state.selected_target = target_name
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.exception(e)
    
    st.divider()
    
    # Check if selected target model is available
    if selected_target and selected_target not in available_target_names and not selected_target.startswith("Custom: "):
        st.warning("⚠️ Selected target model not available. Please train it first using the button above.")
        target_model = None
    else:
        target_model = reg.get_target_model(st.session_state.selected_target) if st.session_state.selected_target else None
    
    # Check model availability
    target_available = target_model is not None
    admet_available = any(reg.has_model(task) for task in ADMET_TASKS.keys())
    
    if not target_available:
        st.error(f"❌ {st.session_state.selected_target} model not found! Please train it first using the target selector above.")
        # Don't return, let them use the target selector
    
    if not admet_available:
        st.warning("⚠️ No ADMET models found. ADMET evaluation will be skipped.")
        with st.expander("📚 How to Train ADMET Models"):
            st.markdown("""
            **Option 1: Train All Models at Once (Recommended)**
            ```bash
            python train_all_admet.py
            ```
            This will train all 11 ADMET tasks automatically (~20-40 minutes).
            
            **Option 2: Train Models Individually**
            ```bash
            python train_admet.py --task hERG
            python train_admet.py --task AMES
            python train_admet.py --task DILI
            # ... etc for all tasks
            ```
            
            **Required First:**
            ```bash
            python main.py  # Train HIV-1 protease model
            ```
            
            See `SETUP_MODELS.md` for detailed instructions.
            """)
    
    # Main workflow tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📤 Step 1: Upload Library",
        "🎯 Step 2: Activity Screening",
        "🔬 Step 3: ADMET Evaluation",
        "🔧 Step 4: Lead Optimization",
        "📊 Lead Ranking Dashboard",
        "🎓 Model Training",
        "⚙️ Advanced Mode"
    ])
    
    # ========== TAB 1: Screen 2 - Upload Library ==========
    with tab1:
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #3b82f6; margin-bottom: 2rem;">
            <h2 style="margin: 0 0 0.5rem 0; color: #1e293b;">📤 Screen 2: Upload Molecular Library</h2>
            <p style="margin: 0; color: #64748b;">
                Upload the molecules you want to test. These can come from internal libraries, vendor databases, 
                virtual designs, or generative models. The system will validate and prepare them for screening.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # How to use this app section
        with st.expander("📖 Workflow Overview", expanded=False):
            st.markdown("""
            **Complete Pipeline Flow:**
            1. **Screen 1: Select Target** → Choose biological target (already done above)
            2. **Screen 2: Upload Library** → Upload CSV with SMILES strings
            3. **Screen 3: Activity Screening** → Predict target binding probability (p_active)
            4. **Screen 4: ADMET Evaluation** → Evaluate toxicity, absorption, distribution, metabolism, excretion
            5. **Screen 5: Lead Ranking** → Composite scoring and prioritization
            6. **Visual Analytics** → UMAP clusters and distribution analysis
            
            **Tip:** Start with a small library (100-1000 molecules) to test the pipeline, then scale up.
            """)
        
        st.info("💡 **Purpose:** Upload your molecular library as a CSV file. The system will auto-detect SMILES columns and validate molecules.")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            help="CSV file with SMILES column (auto-detected)"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Auto-detect SMILES column
                detected = auto_detect_csv_format(df)
                smiles_col = detected.get("smiles")
                
                if not smiles_col:
                    st.error("❌ Could not find SMILES column. Please ensure your CSV has a column named 'smiles', 'SMILES', 'canonical_smiles', 'mol', or 'molecule'")
                    st.write("**Available columns:**", list(df.columns))
                    
                    # Allow manual selection
                    manual_smiles_col = st.selectbox("Select SMILES column manually:", df.columns)
                    if manual_smiles_col:
                        df = df.rename(columns={manual_smiles_col: "smiles"})
                        smiles_col = "smiles"
                else:
                    # Rename to standard 'smiles' column
                    if smiles_col != "smiles":
                        df = df.rename(columns={smiles_col: "smiles"})
                    st.success(f"✅ Auto-detected SMILES column: '{smiles_col}'")
                
                if "smiles" in df.columns:
                    st.success(f"✅ Loaded {len(df)} molecules")
                    st.session_state.uploaded_df = df
                    
                    # Show preview
                    with st.expander("📋 Preview uploaded data", expanded=False):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    # Show column info
                    st.info(f"**Columns:** {', '.join(df.columns)}")
                    
                    if st.button("Clear Upload", type="secondary"):
                        st.session_state.uploaded_df = None
                        st.session_state.activity_results = None
                        st.session_state.selected_hits = None
                        st.session_state.admet_results = None
                        st.rerun()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    # ========== TAB 2: Activity Screening ==========
    with tab2:
        st.header("🎯 Activity Screening")
        target_name = st.session_state.get("selected_target", "Target")
        st.info(f"💡 **This step scores molecules for {target_name} activity** using a trained machine learning model. Molecules with higher p_active values are more likely to be active against the target.")
        
        if st.session_state.uploaded_df is None:
            st.info("👆 Please upload a library in Step 1 first.")
        else:
            df = st.session_state.uploaded_df.copy()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**Library size:** {len(df)} molecules")
            with col2:
                if st.button("Run Activity Screening", type="primary"):
                    with st.spinner("Featurizing SMILES and predicting activity..."):
                        # Featurize
                        smiles_list = df["smiles"].astype(str).tolist()
                        X, idx, valid_smiles = featurize_smiles_batch(smiles_list)
                        
                        # Predict target activity
                        target_model = reg.get_target_model(st.session_state.selected_target)
                        if target_model is None:
                            st.error(f"❌ {st.session_state.selected_target} model not available. Please train it first.")
                            st.stop()
                        p_active = predict_target_activity_batch(X, target_model)
                        
                        # Create results dataframe
                        results_df = df.iloc[idx].copy()
                        results_df["hiv1_p_active"] = p_active
                        results_df = results_df.reset_index(drop=True)
                        
                        st.session_state.activity_results = results_df
                        st.success(f"✅ Screened {len(results_df)} valid molecules")
                        st.rerun()
            
            if st.session_state.activity_results is not None:
                results_df = st.session_state.activity_results.copy()
                
                # Threshold filter
                st.subheader("Filter Active Hits")
                target_name = st.session_state.get("selected_target", "Target")
                
                col_thresh1, col_thresh2 = st.columns([2, 1])
                with col_thresh1:
                    threshold = st.slider(
                        f"{target_name} p_active threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.65,
                        step=0.05,
                        help="Molecules with p_active above this threshold are considered 'active'. Recommended: 0.6-0.7 for balanced sensitivity/specificity.",
                        key="activity_threshold"
                    )
                
                with col_thresh2:
                    show_impact = st.checkbox("Show threshold impact", value=True, key="show_threshold_impact")
                    with st.expander("❓ What does this threshold mean?"):
                        st.markdown(parameter_explainer("p_active_threshold", threshold))
                
                # Show threshold impact visualization
                if show_impact:
                    st.subheader("📊 Threshold Impact")
                    show_threshold_impact(results_df, "p_active", threshold, target_col="hiv1_p_active")
                    st.divider()
                
                # Filter hits
                hits_df = results_df[results_df["hiv1_p_active"] >= threshold].copy()
                hits_df = hits_df.sort_values("hiv1_p_active", ascending=False)
                
                # Compute clustering for active hits
                if len(hits_df) > 0:
                    with st.spinner("Computing molecular similarity clusters..."):
                        smiles_list = hits_df["smiles"].astype(str).tolist()
                        cluster_labels, umap_coords, valid_cluster_idx = cluster_molecules(
                            smiles_list,
                            tanimoto_threshold=DEFAULT_TANIMOTO_THRESHOLD
                        )
                        
                        # Initialize cluster columns with NaN
                        hits_df = hits_df.reset_index(drop=True)  # Ensure integer index
                        hits_df["cluster_id"] = np.nan
                        hits_df["umap_x"] = np.nan
                        hits_df["umap_y"] = np.nan
                        
                        # Add cluster_id and UMAP coordinates only for successfully clustered molecules
                        for i, orig_idx in enumerate(valid_cluster_idx):
                            hits_df.loc[orig_idx, "cluster_id"] = cluster_labels[i]
                            hits_df.loc[orig_idx, "umap_x"] = umap_coords[i, 0]
                            hits_df.loc[orig_idx, "umap_y"] = umap_coords[i, 1]
                        
                        # Store cluster data for visualization
                        st.session_state.cluster_data = {
                            "cluster_labels": cluster_labels,
                            "umap_coords": umap_coords,
                            "valid_indices": valid_cluster_idx
                        }
                else:
                    st.session_state.cluster_data = None
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Screened", len(results_df))
                with col2:
                    st.metric("Valid SMILES", len(results_df))
                with col3:
                    st.metric("Active Hits", len(hits_df))
                with col4:
                    if len(hits_df) > 0:
                        st.metric("Avg p_active", f"{hits_df['hiv1_p_active'].mean():.3f}")
                
                # Display hits table
                st.subheader("Active Hits Table")
                
                # Select columns to display
                display_cols = ["smiles", "hiv1_p_active"]
                if "cluster_id" in hits_df.columns:
                    display_cols.insert(1, "cluster_id")
                if "compound_id" in hits_df.columns:
                    display_cols.insert(0, "compound_id")
                elif "id" in hits_df.columns:
                    display_cols.insert(0, "id")
                elif "name" in hits_df.columns:
                    display_cols.insert(0, "name")
                
                # Format for display
                display_df = hits_df[display_cols].copy()
                display_df["hiv1_p_active"] = display_df["hiv1_p_active"].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
                
                # Store filtered hits
                st.session_state.selected_hits = hits_df
                
                # Download activity results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Activity Results CSV",
                    data=csv,
                    file_name="activity_screening_results.csv",
                    mime="text/csv"
                )
    
    # ========== TAB 3: ADMET Evaluation ==========
    with tab3:
        st.header("🔬 ADMET Evaluation")
        st.info("💡 **This step evaluates ADMET properties and computes a composite developability score.** Properties include toxicity (hERG, AMES, DILI), absorption (Caco-2, HIA), distribution (BBB, PPBR), metabolism (CYP), and excretion (half-life).")
        
        if st.session_state.selected_hits is None or len(st.session_state.selected_hits) == 0:
            st.info("👆 Please run activity screening and filter hits in Step 2 first.")
        else:
            hits_df = st.session_state.selected_hits.copy()
            
            st.write(f"**Evaluating {len(hits_df)} active hits**")
            
            # Molecule selection
            st.subheader("Select Molecules for ADMET")
            
            selection_mode = st.radio(
                "Selection mode:",
                ["All active hits (automatic)", "Manual selection"],
                horizontal=True
            )
            
            if selection_mode == "Manual selection":
                # Multi-select from hits
                hit_options = [
                    f"{row.get('compound_id', row.get('id', row.get('name', f'Hit_{idx}')))}, "
                    f"p_active={row['hiv1_p_active']:.3f}"
                    for idx, row in hits_df.iterrows()
                ]
                selected_indices = st.multiselect(
                    "Choose molecules to evaluate:",
                    options=list(range(len(hits_df))),
                    format_func=lambda x: hit_options[x],
                    default=list(range(min(10, len(hits_df))))  # Default to first 10
                )
                molecules_to_evaluate = hits_df.iloc[selected_indices].copy()
            else:
                # Use all active hits
                molecules_to_evaluate = hits_df.copy()
            
            st.write(f"**{len(molecules_to_evaluate)} molecules selected for ADMET evaluation**")
            
            if st.button("Run ADMET Panel", type="primary"):
                if not admet_available:
                    st.error("No ADMET models available. Please train ADMET models first.")
                else:
                    with st.spinner("Running ADMET predictions..."):
                        # Featurize selected molecules
                        smiles_list = molecules_to_evaluate["smiles"].astype(str).tolist()
                        X_admet, idx_admet, valid_smiles_admet = featurize_smiles_batch(smiles_list)
                        
                        # Align molecules with valid featurization
                        molecules_valid = molecules_to_evaluate.iloc[idx_admet].copy().reset_index(drop=True)
                        
                        # Initialize results dataframe
                        admet_results = molecules_valid.copy()
                        
                        # Run ADMET predictions
                        admet_cols = {}
                        for task_key in ADMET_TASKS.keys():
                            model = reg.get(task_key)
                            if model is None:
                                continue
                            
                            task_config = ADMET_TASKS[task_key]
                            task_type = task_config["type"]
                            
                            predictions = predict_admet_batch(X_admet, task_key, model, task_type)
                            
                            if task_type == "classification":
                                admet_cols[f"{task_key}_prob"] = predictions
                            else:  # regression
                                admet_cols[f"{task_key}_value"] = predictions
                        
                        # Add ADMET columns
                        for col, values in admet_cols.items():
                            admet_results[col] = values
                        
                        # Compute composite scores
                        composite_scores = []
                        for _, row in admet_results.iterrows():
                            admet_outputs = {}
                            for task_key in ADMET_TASKS.keys():
                                prob_col = f"{task_key}_prob"
                                value_col = f"{task_key}_value"
                                
                                if prob_col in admet_results.columns:
                                    admet_outputs[task_key] = {"prob": row[prob_col]}
                                elif value_col in admet_results.columns:
                                    admet_outputs[task_key] = {"value": row[value_col]}
                            
                            score = composite_score(row["hiv1_p_active"], admet_outputs)
                            composite_scores.append(score)
                        
                        admet_results["composite_score"] = composite_scores
                        admet_results = admet_results.sort_values("composite_score", ascending=False)
                        
                        st.session_state.admet_results = admet_results
                        st.success(f"✅ ADMET evaluation complete for {len(admet_results)} molecules")
                        st.rerun()
            
            # Display ADMET results
            if st.session_state.admet_results is not None:
                admet_results = st.session_state.admet_results.copy()
                
                st.subheader("📊 ADMET Results")
                
                # ADMET property explanations
                with st.expander("ℹ️ What do these ADMET properties mean?", expanded=False):
                    st.markdown("""
                    **Toxicity Properties (Lower is Better):**
                    - **hERG_prob**: Probability of blocking hERG ion channel (cardiac risk)
                    - **AMES_prob**: Probability of mutagenicity (DNA damage risk)
                    - **DILI_prob**: Probability of drug-induced liver injury
                    - **CYP3A4_Veith_prob / CYP2D6_prob**: Probability of enzyme inhibition (drug-drug interaction risk)
                    
                    **Absorption/Distribution (Higher is Better):**
                    - **HIA_Hou_prob**: Human intestinal absorption probability
                    - **BBB_Martins_prob**: Blood-brain barrier penetration probability
                    - **caco2_wang_value**: Caco-2 cell permeability (log units, > -5.15 is good)
                    
                    **Other Properties:**
                    - **LD50_Zhu_value**: Toxicity dose (higher = less toxic)
                    - **PPBR_AZ_value**: Plasma protein binding (0.3-0.9 ideal)
                    - **Half_Life_Obach_value**: Half-life in hours (1-24h ideal)
                    """)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Molecules Evaluated", len(admet_results))
                with col2:
                    st.metric("Avg Composite Score", f"{admet_results['composite_score'].mean():.3f}")
                with col3:
                    st.metric("Top Score", f"{admet_results['composite_score'].max():.3f}")
                
                # Results tabs
                results_tab1, results_tab2, results_tab3 = st.tabs([
                    "📋 Results Table", 
                    "📈 Summary Statistics",
                    "🔬 Chemotype Clustering"
                ])
                
                with results_tab1:
                    # Display full results table
                    st.caption("💡 See 'What do these ADMET properties mean?' above for property explanations")
                    st.dataframe(
                        admet_results,
                        use_container_width=True,
                        height=500
                    )
                    
                    # Download button
                    csv = admet_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download ADMET Results CSV",
                        data=csv,
                        file_name="admet_results.csv",
                        mime="text/csv"
                    )
                
                with results_tab2:
                    # Summary statistics
                    st.write("**HIV-1 p_active:**")
                    st.write(admet_results["hiv1_p_active"].describe())
                    
                    st.write("**Composite Score:**")
                    st.write(admet_results["composite_score"].describe())
                    
                    # ADMET property summaries
                    st.write("**ADMET Properties:**")
                    admet_summary = {}
                    for task_key in ADMET_TASKS.keys():
                        prob_col = f"{task_key}_prob"
                        value_col = f"{task_key}_value"
                        
                        if prob_col in admet_results.columns:
                            admet_summary[task_key] = admet_results[prob_col].describe()
                        elif value_col in admet_results.columns:
                            admet_summary[task_key] = admet_results[value_col].describe()
                    
                    for task, stats in admet_summary.items():
                        description = get_admet_property_description(task)
                        with st.expander(f"{task} - {description.split('.')[0]}"):
                            st.caption(description)
                            st.write(stats)
                
                with results_tab3:
                    # Chemotype Clustering Section
                    st.subheader("🔬 Chemotype Clustering")
                    
                    if st.session_state.cluster_data is None or len(admet_results) == 0:
                        st.info("Clustering data not available. Please run activity screening first.")
                    else:
                        # Check if cluster_id exists in admet_results
                        if "cluster_id" not in admet_results.columns:
                            # Try to merge cluster data from hits_df
                            if "smiles" in admet_results.columns:
                                # Merge cluster info from selected_hits if available
                                if st.session_state.selected_hits is not None:
                                    cluster_info = st.session_state.selected_hits[["smiles", "cluster_id", "umap_x", "umap_y"]].copy()
                                    admet_results = admet_results.merge(
                                        cluster_info,
                                        on="smiles",
                                        how="left"
                                    )
                        
                        if "cluster_id" in admet_results.columns and "umap_x" in admet_results.columns:
                            # Filter out molecules without cluster data
                            clustered_results = admet_results.dropna(subset=["cluster_id", "umap_x", "umap_y"])
                            
                            if len(clustered_results) == 0:
                                st.warning("No molecules have cluster information. Please ensure clustering was computed in Step 2.")
                            else:
                                # Cluster summary statistics
                                cluster_summary = clustered_results["cluster_id"].value_counts().sort_index()
                            
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    n_clusters = clustered_results["cluster_id"].nunique()
                                    st.metric("Number of Clusters", n_clusters)
                                with col2:
                                    largest_cluster_size = cluster_summary.max()
                                    st.metric("Largest Cluster Size", largest_cluster_size)
                                with col3:
                                    avg_cluster_size = cluster_summary.mean()
                                    st.metric("Avg Cluster Size", f"{avg_cluster_size:.1f}")
                                with col4:
                                    singletons = (cluster_summary == 1).sum()
                                    st.metric("Singleton Clusters", singletons)
                                
                                # Cluster size distribution
                                st.write("**Cluster Size Distribution:**")
                                cluster_size_df = pd.DataFrame({
                                    "Cluster ID": cluster_summary.index,
                                    "Size": cluster_summary.values
                                }).sort_values("Size", ascending=False)
                                st.dataframe(cluster_size_df, use_container_width=True, height=300)
                                
                                # UMAP visualization with professional styling
                                st.subheader("📊 UMAP 2D Embedding")
                                st.markdown("""
                                <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #0284c7;">
                                    <p style="margin: 0; color: #0c4a6e;">
                                        <strong>What is this visualization?</strong><br>
                                        This UMAP projection clusters molecules by <strong>chemical similarity</strong>. 
                                        Each point represents a molecule; colors indicate clusters. 
                                        Clusters represent distinct <strong>chemotypes</strong> that can be explored independently.
                                        Molecules close together are chemically similar.
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Create professional scatter plot
                                fig, ax = plt.subplots(figsize=(12, 10))
                                
                                # Get unique clusters and assign colors
                                unique_clusters = sorted(clustered_results["cluster_id"].unique())
                                n_clusters_plot = len(unique_clusters)
                                
                                # Color by p_active for better visualization (more informative than cluster colors)
                                if "hiv1_p_active" in clustered_results.columns:
                                    scatter = ax.scatter(
                                        clustered_results["umap_x"],
                                        clustered_results["umap_y"],
                                        c=clustered_results["hiv1_p_active"],
                                        cmap="viridis",
                                        s=120,
                                        alpha=0.7,
                                        edgecolors='white',
                                        linewidths=1.0
                                    )
                                    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
                                    cbar.set_label("p_active (Activity Probability)", fontsize=12, fontweight='bold')
                                    cbar.ax.tick_params(labelsize=10)
                                else:
                                    # Fallback to cluster colors
                                    colors = plt.cm.tab20(np.linspace(0, 1, min(n_clusters_plot, 20)))
                                    if n_clusters_plot > 20:
                                        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters_plot))
                                    
                                    for i, cluster_id in enumerate(unique_clusters):
                                        cluster_data = clustered_results[clustered_results["cluster_id"] == cluster_id]
                                        color = colors[i % len(colors)]
                                        ax.scatter(
                                            cluster_data["umap_x"],
                                            cluster_data["umap_y"],
                                            c=[color],
                                            label=f"Cluster {int(cluster_id)} (n={len(cluster_data)})",
                                            alpha=0.7,
                                            s=120,
                                            edgecolors='white',
                                            linewidths=1.0
                                        )
                                    
                                    if n_clusters_plot <= 20:
                                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
                                
                                ax.set_xlabel("UMAP Dimension 1", fontsize=13, fontweight='bold')
                                ax.set_ylabel("UMAP Dimension 2", fontsize=13, fontweight='bold')
                                ax.set_title("Molecular Similarity Clustering (UMAP Projection)", fontsize=15, fontweight='bold', pad=20)
                                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Show cluster details table
                                st.subheader("Cluster Details")
                                cluster_details = []
                                for cluster_id in unique_clusters:
                                    cluster_mols = clustered_results[clustered_results["cluster_id"] == cluster_id]
                                    cluster_details.append({
                                        "Cluster ID": int(cluster_id),
                                        "Size": len(cluster_mols),
                                        "Avg p_active": cluster_mols["hiv1_p_active"].mean(),
                                        "Avg Composite Score": cluster_mols["composite_score"].mean() if "composite_score" in cluster_mols.columns else None
                                    })
                                
                                cluster_details_df = pd.DataFrame(cluster_details)
                                st.dataframe(cluster_details_df, use_container_width=True)
                        else:
                            st.warning("Cluster information not available in ADMET results. Please ensure clustering was computed in Step 2.")
                
                # ========== Docking Section ==========
                st.divider()
                target_name = st.session_state.get("selected_target", "Target")
                st.subheader(f"🔬 Docking ({target_name})")
                
                if st.session_state.admet_results is not None and len(st.session_state.admet_results) > 0:
                    admet_df = st.session_state.admet_results.copy()
                    
                    # Molecule selection for docking
                    if "smiles" in admet_df.columns:
                        # Create selection options
                        mol_options = []
                        for idx, row in admet_df.iterrows():
                            mol_id = row.get("compound_id", row.get("id", row.get("name", f"Hit_{idx}")))
                            p_active = row.get("hiv1_p_active", "N/A")
                            mol_options.append(f"{mol_id} (p_active={p_active:.3f})" if isinstance(p_active, (int, float)) else f"{mol_id}")
                        
                        selected_mol_idx = st.selectbox(
                            "Select molecule for docking:",
                            options=list(range(len(admet_df))),
                            format_func=lambda x: mol_options[x] if x < len(mol_options) else f"Row {x}"
                        )
                        
                        selected_smiles = admet_df.iloc[selected_mol_idx]["smiles"]
                        selected_mol_id = admet_df.iloc[selected_mol_idx].get("compound_id", admet_df.iloc[selected_mol_idx].get("id", f"mol_{selected_mol_idx}"))
                        
                        # Check Vina and Open Babel availability
                        vina_available, vina_msg = check_vina_available()
                        obabel_available, obabel_msg = check_obabel_available()
                        
                        if not vina_available or not obabel_available:
                            if not vina_available:
                                st.warning(f"⚠️ {vina_msg}")
                            if not obabel_available:
                                st.warning(f"⚠️ {obabel_msg}")
                            with st.expander("📚 Setup Instructions", expanded=True):
                                st.markdown("""
                                **To enable docking, you need:**
                                
                                1. **AutoDock Vina**
                                   - Download: https://vina.scripps.edu/downloads/
                                   - Install and add to PATH
                                   - Test: `vina --help` should work in terminal
                                
                                2. **Open Babel** (for ligand preparation)
                                   - Install: `conda install -c conda-forge openbabel` or `brew install openbabel`
                                   - Test: `obabel --help` should work in terminal
                                
                                3. **Receptor PDBQT file**
                                   - Prepare HIV-1 protease structure as PDBQT
                                   - Place at: `data/hiv1_protease/receptor.pdbqt`
                                   - Use AutoDock Tools or MGLTools to prepare from PDB
                                
                                4. **Vina config file** (optional)
                                   - Define search space (center_x, center_y, center_z, size_x, size_y, size_z)
                                   - Place at: `data/hiv1_protease/vina_config.txt`
                                   - Format example:
                                     ```
                                     center_x = 0.0
                                     center_y = 0.0
                                     center_z = 0.0
                                     size_x = 20
                                     size_y = 20
                                     size_z = 20
                                     ```
                                
                                **Note:** If Open Babel is not available, ligand preparation will fail with instructions.
                                """)
                        else:
                            st.success(f"✅ {vina_msg}")
                            st.success(f"✅ {obabel_msg}")
                            
                            # Receptor and config paths
                            receptor_path = st.text_input(
                                "Receptor PDBQT path:",
                                value=DEFAULT_RECEPTOR_PATH,
                                help="Path to prepared receptor PDBQT file"
                            )
                            
                            config_path = st.text_input(
                                "Vina config path (optional):",
                                value=DEFAULT_CONFIG_PATH if os.path.exists(DEFAULT_CONFIG_PATH) else "",
                                help="Path to Vina configuration file (optional)"
                            )
                            
                            if st.button("Run Docking", type="primary"):
                                if not os.path.exists(receptor_path):
                                    st.error(f"❌ Receptor file not found: {receptor_path}")
                                else:
                                    with st.spinner("Preparing ligand and running docking (this may take a few minutes)..."):
                                        try:
                                            config = config_path if config_path and os.path.exists(config_path) else None
                                            affinity, ligand_path, output_dir = dock_smiles(
                                                smiles=selected_smiles,
                                                receptor_path=receptor_path,
                                                config_path=config,
                                                ligand_id=str(selected_mol_id)
                                            )
                                            
                                            if affinity is not None:
                                                st.success(f"✅ Docking complete!")
                                                st.metric("Best Binding Affinity", f"{affinity:.2f} kcal/mol")
                                                st.info(f"**Pose files saved to:** `{output_dir}`")
                                                st.write(f"- Ligand PDBQT: `{os.path.join(output_dir, 'ligand.pdbqt')}`")
                                                st.write(f"- Docked poses: `{os.path.join(output_dir, 'ligand_out.pdbqt')}`")
                                                st.write(f"- Vina log: `{os.path.join(output_dir, 'vina_log.txt')}`")
                                                
                                                # Store result
                                                st.session_state.docking_results[str(selected_mol_id)] = {
                                                    "affinity": affinity,
                                                    "output_dir": output_dir,
                                                    "smiles": selected_smiles
                                                }
                                            else:
                                                st.warning("Docking completed but could not parse binding affinity from log file.")
                                        except Exception as e:
                                            st.error(f"❌ Docking failed: {e}")
                                            st.exception(e)
                        
                        # Show previous docking results
                        if st.session_state.docking_results:
                            st.subheader("Previous Docking Results")
                            docking_df = pd.DataFrame([
                                {
                                    "Molecule ID": mol_id,
                                    "Affinity (kcal/mol)": result["affinity"],
                                    "Output Directory": result["output_dir"]
                                }
                                for mol_id, result in st.session_state.docking_results.items()
                            ])
                            st.dataframe(docking_df, use_container_width=True)
                else:
                    st.info("👆 Please run ADMET evaluation first to select molecules for docking.")
    
    # ========== TAB 4: Lead Optimization ==========
    with tab4:
        st.header("🔧 Step 4: Lead Optimization")
        st.info("💡 **This step generates new molecules by mutating top leads** using RDKit transformations (methyl addition/removal, halogen replacement, BRICS fragmentation). Generated candidates are rescored with the same HIV-1 and ADMET models.")
        
        if st.session_state.admet_results is None or len(st.session_state.admet_results) == 0:
            st.info("👆 Please run ADMET evaluation in Step 3 first.")
        else:
            hits_df = st.session_state.admet_results.copy()
            
            st.write(f"**Starting from {len(hits_df)} ADMET-evaluated hits**")
            
            # Controls section
            st.subheader("⚙️ Optimization Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                n_parents = st.slider(
                    "Number of parent molecules",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Number of top-scoring hits to use as parents for mutation"
                )
                n_children_per_parent = st.slider(
                    "Children per parent",
                    min_value=5,
                    max_value=30,
                    value=10,
                    step=5,
                    help="Target number of mutated variants per parent"
                )
            
            with col2:
                min_hiv1_p_active = st.slider(
                    "Min HIV-1 p_active",
                    min_value=0.5,
                    max_value=0.95,
                    value=0.65,
                    step=0.05,
                    help="Minimum HIV-1 activity probability for optimized leads. Recommended: 0.6-0.7"
                )
                with st.expander("❓ What does p_active threshold mean?"):
                    st.markdown(parameter_explainer("p_active_threshold", min_hiv1_p_active))
                
                max_hERG_prob = st.slider(
                    "Max hERG probability",
                    min_value=0.3,
                    max_value=0.9,
                    value=0.6,
                    step=0.05,
                    help="Maximum hERG probability (cardiac risk threshold). Lower is safer. Recommended: 0.5-0.7"
                )
                with st.expander("❓ What does hERG threshold mean?"):
                    st.markdown(parameter_explainer("herg_threshold", max_hERG_prob))
            
            col3, col4 = st.columns(2)
            with col3:
                enforce_lipinski = st.checkbox(
                    "Enforce Lipinski drug-likeness filters",
                    value=True,
                    help="Filter out molecules that violate Lipinski's Rule of Five"
                )
                enforce_medchem = st.checkbox(
                    "Enforce medicinal chemistry filters",
                    value=True,
                    help="Filter out PAINS, structural alerts, and problematic substructures"
                )
                check_sa_score = st.checkbox(
                    "Check synthetic accessibility (SA Score)",
                    value=False,
                    help="Filter molecules by synthetic accessibility score (lower = easier to synthesize)"
                )
                max_sa_score = None
                if check_sa_score:
                    max_sa_score = st.slider(
                        "Max SA Score",
                        min_value=3.0,
                        max_value=10.0,
                        value=6.0,
                        step=0.5,
                        help="Maximum synthetic accessibility score (1=easy, 10=hard). Recommended: 4.0-7.0"
                    )
                    with st.expander("❓ What does SA Score mean?"):
                        st.markdown(parameter_explainer("sa_score", max_sa_score))
            
            with col4:
                min_similarity = st.slider(
                    "Min similarity to parent",
                    min_value=0.0,
                    max_value=0.9,
                    value=0.4,
                    step=0.1,
                    help="Minimum Tanimoto similarity to parent (too low = random scaffold)"
                )
                max_similarity = st.slider(
                    "Max similarity to parent",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.9,
                    step=0.1,
                    help="Maximum Tanimoto similarity to parent (too high = trivial changes)"
                )
            
            if st.button("Run Lead Optimization", type="primary"):
                with st.spinner("Generating mutations and scoring candidates (this may take a few minutes)..."):
                    try:
                        # Get models
                        target_model = reg.get_target_model(st.session_state.selected_target)
                        admet_models = {
                            task: reg.get(task)
                            for task in ADMET_TASKS.keys()
                            if reg.has_model(task)
                        }
                        
                        if target_model is None:
                            st.error(f"❌ {st.session_state.selected_target} model not available. Please train it first.")
                        else:
                            # Run optimization
                            optimized_df = optimize_leads(
                                hits_df=hits_df,
                                hiv_model=target_model,
                                admet_models=admet_models,
                                composite_fn=composite_score,
                                n_parents=n_parents,
                                n_children_per_parent=n_children_per_parent,
                                min_hiv1_p_active=min_hiv1_p_active,
                                max_hERG_prob=max_hERG_prob,
                                enforce_lipinski=enforce_lipinski,
                                enforce_medchem_filters=enforce_medchem,
                                check_sa_score=check_sa_score,
                                max_sa_score=max_sa_score if check_sa_score else 6.0,
                                min_similarity=min_similarity,
                                max_similarity=max_similarity
                            )
                            
                            st.session_state.optimization_results = optimized_df
                            st.success(f"✅ Generated {len(optimized_df)} optimized lead candidates")
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Optimization failed: {e}")
                        st.exception(e)
            
            # Display results
            if st.session_state.optimization_results is not None:
                optimized_df = st.session_state.optimization_results.copy()
                
                if len(optimized_df) == 0:
                    st.warning("⚠️ No optimized leads generated. Try adjusting parameters (e.g., lower thresholds or disable Lipinski filter).")
                else:
                    st.subheader("📊 Optimization Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Parents Used", n_parents)
                    with col2:
                        st.metric("Candidates Generated", len(optimized_df))
                    with col3:
                        st.metric("Passing Filters", len(optimized_df))
                    with col4:
                        if len(optimized_df) > 0:
                            st.metric("Best Composite Score", f"{optimized_df['composite_score'].max():.3f}")
                    
                    # Visualization
                    if len(optimized_df) > 0 and "hERG_prob" in optimized_df.columns:
                        st.subheader("📈 hERG vs Composite Score")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Color by parent if available
                        if "parent_index" in optimized_df.columns:
                            scatter = ax.scatter(
                                optimized_df["hERG_prob"],
                                optimized_df["composite_score"],
                                c=optimized_df["parent_index"],
                                cmap="tab20",
                                alpha=0.6,
                                s=50,
                                edgecolors='black',
                                linewidths=0.5
                            )
                            plt.colorbar(scatter, ax=ax, label="Parent Index")
                        else:
                            ax.scatter(
                                optimized_df["hERG_prob"],
                                optimized_df["composite_score"],
                                alpha=0.6,
                                s=50,
                                edgecolors='black',
                                linewidths=0.5
                            )
                        
                        ax.set_xlabel("hERG Probability (cardiac risk)", fontsize=12)
                        ax.set_ylabel("Composite Score", fontsize=12)
                        ax.set_title("Optimized Leads: hERG Risk vs Composite Score", fontsize=14, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Results table
                    st.subheader("📋 Optimized Leads Table")
                    
                    # Add tier column for visual highlighting
                    optimized_df_display = optimized_df.copy()
                    optimized_df_display["tier"] = "Others"
                    if len(optimized_df_display) > 0:
                        optimized_df_display.loc[optimized_df_display.index[:5], "tier"] = "Top 5"
                        if len(optimized_df_display) > 5:
                            optimized_df_display.loc[optimized_df_display.index[5:20], "tier"] = "Top 20"
                    
                    # Select columns to display
                    display_cols = ["tier", "smiles", "parent_smiles", "hiv1_p_active", "composite_score"]
                    if "parent_index" in optimized_df_display.columns:
                        display_cols.insert(3, "parent_index")
                    if "parent_cluster_id" in optimized_df_display.columns:
                        display_cols.insert(4, "parent_cluster_id")
                    
                    # Add ADMET columns
                    for task_key in ADMET_TASKS.keys():
                        prob_col = f"{task_key}_prob"
                        value_col = f"{task_key}_value"
                        if prob_col in optimized_df_display.columns:
                            display_cols.append(prob_col)
                        elif value_col in optimized_df_display.columns:
                            display_cols.append(value_col)
                    
                    st.dataframe(
                        optimized_df_display[display_cols],
                        use_container_width=True,
                        height=500
                    )
                    
                    # Download button
                    csv = optimized_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Optimized Leads CSV",
                        data=csv,
                        file_name="optimized_leads.csv",
                        mime="text/csv"
                    )
    
    # ========== TAB 5: Visual Analytics ==========
    with tab5:
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #ec4899; margin-bottom: 2rem;">
            <h2 style="margin: 0 0 0.5rem 0; color: #1e293b;">📈 Visual Analytics</h2>
            <p style="margin: 0; color: #64748b;">
                Interactive visualizations including UMAP clusters, distribution plots, and chemotype analysis. 
                Each graph includes interpretive descriptions to help scientists understand the data.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.admet_results is None or len(st.session_state.admet_results) == 0:
            st.info("👆 Please complete ADMET evaluation in Screen 4 first to see visualizations.")
        else:
            leads_df = st.session_state.admet_results.copy()
            
            viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                "📊 UMAP Clustering",
                "📈 Distribution Analysis",
                "🔬 Chemotype Explorer"
            ])
            
            with viz_tab1:
                st.subheader("📊 UMAP 2D Molecular Similarity Projection")
                st.markdown("""
                <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #0284c7;">
                    <p style="margin: 0; color: #0c4a6e;">
                        <strong>Interpretation:</strong> This UMAP projection clusters molecules by <strong>chemical similarity</strong>. 
                        Each point represents a molecule; colors indicate activity (p_active) or clusters. 
                        Clusters represent distinct <strong>chemotypes</strong> that can be explored independently.
                        Molecules close together are chemically similar and may share similar biological activity.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if "umap_x" in leads_df.columns and "umap_y" in leads_df.columns:
                    # Create enhanced UMAP plot
                    fig, ax = plt.subplots(figsize=(14, 10))
                    
                    # Color by p_active (most informative)
                    if "hiv1_p_active" in leads_df.columns:
                        scatter = ax.scatter(
                            leads_df["umap_x"],
                            leads_df["umap_y"],
                            c=leads_df["hiv1_p_active"],
                            cmap="viridis",
                            s=150,
                            alpha=0.8,
                            edgecolors='white',
                            linewidths=1.5
                        )
                        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
                        cbar.set_label("p_active (Activity Probability)", fontsize=13, fontweight='bold')
                        cbar.ax.tick_params(labelsize=11)
                    else:
                        # Fallback to composite score
                        if "composite_score" in leads_df.columns:
                            scatter = ax.scatter(
                                leads_df["umap_x"],
                                leads_df["umap_y"],
                                c=leads_df["composite_score"],
                                cmap="plasma",
                                s=150,
                                alpha=0.8,
                                edgecolors='white',
                                linewidths=1.5
                            )
                            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
                            cbar.set_label("Composite Score", fontsize=13, fontweight='bold')
                            cbar.ax.tick_params(labelsize=11)
                    
                    ax.set_xlabel("UMAP Dimension 1", fontsize=14, fontweight='bold')
                    ax.set_ylabel("UMAP Dimension 2", fontsize=14, fontweight='bold')
                    ax.set_title("Molecular Similarity Clustering (UMAP Projection)", fontsize=16, fontweight='bold', pad=25)
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Cluster statistics if available
                    if "cluster_id" in leads_df.columns:
                        st.subheader("📊 Cluster Statistics")
                        cluster_summary = leads_df["cluster_id"].value_counts()
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Clusters", cluster_summary.nunique())
                        with col2:
                            st.metric("Largest Cluster", cluster_summary.max())
                        with col3:
                            st.metric("Avg Cluster Size", f"{cluster_summary.mean():.1f}")
                        with col4:
                            st.metric("Singleton Clusters", (cluster_summary == 1).sum())
                else:
                    st.info("UMAP coordinates not available. Please run activity screening first.")
            
            with viz_tab2:
                st.subheader("📈 Distribution Analysis")
                st.markdown("""
                <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #0284c7;">
                    <p style="margin: 0; color: #0c4a6e;">
                        <strong>Purpose:</strong> Understand the distribution of key properties across your lead set.
                        These histograms help identify outliers, understand property ranges, and guide filtering decisions.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Property selection
                prop_options = ["hiv1_p_active", "composite_score"]
                for task_key in ADMET_TASKS.keys():
                    if f"{task_key}_prob" in leads_df.columns:
                        prop_options.append(f"{task_key}_prob")
                    elif f"{task_key}_value" in leads_df.columns:
                        prop_options.append(f"{task_key}_value")
                
                selected_prop = st.selectbox("Select property to visualize:", prop_options)
                
                if selected_prop in leads_df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(leads_df[selected_prop].dropna(), bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
                    ax.set_xlabel(selected_prop.replace("_", " ").title(), fontsize=12, fontweight='bold')
                    ax.set_ylabel("Number of Molecules", fontsize=12, fontweight='bold')
                    ax.set_title(f"Distribution of {selected_prop.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add statistics
                    mean_val = leads_df[selected_prop].mean()
                    median_val = leads_df[selected_prop].median()
                    ax.axvline(x=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
                    ax.axvline(x=median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Statistics summary
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{leads_df[selected_prop].mean():.3f}")
                    with col2:
                        st.metric("Median", f"{leads_df[selected_prop].median():.3f}")
                    with col3:
                        st.metric("Std Dev", f"{leads_df[selected_prop].std():.3f}")
                    with col4:
                        st.metric("Range", f"{leads_df[selected_prop].min():.3f} - {leads_df[selected_prop].max():.3f}")
            
            with viz_tab3:
                st.subheader("🔬 Chemotype Explorer")
                st.markdown("""
                <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #0284c7;">
                    <p style="margin: 0; color: #0c4a6e;">
                        <strong>Purpose:</strong> Explore distinct chemotypes (chemical classes) in your lead set.
                        Each cluster represents a different structural scaffold that can be optimized independently.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if "cluster_id" in leads_df.columns:
                    # Cluster analysis
                    cluster_summary = leads_df.groupby("cluster_id").agg({
                        "hiv1_p_active": "mean",
                        "composite_score": "mean",
                        "smiles": "count"
                    }).rename(columns={"smiles": "size"})
                    cluster_summary = cluster_summary.sort_values("composite_score", ascending=False)
                    
                    st.subheader("Top Clusters by Composite Score")
                    st.dataframe(cluster_summary.head(20), use_container_width=True)
                    
                    # Select cluster to explore
                    selected_cluster = st.selectbox(
                        "Select cluster to explore:",
                        options=sorted(leads_df["cluster_id"].dropna().unique()),
                        format_func=lambda x: f"Cluster {int(x)} ({len(leads_df[leads_df['cluster_id']==x])} molecules)"
                    )
                    
                    if selected_cluster is not None:
                        cluster_mols = leads_df[leads_df["cluster_id"] == selected_cluster]
                        st.subheader(f"Cluster {int(selected_cluster)} Details")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Cluster Size", len(cluster_mols))
                        with col2:
                            st.metric("Avg p_active", f"{cluster_mols['hiv1_p_active'].mean():.3f}")
                        with col3:
                            st.metric("Avg Composite", f"{cluster_mols['composite_score'].mean():.3f}")
                        
                        st.dataframe(cluster_mols[["smiles", "hiv1_p_active", "composite_score"]].head(10), use_container_width=True)
                else:
                    st.info("Cluster information not available. Please run activity screening first.")
    
    # ========== TAB 4: Lead Optimization (Advanced Feature) ==========
    # Note: Lead Optimization is kept as tab4 but could be moved to Advanced tab
    # Keeping it here for now as it's a key feature
    
    # ========== TAB 6: Lead Ranking Dashboard (Screen 5) ==========
    # This is now in tab4 based on the tab structure, but let's check the actual tab order
    # The tab order is: tab1, tab2, tab3, tab4, tab5, tab6, tab7
    # So tab4 should be Screen 5: Lead Ranking, but currently it's Lead Optimization
    
    # Let's add Lead Ranking to a different location - it should be after ADMET
    # Actually, looking at the code, tab5 is Visual Analytics, so Lead Ranking should be integrated there or be tab4
    
    # For now, let's keep the structure as is and add Lead Ranking content to tab4 after optimization
                st.markdown("""
                **Toxicity (Lower is Better):**
                - **hERG**: Blocks cardiac ion channel → cardiac risk
                - **AMES**: Mutagenic → DNA damage risk  
                - **DILI**: Drug-induced liver injury risk
                - **CYP3A4/CYP2D6**: Enzyme inhibition → drug-drug interactions
                
                **Absorption/Distribution (Higher is Better):**
                - **HIA**: Human intestinal absorption
                - **BBB**: Blood-brain barrier penetration
                - **Caco-2**: Cell permeability (absorption indicator)
                
                **Other:**
                - **LD50**: Toxicity dose (higher = less toxic)
                - **PPBR**: Plasma protein binding (0.3-0.9 ideal)
                - **Half-Life**: Elimination half-life (1-24h ideal)
                """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_p_active = st.slider(
                    "Min HIV-1 p_active",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )
            
            with col2:
                max_herg = st.slider(
                    "Max hERG probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.05,
                    help="hERG: Probability of blocking hERG ion channel (cardiac risk). Lower is better."
                ) if "hERG_prob" in leads_df.columns else None
            
            with col3:
                min_composite = st.slider(
                    "Min Composite Score",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )
            
            # Apply filters
            filtered_leads = leads_df[
                (leads_df["hiv1_p_active"] >= min_p_active) &
                (leads_df["composite_score"] >= min_composite)
            ].copy()
            
            if max_herg is not None and "hERG_prob" in filtered_leads.columns:
                filtered_leads = filtered_leads[filtered_leads["hERG_prob"] <= max_herg]
            
            # Sort options
            sort_by = st.selectbox(
                "Sort by:",
                options=["composite_score", "hiv1_p_active", "hERG_prob"] if "hERG_prob" in filtered_leads.columns else ["composite_score", "hiv1_p_active"],
                index=0
            )
            sort_ascending = st.checkbox("Ascending order", value=False)
            
            filtered_leads = filtered_leads.sort_values(sort_by, ascending=sort_ascending)
            
            # ========== Auto-Generated Lead Report ==========
            st.divider()
            st.subheader("📄 Auto-Generated Lead Report")
            
            report_n_leads = st.number_input(
                "Number of top leads to analyze:",
                min_value=5,
                max_value=min(50, len(filtered_leads)),
                value=min(10, len(filtered_leads)),
                step=5,
                help="Generate detailed analysis report for top N leads"
            )
            
            if st.button("Generate Report", type="primary"):
                if len(filtered_leads) == 0:
                    st.warning("No leads to analyze. Adjust filters and try again.")
                else:
                    top_leads_for_report = filtered_leads.head(report_n_leads).copy()
                    
                    # Generate report for each lead
                    st.markdown("---")
                    st.markdown(f"## 📊 Lead Ranking Report: Top {len(top_leads_for_report)} Leads")
                    st.caption(f"Generated from {len(filtered_leads)} filtered leads, sorted by {sort_by}")
                    
                    for rank, (idx, lead) in enumerate(top_leads_for_report.iterrows(), 1):
                        analysis = analyze_lead_factors(lead)
                        
                        # Lead header
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.markdown(f"### 🥇 Lead #{rank}")
                            mol_id = lead.get("compound_id", lead.get("id", lead.get("name", f"Lead_{rank}")))
                            st.write(f"**ID:** {mol_id}")
                        with col2:
                            st.metric("Composite Score", f"{lead['composite_score']:.3f}")
                        with col3:
                            st.metric("HIV-1 p_active", f"{lead['hiv1_p_active']:.3f}")
                        
                        # SMILES
                        with st.expander("View SMILES", expanded=False):
                            st.code(lead.get("smiles", "N/A"), language=None)
                        
                        # Key factors
                        col_strength, col_weakness = st.columns(2)
                        
                        with col_strength:
                            st.markdown("#### ✅ Key Strengths")
                            for strength in analysis["strengths"]:
                                st.success(f"• {strength}")
                        
                        with col_weakness:
                            st.markdown("#### ⚠️ Key Weaknesses")
                            for weakness in analysis["weaknesses"]:
                                st.warning(f"• {weakness}")
                        
                        # Evidence table
                        st.markdown("#### 📋 Evidence Summary")
                        evidence_df = pd.DataFrame([
                            {"Property": k, "Value": v}
                            for k, v in analysis["evidence"].items()
                        ])
                        st.dataframe(evidence_df, use_container_width=True, hide_index=True)
                        
                        # Cluster info if available
                        if "cluster_id" in lead and pd.notna(lead["cluster_id"]):
                            st.caption(f"📍 Cluster ID: {int(lead['cluster_id'])}")
                        
                        st.markdown("---")
                    
                    # Summary statistics
                    st.markdown("## 📈 Top Leads Summary Statistics")
                    summary_cols = ["hiv1_p_active", "composite_score"]
                    if "hERG_prob" in top_leads_for_report.columns:
                        summary_cols.append("hERG_prob")
                    if "cluster_id" in top_leads_for_report.columns:
                        summary_cols.append("cluster_id")
                    
                    summary_stats = top_leads_for_report[summary_cols].describe()
                    st.dataframe(summary_stats, use_container_width=True)
                    
                    # Download report
                    report_data = []
                    for rank, (idx, lead) in enumerate(top_leads_for_report.iterrows(), 1):
                        analysis = analyze_lead_factors(lead)
                        report_data.append({
                            "Rank": rank,
                            "SMILES": lead.get("smiles", ""),
                            "Composite_Score": lead.get("composite_score", 0),
                            "HIV1_p_active": lead.get("hiv1_p_active", 0),
                            "Strengths": "; ".join(analysis["strengths"]),
                            "Weaknesses": "; ".join(analysis["weaknesses"]),
                            **analysis["evidence"]
                        })
                    
                    report_df = pd.DataFrame(report_data)
                    report_csv = report_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Lead Report CSV",
                        data=report_csv,
                        file_name=f"lead_report_top_{report_n_leads}.csv",
                        mime="text/csv"
                    )
            
            # Natural Language Query
            st.subheader("💬 Natural Language Query")
            st.caption("💡 Tip: Use terms like 'high hiv activity', 'low herg risk', 'good bbb', 'low dili', etc.")
            nl_query = st.text_input(
                "Enter query (e.g., 'high hiv activity and low herg risk'):",
                placeholder="high hiv activity and low herg risk"
            )
            
            if nl_query:
                nl_filtered, interpretation = apply_nl_query_to_dataframe(filtered_leads, nl_query)
                st.info(interpretation)
                if len(nl_filtered) < len(filtered_leads):
                    filtered_leads = nl_filtered
                    st.success(f"Filtered to {len(filtered_leads)} leads matching query")
            
            # Summary metrics
            st.subheader("📈 Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Leads", len(filtered_leads))
            with col2:
                st.metric("Avg p_active", f"{filtered_leads['hiv1_p_active'].mean():.3f}" if len(filtered_leads) > 0 else "N/A")
            with col3:
                st.metric("Avg Composite", f"{filtered_leads['composite_score'].mean():.3f}" if len(filtered_leads) > 0 else "N/A")
            with col4:
                if "cluster_id" in filtered_leads.columns:
                    n_clusters = filtered_leads["cluster_id"].nunique()
                    st.metric("Clusters", n_clusters)
            
            # Visualization
            if len(filtered_leads) > 0:
                st.subheader("📊 Top Leads Visualization")
                top_n = st.slider("Show top N leads", min_value=5, max_value=min(50, len(filtered_leads)), value=min(20, len(filtered_leads)))
                top_leads = filtered_leads.head(top_n)
                
                # Bar plot
                fig, ax = plt.subplots(figsize=(12, 6))
                x_pos = range(len(top_leads))
                ax.barh(x_pos, top_leads["composite_score"], color='steelblue', alpha=0.7)
                ax.set_yticks(x_pos)
                ax.set_yticklabels([f"Lead {i+1}" for i in range(len(top_leads))], fontsize=8)
                ax.set_xlabel("Composite Score", fontsize=12)
                ax.set_title(f"Top {top_n} Leads by Composite Score", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Leads table
            st.subheader("📋 Ranked Leads Table")
            st.caption("💡 Hover over column names or check the ADMET Property Guide above for explanations")
            
            # Add tier column for visual highlighting
            filtered_leads_display = filtered_leads.copy()
            filtered_leads_display["tier"] = "Others"
            if len(filtered_leads_display) > 0:
                filtered_leads_display.loc[filtered_leads_display.index[:5], "tier"] = "Top 5"
                if len(filtered_leads_display) > 5:
                    filtered_leads_display.loc[filtered_leads_display.index[5:20], "tier"] = "Top 20"
            
            display_cols = ["tier", "smiles", "hiv1_p_active", "composite_score"]
            if "cluster_id" in filtered_leads_display.columns:
                display_cols.insert(2, "cluster_id")
            
            # Add ADMET columns if available
            for task_key in ADMET_TASKS.keys():
                prob_col = f"{task_key}_prob"
                value_col = f"{task_key}_value"
                if prob_col in filtered_leads_display.columns:
                    display_cols.append(prob_col)
                elif value_col in filtered_leads_display.columns:
                    display_cols.append(value_col)
            
            st.dataframe(
                filtered_leads_display[display_cols],
                use_container_width=True,
                height=500
            )
            
            # Download
            csv = filtered_leads.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Leads CSV",
                data=csv,
                file_name="ranked_leads.csv",
                mime="text/csv"
            )
    
    # ========== TAB 6: Model Training ==========
    with tab6:
        st.header("🎓 Model Training")
        st.markdown("""
        **Train your own models by uploading data files.**
        - Upload bioactivity data to train target models
        - Train ADMET models from TDC datasets
        - One-click training for all models
        """)
        
        train_tab1, train_tab2, train_tab3 = st.tabs([
            "📊 Train Target Model (Upload CSV)",
            "🧬 Train ADMET Models",
            "⚙️ Auto-Train All Models"
        ])
        
        with train_tab1:
            st.markdown("""
            **Upload your bioactivity data to train a target model:**
            - CSV file with columns: `smiles`, `activity_value` (IC50 in nM)
            - Or use ChEMBL download (in target selector above)
            """)
            
            uploaded_train_file = st.file_uploader(
                "Upload bioactivity CSV",
                type=["csv"],
                help="CSV with columns: smiles, activity_value (IC50 in nM)",
                key="train_file_uploader"
            )
            
            if uploaded_train_file:
                try:
                    df_preview = pd.read_csv(uploaded_train_file)
                    
                    # Auto-detect format
                    detected = auto_detect_csv_format(df_preview)
                    
                    st.success(f"✅ Loaded {len(df_preview)} molecules")
                    
                    with st.expander("📋 Preview data", expanded=True):
                        st.dataframe(df_preview.head(10), use_container_width=True)
                    
                    if detected:
                        st.info(f"🔍 **Auto-detected:** SMILES='{detected.get('smiles', 'Not found')}', Activity='{detected.get('activity_value', 'Not found')}'")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        target_name = st.text_input("Target name:", value="My Target", key="train_target_name")
                        ic50_threshold = st.slider(
                            "IC50 threshold (nM):",
                            min_value=100.0,
                            max_value=10000.0,
                            value=1000.0,
                            step=100.0,
                            help="Molecules with IC50 below this are considered 'active'",
                            key="train_ic50_threshold"
                        )
                        
                        # Show threshold impact
                        if detected and detected.get("activity_value") in df_preview.columns:
                            active_count = (df_preview[detected["activity_value"]] < ic50_threshold).sum()
                            total = len(df_preview)
                            st.metric(
                                "Threshold Impact",
                                f"{active_count}/{total} active",
                                f"{100*active_count/total:.1f}%"
                            )
                            with st.expander("❓ What does this threshold mean?"):
                                st.markdown(parameter_explainer("ic50_threshold", ic50_threshold))
                    
                    with col2:
                        max_records = st.number_input(
                            "Max records:", 
                            min_value=100, 
                            max_value=10000, 
                            value=min(2000, len(df_preview)),
                            key="train_max_records"
                        )
                    
                    if st.button("🚀 Train Target Model", type="primary", use_container_width=True, key="train_target_btn"):
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            # Step 1: Load and validate data
                            status_text.text("📥 Loading and validating data...")
                            progress_bar.progress(10)
                            
                            uploaded_train_file.seek(0)  # Reset file pointer
                            df_train = pd.read_csv(uploaded_train_file)
                            
                            # Use detected columns or manual selection
                            if detected:
                                smiles_col = detected.get("smiles")
                                activity_col = detected.get("activity_value")
                            else:
                                st.error("❌ Could not auto-detect columns. Please ensure your CSV has 'smiles' and 'activity_value' columns.")
                                st.stop()
                            
                            if not smiles_col or smiles_col not in df_train.columns:
                                st.error(f"❌ SMILES column '{smiles_col}' not found in CSV")
                                st.stop()
                            
                            if not activity_col or activity_col not in df_train.columns:
                                st.error(f"❌ Activity column '{activity_col}' not found in CSV")
                                st.stop()
                            
                            # Rename columns
                            df_train = df_train.rename(columns={smiles_col: "smiles", activity_col: "activity_value"})
                            df_train = df_train[["smiles", "activity_value"]].dropna()
                            df_train = df_train.head(max_records)
                            
                            # Step 2: Preprocess
                            status_text.text("🔧 Preprocessing data...")
                            progress_bar.progress(30)
                            
                            # Create active/inactive labels
                            df_train["active"] = (df_train["activity_value"] < ic50_threshold).astype(int)
                            
                            # Step 3: Featurize
                            status_text.text("🧪 Converting SMILES to fingerprints...")
                            progress_bar.progress(50)
                            
                            from featurization import smiles_to_matrix
                            X, idx = smiles_to_matrix(df_train["smiles"].tolist(), radius=2, n_bits=2048)
                            y = df_train["active"].values[idx]
                            
                            if len(X) == 0:
                                st.error("❌ No valid SMILES found in dataset")
                                st.stop()
                            
                            # Step 4: Train
                            status_text.text("🎓 Training model...")
                            progress_bar.progress(70)
                            
                            from sklearn.model_selection import train_test_split
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.metrics import roc_auc_score
                            
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42, stratify=y
                            )
                            
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
                            roc_auc = roc_auc_score(y_test, y_proba)
                            
                            # Step 5: Save
                            status_text.text("💾 Saving model...")
                            progress_bar.progress(90)
                            
                            import joblib
                            import os
                            os.makedirs("models", exist_ok=True)
                            
                            safe_name = target_name.replace(" ", "_").replace("-", "_").lower()
                            model_path = f"models/{safe_name}_rf.pkl"
                            joblib.dump(clf, model_path)
                            
                            # Save metadata
                            import json
                            metadata = {
                                "target_name": target_name,
                                "model_path": model_path,
                                "roc_auc": float(roc_auc),
                                "n_train": len(X_train),
                                "n_test": len(X_test),
                                "n_active": int(y_train.sum()),
                                "n_inactive": int((~y_train.astype(bool)).sum()),
                                "ic50_threshold_nM": ic50_threshold
                            }
                            
                            metadata_path = f"models/{safe_name}_metadata.json"
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=2)
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Training complete!")
                            
                            st.success(f"✅ Model trained successfully for **{target_name}**!")
                            st.json(metadata)
                            
                            # Reload models
                            reg.load()
                            st.session_state.selected_target = target_name
                            st.balloons()
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Training failed: {e}")
                            st.exception(e)
                
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {e}")
        
        with train_tab2:
            st.markdown("""
            **Train ADMET models from TDC datasets:**
            Select which ADMET properties to train. Each model takes ~2-5 minutes.
            """)
            
            admet_tasks = list(ADMET_TASKS.keys())
            selected_tasks = st.multiselect(
                "Select ADMET tasks to train:",
                options=admet_tasks,
                default=admet_tasks[:3] if len(admet_tasks) >= 3 else admet_tasks,
                help="Select one or more ADMET properties to train"
            )
            
            if st.button("🚀 Train Selected ADMET Models", type="primary", use_container_width=True, key="train_admet_btn"):
                if len(selected_tasks) == 0:
                    st.warning("⚠️ Please select at least one ADMET task")
                else:
                    # Progress tracking
                    total_tasks = len(selected_tasks)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_container = st.container()
                    
                    success_count = 0
                    failed_tasks = []
                    
                    for i, task in enumerate(selected_tasks):
                        status_text.text(f"Training {task} ({i+1}/{total_tasks})...")
                        progress = int((i / total_tasks) * 100)
                        progress_bar.progress(progress)
                        
                        try:
                            from train_admet import train_admet_model
                            # Capture output
                            import sys
                            from io import StringIO
                            
                            old_stdout = sys.stdout
                            sys.stdout = StringIO()
                            
                            train_admet_model(task, save_model=True)
                            
                            output = sys.stdout.getvalue()
                            sys.stdout = old_stdout
                            
                            with results_container:
                                st.success(f"✅ {task} trained successfully")
                            success_count += 1
                        except Exception as e:
                            with results_container:
                                st.error(f"❌ {task} failed: {e}")
                            failed_tasks.append(task)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Training complete!")
                    
                    if success_count == total_tasks:
                        st.success(f"✅ All {total_tasks} ADMET models trained successfully!")
                        st.balloons()
                    else:
                        st.warning(f"⚠️ {success_count}/{total_tasks} models trained successfully. Failed: {', '.join(failed_tasks)}")
                    
                    # Reload models
                    reg.load()
                    st.rerun()
        
        with train_tab3:
            st.markdown("""
            **One-click training for all models:**
            This will train all available models (target + ADMET).
            ⚠️ **This may take 30-60 minutes.**
            """)
            
            st.warning("⚠️ This will train all models and may take a long time. Make sure you have time to wait.")
            
            if st.button("🚀 Train All Models", type="primary", use_container_width=True, key="train_all_btn"):
                # Overall progress
                overall_progress = st.progress(0)
                overall_status = st.empty()
                results_container = st.container()
                
                try:
                    # Step 1: Train target models (if needed)
                    overall_status.text("🎯 Checking target models...")
                    overall_progress.progress(10)
                    
                    # Check which targets need training
                    available_targets = list_available_targets()
                    trained_targets = [t["target_name"] for t in available_targets]
                    
                    predefined_to_train = [name for name in PREDEFINED_TARGETS.keys() if name not in trained_targets]
                    
                    if predefined_to_train:
                        overall_status.text(f"🎯 Training {len(predefined_to_train)} target models...")
                        for i, target_name in enumerate(predefined_to_train):
                            target_info = PREDEFINED_TARGETS[target_name]
                            try:
                                model_path, metadata = download_and_train_target(
                                    target_name=target_name,
                                    chembl_id=target_info["chembl_id"],
                                    max_records=2000,
                                    ic50_threshold_nM=1000.0
                                )
                                with results_container:
                                    st.success(f"✅ {target_name} trained")
                            except Exception as e:
                                with results_container:
                                    st.error(f"❌ {target_name} failed: {e}")
                    else:
                        with results_container:
                            st.info("✅ All predefined targets already trained")
                    
                    overall_progress.progress(30)
                    
                    # Step 2: Train ADMET models
                    overall_status.text("🧬 Training ADMET models...")
                    admet_tasks = list(ADMET_TASKS.keys())
                    total_admet = len(admet_tasks)
                    
                    for i, task in enumerate(admet_tasks):
                        overall_status.text(f"🧬 Training {task} ({i+1}/{total_admet})...")
                        progress = 30 + int((i / total_admet) * 60)
                        overall_progress.progress(progress)
                        
                        try:
                            from train_admet import train_admet_model
                            train_admet_model(task, save_model=True)
                            with results_container:
                                st.success(f"✅ {task} trained")
                        except Exception as e:
                            with results_container:
                                st.error(f"❌ {task} failed: {e}")
                    
                    overall_progress.progress(100)
                    overall_status.text("✅ All models trained!")
                    st.success("🎉 All models trained successfully!")
                    st.balloons()
                    
                    # Reload models
                    reg.load()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
                    st.exception(e)
    
    # ========== TAB 7: Advanced Mode ==========
    with tab7:
        st.header("⚙️ Advanced Mode")
        st.write("Manual SMILES input for quick testing or single-molecule evaluation.")
        
        smiles_input = st.text_area(
            "Enter SMILES string(s) - one per line:",
            height=150,
            placeholder="CCO\nCc1ccccc1\nCC(C)NC(=O)...",
        )
        
        if st.button("Evaluate Single Molecule(s)", type="primary"):
            if not smiles_input.strip():
                st.warning("Please enter at least one SMILES string.")
            else:
                lines = [s.strip() for s in smiles_input.splitlines() if s.strip()]
                
                try:
                    from pipeline import evaluate_single_smiles
                    
                    reports = []
                    errors = []
                    
                    for smi in lines:
                        try:
                            report = evaluate_single_smiles(smi)
                            reports.append(report)
                        except ValueError as e:
                            errors.append({"smiles": smi, "error": str(e)})
                    
                    if reports:
                        st.subheader("📊 Results")
                        
                        # Create results dataframe
                        results_data = []
                        for report in reports:
                            row = {
                                "SMILES": report.smiles,
                                "HIV-1 p_active": report.target.get("p_active", "N/A"),
                                "Composite Score": report.score,
                            }
                            
                            for task_key in ADMET_TASKS.keys():
                                if task_key in report.admet:
                                    if "prob" in report.admet[task_key]:
                                        row[task_key] = report.admet[task_key]["prob"]
                                    elif "value" in report.admet[task_key]:
                                        row[task_key] = report.admet[task_key]["value"]
                            
                            results_data.append(row)
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df, use_container_width=True)
                    
                    if errors:
                        st.warning(f"{len(errors)} molecules failed to process")
                        st.dataframe(pd.DataFrame(errors))
                
                except Exception as e:
                    st.error(f"Error: {e}")
            
            # ========== SAR Explorer Section ==========
            st.divider()
            st.subheader("🧬 SAR Explorer")
            st.write("Generate structural analogs and compare their predicted properties.")
            
            sar_smiles_input = st.text_input(
                "Enter parent SMILES for analog generation:",
                placeholder="CC(C)NC(=O)...",
                help="SMILES string of the molecule to generate analogs from"
            )
            
            max_analogs = st.slider(
                "Maximum number of analogs:",
                min_value=5,
                max_value=50,
                value=20,
                step=5
            )
            
            if st.button("Generate & Score Analogs", type="primary"):
                if not sar_smiles_input.strip():
                    st.warning("Please enter a SMILES string.")
                else:
                    with st.spinner("Generating analogs and computing predictions..."):
                        try:
                            sar_results = generate_and_score_analogs(
                                parent_smiles=sar_smiles_input.strip(),
                                max_analogs=max_analogs,
                                hiv_model=reg.get_target_model(st.session_state.selected_target),
                                admet_models={task: reg.get(task) for task in ADMET_TASKS.keys() if reg.has_model(task)}
                            )
                            
                            if len(sar_results) == 0:
                                st.warning("No valid analogs could be generated from this SMILES.")
                            else:
                                st.session_state.sar_results = sar_results
                                st.success(f"✅ Generated {len(sar_results)} analogs")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error generating analogs: {e}")
                            st.exception(e)
            
            # Display SAR results
            if st.session_state.sar_results is not None:
                sar_df = st.session_state.sar_results.copy()
                
                st.subheader("📊 SAR Results")
                
                # Summary
                if "is_parent" in sar_df.columns:
                    parent_row = sar_df[sar_df["is_parent"] == True]
                    if len(parent_row) > 0:
                        parent_p_active = parent_row.iloc[0]["hiv1_p_active"]
                        parent_composite = parent_row.iloc[0]["composite_score"]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Parent p_active", f"{parent_p_active:.3f}")
                        with col2:
                            st.metric("Parent Composite", f"{parent_composite:.3f}")
                        with col3:
                            better_count = sar_df["better_than_parent"].sum() if "better_than_parent" in sar_df.columns else 0
                            st.metric("Better Analogs", better_count)
                
                # Display table
                display_cols_sar = ["smiles", "hiv1_p_active", "composite_score"]
                if "better_than_parent" in sar_df.columns:
                    display_cols_sar.append("better_than_parent")
                if "is_parent" in sar_df.columns:
                    display_cols_sar.append("is_parent")
                
                # Format for display
                display_sar = sar_df[display_cols_sar].copy()
                if "better_than_parent" in display_sar.columns:
                    display_sar["better_than_parent"] = display_sar["better_than_parent"].map({True: "✅", False: "❌"})
                if "is_parent" in display_sar.columns:
                    display_sar["is_parent"] = display_sar["is_parent"].map({True: "⭐", False: ""})
                
                st.dataframe(display_sar, use_container_width=True, height=400)
                
                # Download
                csv_sar = sar_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download SAR Results CSV",
                    data=csv_sar,
                    file_name="sar_analogs.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()
