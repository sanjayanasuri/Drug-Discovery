"""
nl_queries.py

Natural-language querying for lead filtering.
Rule-based parser that translates text queries into DataFrame filters.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import re
from pipeline import MoleculeReport


def filter_reports_by_text_query(
    reports: List[MoleculeReport], 
    query: str
) -> List[MoleculeReport]:
    """
    Filter MoleculeReports based on a text query.
    
    For now, implements very simple keyword-based filtering:
    - "safe" → filter by low toxicity probabilities
    - "potent" → filter by high HIV p_active
    - "brain-penetrant" → filter by high BBB probability
    - "absorbable" → filter by high HIA or Caco-2
    - "non-toxic" → filter by low hERG, AMES, DILI probabilities
    
    Parameters
    ----------
    reports : List[MoleculeReport]
        List of reports to filter
    query : str
        Text query (case-insensitive)
        
    Returns
    -------
    List[MoleculeReport]
        Filtered list of reports
        
    Note
    ----
    This is a placeholder implementation. In the future, this will be
    replaced with an LLM-based query system that can understand more
    complex natural language queries.
    """
    query_lower = query.lower()
    filtered = []
    
    for report in reports:
        include = True
        
        # "safe" or "non-toxic" → low toxicity
        if "safe" in query_lower or "non-toxic" in query_lower:
            tox_tasks = ["hERG", "AMES", "DILI"]
            for tox_key in tox_tasks:
                if tox_key in report.admet and "prob" in report.admet[tox_key]:
                    if report.admet[tox_key]["prob"] > 0.3:  # Threshold can be tuned
                        include = False
                        break
        
        # "potent" → high target activity
        if "potent" in query_lower:
            if report.target.get("p_active", 0.0) or 0.0 < 0.5:
                include = False
        
        # "brain-penetrant" or "bbb" → high BBB probability
        if "brain" in query_lower or "bbb" in query_lower or "penetrant" in query_lower:
            if "BBB_Martins" in report.admet and "prob" in report.admet["BBB_Martins"]:
                if report.admet["BBB_Martins"]["prob"] < 0.5:
                    include = False
            else:
                include = False  # No BBB prediction available
        
        # "absorbable" or "absorption" → good absorption properties
        if "absorb" in query_lower:
            has_good_absorption = False
            if "HIA_Hou" in report.admet and "prob" in report.admet["HIA_Hou"]:
                if report.admet["HIA_Hou"]["prob"] > 0.5:
                    has_good_absorption = True
            if "caco2_wang" in report.admet and "value" in report.admet["caco2_wang"]:
                if report.admet["caco2_wang"]["value"] > -5.15:  # Typical threshold
                    has_good_absorption = True
            if not has_good_absorption:
                include = False
        
        # "high score" → filter by composite score
        if "high score" in query_lower or "top" in query_lower:
            if report.score < 0.5:
                include = False
        
        if include:
            filtered.append(report)
    
    return filtered


def rank_reports_by_query(
    reports: List[MoleculeReport],
    query: str
) -> List[MoleculeReport]:
    """
    Rank reports by relevance to a query.
    
    Similar to filter_reports_by_text_query but returns all reports
    sorted by relevance score.
    
    Parameters
    ----------
    reports : List[MoleculeReport]
        List of reports to rank
    query : str
        Text query
        
    Returns
    -------
    List[MoleculeReport]
        Reports sorted by relevance (most relevant first)
    """
    query_lower = query.lower()
    
    # Simple scoring: add points based on query matches
    scored = []
    for report in reports:
        score = report.score  # Start with composite score
        
        if "potent" in query_lower:
            score += (report.target.get("p_active", 0.0) or 0.0) * 0.3
        
        if "safe" in query_lower or "non-toxic" in query_lower:
            tox_penalty = 0.0
            for tox_key in ["hERG", "AMES", "DILI"]:
                if tox_key in report.admet and "prob" in report.admet[tox_key]:
                    tox_penalty += report.admet[tox_key]["prob"]
            score -= tox_penalty * 0.2
        
        scored.append((score, report))
    
    # Sort by score (descending)
    scored.sort(key=lambda x: x[0], reverse=True)
    return [report for _, report in scored]


def parse_nl_query_to_filters(
    query: str,
    available_columns: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Parse a natural-language query into DataFrame filter constraints.
    
    Parameters
    ----------
    query : str
        Natural language query (e.g., "high hiv activity and low herg risk")
    available_columns : Optional[List[str]]
        List of available column names in the DataFrame
        
    Returns
    -------
    Tuple[Dict[str, Any], str]
        (filters_dict, interpretation_string)
        filters_dict maps column names to (operator, value) tuples
        interpretation_string explains how the query was parsed
        
    Examples
    --------
    >>> filters, interp = parse_nl_query_to_filters("high hiv activity and low herg")
    >>> # filters = {"hiv1_p_active": (">=", 0.8), "hERG_prob": ("<=", 0.3)}
    """
    query_lower = query.lower()
    filters = {}
    interpretations = []
    
    # Column name mappings
    column_mappings = {
        "hiv": "hiv1_p_active",
        "hiv1": "hiv1_p_active",
        "activity": "hiv1_p_active",
        "p_active": "hiv1_p_active",
        "herg": "hERG_prob",
        "herg_prob": "hERG_prob",
        "ames": "AMES_prob",
        "ames_prob": "AMES_prob",
        "dili": "DILI_prob",
        "dili_prob": "DILI_prob",
        "bbb": "BBB_Martins_prob",
        "bbb_prob": "BBB_Martins_prob",
        "hia": "HIA_Hou_prob",
        "hia_prob": "HIA_Hou_prob",
        "caco2": "caco2_wang_value",
        "caco2": "caco2_wang_value",
        "caco-2": "caco2_wang_value",
        "composite": "composite_score",
        "score": "composite_score",
        "composite_score": "composite_score",
    }
    
    # Extract numeric thresholds (e.g., "> 0.8", ">= 0.7", "< 0.3")
    numeric_pattern = r'([><=]+)\s*([0-9]+\.?[0-9]*)'
    numeric_matches = re.findall(numeric_pattern, query)
    
    for op_str, val_str in numeric_matches:
        val = float(val_str)
        # Find which column this refers to (look for column name before the operator)
        for key, col_name in column_mappings.items():
            if key in query_lower[:query_lower.find(op_str)]:
                if col_name not in filters:
                    filters[col_name] = (op_str, val)
                    interpretations.append(f"{col_name} {op_str} {val}")
                break
    
    # Parse qualitative descriptors
    # High/Low patterns
    high_patterns = [
        (r'high\s+hiv', "hiv1_p_active", ">=", 0.8),
        (r'high\s+activity', "hiv1_p_active", ">=", 0.8),
        (r'active\s*>\s*0\.?[89]', "hiv1_p_active", ">=", 0.85),
        (r'high\s+herg', "hERG_prob", ">=", 0.7),
        (r'high\s+bbb', "BBB_Martins_prob", ">=", 0.7),
        (r'good\s+bbb', "BBB_Martins_prob", ">=", 0.7),
        (r'high\s+hia', "HIA_Hou_prob", ">=", 0.7),
        (r'good\s+absorption', "HIA_Hou_prob", ">=", 0.7),
        (r'high\s+composite', "composite_score", ">=", 0.6),
        (r'high\s+score', "composite_score", ">=", 0.6),
    ]
    
    low_patterns = [
        (r'low\s+herg', "hERG_prob", "<=", 0.3),
        (r'low\s+herg\s+risk', "hERG_prob", "<=", 0.3),
        (r'low\s+toxicity', "hERG_prob", "<=", 0.3),  # Will also check AMES, DILI
        (r'low\s+ames', "AMES_prob", "<=", 0.3),
        (r'low\s+dili', "DILI_prob", "<=", 0.3),
        (r'low\s+risk', "hERG_prob", "<=", 0.3),
        (r'safe', "hERG_prob", "<=", 0.3),
        (r'non-toxic', "hERG_prob", "<=", 0.3),
    ]
    
    for pattern, col, op, val in high_patterns:
        if re.search(pattern, query_lower):
            mapped_col = column_mappings.get(col, col)
            if mapped_col not in filters:
                filters[mapped_col] = (op, val)
                interpretations.append(f"{mapped_col} {op} {val} (from 'high' descriptor)")
    
    for pattern, col, op, val in low_patterns:
        if re.search(pattern, query_lower):
            mapped_col = column_mappings.get(col, col)
            if mapped_col not in filters:
                filters[mapped_col] = (op, val)
                interpretations.append(f"{mapped_col} {op} {val} (from 'low' descriptor)")
    
    # Special cases
    if "active" in query_lower and ">" not in query_lower:
        # Default "active" means high activity
        if "hiv1_p_active" not in filters:
            filters["hiv1_p_active"] = (">=", 0.7)
            interpretations.append("hiv1_p_active >= 0.7 (from 'active' keyword)")
    
    if "good bbb" in query_lower or "brain-penetrant" in query_lower:
        if "BBB_Martins_prob" not in filters:
            filters["BBB_Martins_prob"] = (">=", 0.7)
            interpretations.append("BBB_Martins_prob >= 0.7 (from 'good BBB' descriptor)")
    
    if "low toxicity" in query_lower or "safe" in query_lower:
        # Apply to multiple toxicity endpoints
        for tox_col in ["hERG_prob", "AMES_prob", "DILI_prob"]:
            if tox_col not in filters:
                filters[tox_col] = ("<=", 0.3)
        interpretations.append("Toxicity probabilities <= 0.3 (from 'low toxicity' descriptor)")
    
    # Build interpretation string
    if interpretations:
        interp_str = "Query interpreted as:\n" + "\n".join(f"  • {i}" for i in interpretations)
    else:
        interp_str = "Query could not be fully interpreted. Using default filters."
    
    return filters, interp_str


def apply_nl_query_to_dataframe(
    df: pd.DataFrame,
    query: str
) -> Tuple[pd.DataFrame, str]:
    """
    Apply a natural-language query to filter a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with lead screening results
    query : str
        Natural language query
        
    Returns
    -------
    Tuple[pd.DataFrame, str]
        (filtered_dataframe, interpretation_string)
    """
    if df.empty:
        return df, "Empty DataFrame"
    
    # Parse query
    filters, interpretation = parse_nl_query_to_filters(query, available_columns=list(df.columns))
    
    if not filters:
        return df, "No filters applied (query not understood)"
    
    # Apply filters
    filtered_df = df.copy()
    
    for col_name, (operator, value) in filters.items():
        if col_name not in filtered_df.columns:
            continue  # Skip if column doesn't exist
        
        if operator == ">=":
            filtered_df = filtered_df[filtered_df[col_name] >= value]
        elif operator == "<=":
            filtered_df = filtered_df[filtered_df[col_name] <= value]
        elif operator == ">":
            filtered_df = filtered_df[filtered_df[col_name] > value]
        elif operator == "<":
            filtered_df = filtered_df[filtered_df[col_name] < value]
        elif operator == "==" or operator == "=":
            filtered_df = filtered_df[filtered_df[col_name] == value]
        elif operator in ["!=", "<>"]:
            filtered_df = filtered_df[filtered_df[col_name] != value]
    
    return filtered_df, interpretation

