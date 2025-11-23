"""
screen_cli.py

Command-line interface for running the full screening pipeline on a SMILES string.
"""

import argparse
from pipeline import evaluate_single_smiles, MoleculeReport
from admet_loader import ADMET_TASKS


def format_report(report: MoleculeReport) -> str:
    """Format a MoleculeReport as a readable table."""
    lines = []
    lines.append("=" * 70)
    lines.append("MOLECULE SCREENING REPORT")
    lines.append("=" * 70)
    lines.append(f"\nSMILES: {report.smiles}")
    lines.append(f"Composite Score: {report.score:.4f}")
    
    # Target activity
    lines.append("\n" + "-" * 70)
    lines.append("TARGET ACTIVITY")
    lines.append("-" * 70)
    if report.target.get("p_active") is not None:
        lines.append(f"  HIV-1 Protease p_active: {report.target['p_active']:.4f}")
    else:
        lines.append("  HIV-1 Protease: Model not available")
    
    # ADMET by category
    lines.append("\n" + "-" * 70)
    lines.append("ADMET PROPERTIES")
    lines.append("-" * 70)
    
    # Absorption (A)
    lines.append("\n  Absorption (A):")
    for task in ["caco2_wang", "HIA_Hou"]:
        if task in report.admet:
            if "prob" in report.admet[task]:
                lines.append(f"    {task}: {report.admet[task]['prob']:.4f} (prob)")
            elif "value" in report.admet[task]:
                lines.append(f"    {task}: {report.admet[task]['value']:.4f}")
    
    # Distribution (D)
    lines.append("\n  Distribution (D):")
    for task in ["BBB_Martins", "PPBR_AZ"]:
        if task in report.admet:
            if "prob" in report.admet[task]:
                lines.append(f"    {task}: {report.admet[task]['prob']:.4f} (prob)")
            elif "value" in report.admet[task]:
                lines.append(f"    {task}: {report.admet[task]['value']:.4f}")
    
    # Metabolism (M)
    lines.append("\n  Metabolism (M):")
    for task in ["CYP3A4_Veith", "CYP2D6_Veith"]:
        if task in report.admet:
            if "prob" in report.admet[task]:
                lines.append(f"    {task}: {report.admet[task]['prob']:.4f} (prob)")
    
    # Excretion (E)
    lines.append("\n  Excretion (E):")
    for task in ["Half_Life_Obach"]:
        if task in report.admet:
            if "value" in report.admet[task]:
                lines.append(f"    {task}: {report.admet[task]['value']:.4f}")
    
    # Toxicity (T)
    lines.append("\n  Toxicity (T):")
    for task in ["hERG", "AMES", "DILI", "LD50_Zhu"]:
        if task in report.admet:
            if "prob" in report.admet[task]:
                lines.append(f"    {task}: {report.admet[task]['prob']:.4f} (prob)")
            elif "value" in report.admet[task]:
                lines.append(f"    {task}: {report.admet[task]['value']:.4f}")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run full screening pipeline on a SMILES string"
    )
    parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="SMILES string to evaluate",
    )
    
    args = parser.parse_args()
    
    try:
        report = evaluate_single_smiles(args.smiles)
        print(format_report(report))
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

