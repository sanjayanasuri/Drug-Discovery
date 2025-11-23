"""
docking.py

AutoDock Vina integration for molecular docking.
Handles ligand preparation from SMILES and Vina docking execution.
"""

import os
import subprocess
import shutil
from typing import Optional, Tuple
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

# Default paths (can be overridden)
DEFAULT_RECEPTOR_PATH = "data/hiv1_protease/receptor.pdbqt"
DEFAULT_CONFIG_PATH = "data/hiv1_protease/vina_config.txt"
DEFAULT_VINA_BINARY = "vina"  # Assumes vina is in PATH
DEFAULT_OUTPUT_DIR = "results/docking"


def check_vina_available() -> Tuple[bool, str]:
    """
    Check if AutoDock Vina is available in the system.
    
    Returns
    -------
    Tuple[bool, str]
        (is_available, message)
    """
    vina_path = shutil.which(DEFAULT_VINA_BINARY)
    if vina_path:
        return True, f"Vina found at: {vina_path}"
    else:
        return False, f"Vina binary '{DEFAULT_VINA_BINARY}' not found in PATH. Please install AutoDock Vina."


def check_obabel_available() -> Tuple[bool, str]:
    """
    Check if Open Babel is available in the system.
    
    Returns
    -------
    Tuple[bool, str]
        (is_available, message)
    """
    obabel_path = shutil.which("obabel")
    if obabel_path:
        return True, f"Open Babel found at: {obabel_path}"
    else:
        return False, "Open Babel (obabel) not found in PATH. Required for ligand PDBQT preparation."


def prepare_ligand_from_smiles(
    smiles: str, 
    out_path: str,
    add_hydrogens: bool = True
) -> bool:
    """
    Prepare a ligand PDBQT file from a SMILES string.
    
    Uses RDKit to generate 3D conformer and write MOL file, then attempts
    to convert to PDBQT using obabel if available, or provides instructions.
    
    Parameters
    ----------
    smiles : str
        SMILES string
    out_path : str
        Output path for PDBQT file
    add_hydrogens : bool
        Whether to add hydrogens (default: True)
        
    Returns
    -------
    bool
        True if successful, False otherwise
        
    Notes
    -----
    This function requires either:
    1. Open Babel (obabel) installed and in PATH for PDBQT conversion
    2. Or manual conversion from MOL/PDB to PDBQT
    
    If obabel is not available, writes a MOL file and provides instructions.
    """
    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Add hydrogens if requested
    if add_hydrogens:
        mol = Chem.AddHs(mol)
    
    # Generate 3D conformer
    try:
        # Try multiple conformer generation attempts
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42)
        if len(conf_ids) == 0:
            # Fallback to single conformer
            AllChem.EmbedMolecule(mol, randomSeed=42)
        
        # Optimize conformer
        for conf_id in conf_ids:
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        
        # Use best conformer (lowest energy)
        if len(conf_ids) > 1:
            energies = []
            for conf_id in conf_ids:
                try:
                    energy = AllChem.MMFFGetMoleculeForceField(mol, confId=conf_id).CalcEnergy()
                    energies.append((energy, conf_id))
                except:
                    energies.append((float('inf'), conf_id))
            energies.sort()
            best_conf_id = energies[0][1]
        else:
            best_conf_id = 0
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate 3D conformer: {e}")
    
    # Create output directory
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # Try to convert to PDBQT using obabel
    obabel_path = shutil.which("obabel")
    
    if obabel_path:
        # Write temporary MOL file
        temp_mol = out_path.replace(".pdbqt", "_temp.mol")
        writer = Chem.SDWriter(temp_mol)
        writer.write(mol, confId=best_conf_id)
        writer.close()
        
        # Convert MOL to PDBQT using obabel
        try:
            cmd = [
                obabel_path,
                "-imol", temp_mol,
                "-opdbqt",
                "-O", out_path,
                "--gen3d"  # Ensure 3D coordinates
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Clean up temp file
            if os.path.exists(temp_mol):
                os.remove(temp_mol)
            
            if result.returncode == 0 and os.path.exists(out_path):
                return True
            else:
                raise RuntimeError(f"obabel conversion failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            if os.path.exists(temp_mol):
                os.remove(temp_mol)
            raise RuntimeError("obabel conversion timed out")
        except Exception as e:
            if os.path.exists(temp_mol):
                os.remove(temp_mol)
            raise RuntimeError(f"obabel conversion error: {e}")
    else:
        # No obabel available - write MOL file and provide instructions
        mol_path = out_path.replace(".pdbqt", ".mol")
        writer = Chem.SDWriter(mol_path)
        writer.write(mol, confId=best_conf_id)
        writer.close()
        
        raise RuntimeError(
            f"Open Babel (obabel) not found. Cannot convert to PDBQT.\n"
            f"Ligand saved as MOL file: {mol_path}\n"
            f"Please install Open Babel or manually convert to PDBQT format."
        )


def parse_vina_output(log_path: str) -> Optional[float]:
    """
    Parse Vina output log to extract best binding affinity.
    
    Parameters
    ----------
    log_path : str
        Path to Vina log file
        
    Returns
    -------
    Optional[float]
        Best binding affinity in kcal/mol, or None if not found
    """
    if not os.path.exists(log_path):
        return None
    
    best_affinity = None
    with open(log_path, 'r') as f:
        for line in f:
            # Vina output format: "   1    -9.2      0.000      0.000"
            if line.strip().startswith("   1"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        best_affinity = float(parts[1])
                        break
                    except ValueError:
                        continue
    
    return best_affinity


def run_vina(
    receptor_path: str,
    ligand_path: str,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    ligand_id: Optional[str] = None
) -> Tuple[Optional[float], str]:
    """
    Run AutoDock Vina docking.
    
    Parameters
    ----------
    receptor_path : str
        Path to receptor PDBQT file
    ligand_path : str
        Path to ligand PDBQT file
    config_path : Optional[str]
        Path to Vina configuration file (optional)
    output_dir : Optional[str]
        Output directory for pose files (default: results/docking/<ligand_id>)
    ligand_id : Optional[str]
        Identifier for the ligand (used in output paths)
        
    Returns
    -------
    Tuple[Optional[float], str]
        (best_affinity in kcal/mol, output_directory_path)
        Returns (None, output_dir) if docking fails
        
    Raises
    ------
    FileNotFoundError
        If receptor or ligand files don't exist
    RuntimeError
        If Vina execution fails
    """
    # Check inputs
    if not os.path.exists(receptor_path):
        raise FileNotFoundError(f"Receptor file not found: {receptor_path}")
    
    if not os.path.exists(ligand_path):
        raise FileNotFoundError(f"Ligand file not found: {ligand_path}")
    
    # Check Vina availability
    vina_available, vina_msg = check_vina_available()
    if not vina_available:
        raise RuntimeError(vina_msg)
    
    # Set up output directory
    if output_dir is None:
        if ligand_id:
            output_dir = os.path.join(DEFAULT_OUTPUT_DIR, str(ligand_id))
        else:
            output_dir = os.path.join(DEFAULT_OUTPUT_DIR, "ligand_1")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Output paths
    out_ligand = os.path.join(output_dir, "ligand_out.pdbqt")
    log_file = os.path.join(output_dir, "vina_log.txt")
    
    # Build Vina command
    cmd = [DEFAULT_VINA_BINARY, "--receptor", receptor_path, "--ligand", ligand_path]
    
    if config_path and os.path.exists(config_path):
        cmd.extend(["--config", config_path])
    
    cmd.extend(["--out", out_ligand])
    cmd.extend(["--log", log_file])
    
    # Run Vina
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            error_msg = f"Vina execution failed:\n{result.stderr}\n{result.stdout}"
            raise RuntimeError(error_msg)
        
        # Parse output
        best_affinity = parse_vina_output(log_file)
        
        return best_affinity, output_dir
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Vina docking timed out (>5 minutes)")
    except Exception as e:
        raise RuntimeError(f"Vina execution error: {e}")


def dock_smiles(
    smiles: str,
    receptor_path: str = DEFAULT_RECEPTOR_PATH,
    config_path: Optional[str] = DEFAULT_CONFIG_PATH,
    ligand_id: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Tuple[Optional[float], str, str]:
    """
    Complete docking workflow: prepare ligand from SMILES and run Vina.
    
    Parameters
    ----------
    smiles : str
        SMILES string of ligand
    receptor_path : str
        Path to receptor PDBQT file
    config_path : Optional[str]
        Path to Vina configuration file
    ligand_id : Optional[str]
        Identifier for the ligand
    output_dir : Optional[str]
        Output directory for all files
        
    Returns
    -------
    Tuple[Optional[float], str, str]
        (best_affinity in kcal/mol, ligand_pdbqt_path, output_directory)
        
    Raises
    ------
    ValueError
        If SMILES is invalid
    RuntimeError
        If ligand preparation or docking fails
    """
    # Set up output directory
    if output_dir is None:
        if ligand_id:
            output_dir = os.path.join(DEFAULT_OUTPUT_DIR, str(ligand_id))
        else:
            # Use hash of SMILES as ID
            import hashlib
            ligand_id = hashlib.md5(smiles.encode()).hexdigest()[:8]
            output_dir = os.path.join(DEFAULT_OUTPUT_DIR, ligand_id)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare ligand
    ligand_pdbqt = os.path.join(output_dir, "ligand.pdbqt")
    prepare_ligand_from_smiles(smiles, ligand_pdbqt)
    
    # Run docking
    best_affinity, final_output_dir = run_vina(
        receptor_path=receptor_path,
        ligand_path=ligand_pdbqt,
        config_path=config_path,
        output_dir=output_dir,
        ligand_id=ligand_id
    )
    
    return best_affinity, ligand_pdbqt, final_output_dir

