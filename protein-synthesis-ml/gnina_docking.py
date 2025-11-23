"""
gnina_docking.py

GNINA docking integration for improved accuracy and GPU acceleration.
GNINA is a fork of AutoDock Vina with CNN-based scoring and GPU support.
"""

import os
import subprocess
import shutil
from typing import Optional, Tuple, List
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

# Default paths
DEFAULT_GNINA_BINARY = "gnina"  # Assumes gnina is in PATH
DEFAULT_OUTPUT_DIR = "results/gnina_docking"


def check_gnina_available() -> Tuple[bool, str]:
    """
    Check if GNINA is available in the system.
    
    Returns
    -------
    Tuple[bool, str]
        (is_available, message)
    """
    gnina_path = shutil.which(DEFAULT_GNINA_BINARY)
    if gnina_path:
        # Check if GPU is available
        try:
            result = subprocess.run(
                [gnina_path, "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if "--gpu" in result.stdout or "--gpu" in result.stderr:
                return True, f"GNINA found at: {gnina_path} (GPU support available)"
            else:
                return True, f"GNINA found at: {gnina_path} (CPU only)"
        except Exception:
            return True, f"GNINA found at: {gnina_path}"
    else:
        return False, f"GNINA binary '{DEFAULT_GNINA_BINARY}' not found in PATH. Install from: https://github.com/gnina/gnina"


def dock_smiles_gnina(
    smiles: str,
    receptor_path: str,
    ligand_id: str = "ligand",
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_gpu: bool = True,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    use_cnn_scoring: bool = True,
    cnn_model: Optional[str] = None,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
    size_x: float = 20.0,
    size_y: float = 20.0,
    size_z: float = 20.0,
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Dock a SMILES molecule to a receptor using GNINA.
    
    Parameters
    ----------
    smiles : str
        SMILES string of ligand
    receptor_path : str
        Path to receptor PDBQT file
    ligand_id : str
        Identifier for ligand (used in output filenames)
    config_path : str, optional
        Path to GNINA config file (alternative to center/size parameters)
    output_dir : str, optional
        Directory to save docking results
    use_gpu : bool
        Whether to use GPU acceleration (default: True)
    exhaustiveness : int
        Exhaustiveness of search (default: 8, higher = more thorough)
    num_modes : int
        Number of binding modes to generate (default: 9)
    use_cnn_scoring : bool
        Whether to use CNN-based scoring (default: True)
    cnn_model : str, optional
        Path to custom CNN model (if None, uses default)
    center_x, center_y, center_z : float
        Center coordinates of search space
    size_x, size_y, size_z : float
        Size of search space in Angstroms
        
    Returns
    -------
    Tuple[Optional[float], Optional[str], Optional[str]]
        (best_affinity_kcal_mol, ligand_pdbqt_path, output_directory)
        Returns (None, None, None) if docking fails
    """
    # Check GNINA availability
    gnina_available, msg = check_gnina_available()
    if not gnina_available:
        raise RuntimeError(f"GNINA not available: {msg}")
    
    gnina_binary = shutil.which(DEFAULT_GNINA_BINARY)
    if not gnina_binary:
        raise RuntimeError("GNINA binary not found")
    
    # Prepare output directory
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare ligand PDBQT
    # Use RDKit to generate 3D conformer
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol = Chem.AddHs(mol)
    
    # Generate 3D conformer
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        # Try alternative method
        try:
            AllChem.EmbedMolecule(mol, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception as e:
            raise ValueError(f"Failed to generate 3D conformer: {e}")
    
    # Write MOL file
    mol_path = os.path.join(output_dir, f"{ligand_id}.mol")
    writer = Chem.SDWriter(mol_path)
    writer.write(mol)
    writer.close()
    
    # Convert to PDBQT using obabel (if available)
    ligand_pdbqt = os.path.join(output_dir, f"{ligand_id}.pdbqt")
    obabel_path = shutil.which("obabel")
    
    if obabel_path:
        try:
            subprocess.run(
                [obabel_path, "-imol", mol_path, "-opdbqt", "-O", ligand_pdbqt, "-x"],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to convert ligand to PDBQT: {e}")
    else:
        raise RuntimeError("Open Babel (obabel) required for ligand preparation. Install from: https://openbabel.org/")
    
    # Prepare GNINA command
    cmd = [gnina_binary, "-r", receptor_path, "-l", ligand_pdbqt]
    
    # Add search space
    if config_path and os.path.exists(config_path):
        cmd.extend(["--config", config_path])
    else:
        cmd.extend([
            "--center_x", str(center_x),
            "--center_y", str(center_y),
            "--center_z", str(center_z),
            "--size_x", str(size_x),
            "--size_y", str(size_y),
            "--size_z", str(size_z),
        ])
    
    # Add docking parameters
    cmd.extend([
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
    ])
    
    # GPU option
    if use_gpu:
        cmd.append("--gpu")
    
    # CNN scoring
    if use_cnn_scoring:
        if cnn_model:
            cmd.extend(["--cnn", cnn_model])
        else:
            cmd.append("--cnn")  # Use default CNN model
    
    # Output file
    output_pdbqt = os.path.join(output_dir, f"{ligand_id}_out.pdbqt")
    cmd.extend(["--out", output_pdbqt])
    
    # Run GNINA
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"GNINA error: {result.stderr}")
            return None, None, None
        
        # Parse output for best affinity
        # GNINA outputs: "   1    -8.5  ..." format
        best_affinity = None
        for line in result.stdout.split("\n"):
            if line.strip().startswith("   1"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        best_affinity = float(parts[1])
                        break
                    except ValueError:
                        continue
        
        # Also try to parse from output file
        if best_affinity is None and os.path.exists(output_pdbqt):
            with open(output_pdbqt, "r") as f:
                for line in f:
                    if "REMARK VINA RESULT:" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                best_affinity = float(parts[3])
                                break
                            except ValueError:
                                continue
        
        return best_affinity, ligand_pdbqt, output_dir
    
    except subprocess.TimeoutExpired:
        print("GNINA docking timed out")
        return None, None, None
    except Exception as e:
        print(f"GNINA docking error: {e}")
        return None, None, None


def batch_dock_gnina(
    smiles_list: List[str],
    receptor_path: str,
    output_dir: Optional[str] = None,
    use_gpu: bool = True,
    use_cnn_scoring: bool = True,
    **dock_kwargs
) -> List[Optional[float]]:
    """
    Dock multiple SMILES strings using GNINA.
    
    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings
    receptor_path : str
        Path to receptor PDBQT file
    output_dir : str, optional
        Directory to save results
    use_gpu : bool
        Whether to use GPU
    use_cnn_scoring : bool
        Whether to use CNN scoring
    **dock_kwargs
        Additional arguments passed to dock_smiles_gnina
        
    Returns
    -------
    List[Optional[float]]
        List of binding affinities (kcal/mol), None if docking failed
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    affinities = []
    for i, smiles in enumerate(smiles_list):
        try:
            affinity, _, _ = dock_smiles_gnina(
                smiles=smiles,
                receptor_path=receptor_path,
                ligand_id=f"ligand_{i}",
                output_dir=output_dir,
                use_gpu=use_gpu,
                use_cnn_scoring=use_cnn_scoring,
                **dock_kwargs
            )
            affinities.append(affinity)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(smiles_list)}")
        except Exception as e:
            print(f"  Warning: GNINA docking failed for molecule {i}: {e}")
            affinities.append(None)
    
    return affinities

