import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

import config

def load_features_target() -> tuple:
    X = np.load(config.FEATURES_PATH)
    y = np.load(config.LABELS_PATH)
    return X, y

def visualize_molecule_3d(smiles: str, width: int = 400, height: int = 400) -> str:
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Convert to 3Dmol view
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(Chem.MolToMolBlock(mol), "mol")
    
    # Set visualization style
    viewer.setStyle({'stick': {'radius': 0.2, 'color': 'spectrum'}})
    viewer.setBackgroundColor('#0E1117')  # Streamlit's dark background color
    viewer.zoomTo()
    
    # Return the HTML
    return viewer._make_html()