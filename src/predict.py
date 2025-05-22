import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from models.neural_network import BbbpNN

import config

def get_fingerprint(mol: Chem.Mol, radius=2, n_bits=2048) -> np.ndarray:
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprintAsNumPy(mol)
    return fp

def load_model(model_path: str = "models/nn_model.pt") -> BbbpNN:
    model = BbbpNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(smiles: str) -> float:
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Generate features
    features = get_fingerprint(mol)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Load the pre-trained NN model
    model = load_model()
    
    # Make prediction
    with torch.no_grad():
        probability = model(features).item()
    
    return int(probability > config.PROB_THRESHOLD)

if __name__ == "__main__":
    smiles = input("Insert SMILES:\n")
    prediction = predict(smiles)
    label = "permeable" if prediction else "non-permeable"
    print(f"According to the model, this molecule is {label}")