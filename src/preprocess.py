import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import os

import config


def load_bbbp(path: str) -> pd.DataFrame:
    # Load dataset from csv
    df = pd.read_csv(path)
    return df.reset_index(drop=True)

def validate_smiles(df: pd.DataFrame) -> pd.DataFrame:
    # Obtain molecule object
    df["mol_object"] = df["smiles"].apply(Chem.MolFromSmiles)
    # Drop the invalid molecules (from invalid smiles)
    df = df[df["mol_object"].notnull()]
    return df.reset_index(drop=True)

def get_fingerprint(mol: Chem.Mol, radius=2, n_bits=2048) -> np.ndarray:
    generator = GetMorganGenerator(radius=radius,fpSize=n_bits)
    fp = generator.GetFingerprintAsNumPy(mol)
    return fp

def generate_features(df: pd.DataFrame) -> np.ndarray:
    mols = df["mol_object"]
    fps = [get_fingerprint(mol) for mol in mols]
    features = np.array([np.array(fp) for fp in fps])
    return features

def generate_labels(df: pd.DataFrame) -> np.ndarray:
    labels = df["p_np"]
    return labels.to_numpy()
 
def save_preprocessed(features: np.ndarray, labels: np.ndarray):
    np.save(config.FEATURES_PATH, features)
    np.save(config.LABELS_PATH, labels)


if __name__ == "__main__":
    df = load_bbbp(config.DATASET_PATH)
    df = validate_smiles(df=df)
    features = generate_features(df=df)
    labels = generate_labels(df=df)
    save_preprocessed(features=features, labels=labels)