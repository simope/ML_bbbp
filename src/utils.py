import numpy as np

import config

def load_features_target() -> tuple:
    X = np.load(config.FEATURES_PATH)
    y = np.load(config.LABELS_PATH)
    return X, y