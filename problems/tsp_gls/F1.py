import numpy as np

def generate_guide_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    # Default implementation: use distance as guide
    # Longer edges are more likely to be "bad" choices
    return distance_matrix
