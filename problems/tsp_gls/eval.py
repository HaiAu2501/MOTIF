import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from gls import run_tsp_gls


PERTURBATION_MOVES = 30
ITER_LIMIT = 1200
DATASET_SIZES = {
    "train": [200],
    "test": [20, 50, 100, 200, 500],
}

def solve_instance(coordinates: np.ndarray, perturbation_moves: int, iter_limit: int, seed: int) -> float:
    np.random.seed(seed)
    dist_matrix = distance_matrix(coordinates, coordinates)
    dist_matrix += np.eye(len(coordinates)) * 1e-5
    return run_tsp_gls(
        distmat=dist_matrix,
        perturbation_moves=perturbation_moves,
        iter_limit=iter_limit
    )

def process_file(path: str, perturbation_moves: int, iter_limit: int) -> np.ndarray:
    """Evaluate every instance stored in a dataset file."""
    return process_data(np.load(path), perturbation_moves, iter_limit)


def process_data(data: np.ndarray, perturbation_moves: int, iter_limit: int) -> np.ndarray:
    """Evaluate an in-memory batch of coordinate instances."""
    return np.array([
        solve_instance(coordinates, perturbation_moves, iter_limit, seed)
        for seed, coordinates in enumerate(data)
    ])

def main(mode: str = "train"):
    """Evaluate the train set used by MOTIF or the held-out test sets."""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if mode not in DATASET_SIZES:
        raise ValueError("Invalid mode. Choose 'train' or 'test'.")

    paths = [
        os.path.join(current_dir, "datasets", f"{mode}_TSP{size}.npy")
        for size in DATASET_SIZES[mode]
    ]
    all_costs = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Dataset file {path} not found. Skipping.")
            continue
        costs = process_file(path, PERTURBATION_MOVES, ITER_LIMIT)
        all_costs.extend(costs.tolist())

    print(float(np.mean(all_costs)) if all_costs else float('inf'))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(mode)
