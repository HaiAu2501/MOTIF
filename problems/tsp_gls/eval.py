import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from concurrent.futures import ThreadPoolExecutor
from gls import run_tsp_gls

def solve_instance(coordinates: np.ndarray, perturbation_moves: int, iter_limit: int, seed: int) -> float:
    """
    Solve a single TSP instance using GLS.
    
    Parameters
    ----------
    coordinates : np.ndarray, shape (n, 2)
        City coordinates
    perturbation_moves : int
        Number of perturbation moves per GLS iteration
    iter_limit : int
        Maximum number of GLS iterations
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    float
        Best tour cost found
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create distance matrix from coordinates
    # Match the ReEvo/MCTS-AHD GLS protocol: keep a small positive diagonal.
    dist_matrix = distance_matrix(coordinates, coordinates)
    dist_matrix += np.eye(len(coordinates)) * 1e-5
    
    # Solve using GLS
    best_cost = run_tsp_gls(
        distmat=dist_matrix,
        perturbation_moves=perturbation_moves,
        iter_limit=iter_limit
    )
    
    # Calculate and return tour cost
    return best_cost

def process_file(path: str, perturbation_moves: int, iter_limit: int) -> np.ndarray:
    """
    Process a dataset file and solve all instances.
    
    Parameters
    ----------
    path : str
        Path to dataset file (.npy format)
    perturbation_moves : int
        Number of perturbation moves per GLS iteration
    iter_limit : int
        Maximum number of GLS iterations
        
    Returns
    -------
    np.ndarray
        Array of tour costs for all instances
    """
    # Load dataset: shape (n_instances, n_cities, 2)
    data = np.load(path)
    return process_data(data, perturbation_moves, iter_limit)


def process_data(data: np.ndarray, perturbation_moves: int, iter_limit: int) -> np.ndarray:
    """Evaluate an in-memory batch of coordinate instances."""
    n_instances = data.shape[0]
    
    # Generate seeds for reproducibility
    seeds = np.arange(n_instances)
    
    # Process instances with ThreadPoolExecutor for parallel execution
    results = []
    for i in range(n_instances):
        coordinates = data[i]
        result = solve_instance(coordinates, perturbation_moves, iter_limit, int(seeds[i]))
        results.append(result)
    
    return np.array(results)

def main(mode: str = "train"):
    """
    Main evaluation function.
    
    Parameters
    ----------
    mode : str
        Evaluation mode: 'train', 'val', or 'test'
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Algorithm parameters
    PERTURBATION_MOVES = 30
    ITER_LIMIT = 1200

    # Determine dataset paths based on mode
    if mode == "train":
        paths = [os.path.join(current_dir, 'datasets', 'train_TSP200.npy')]
    elif mode == "val":
        paths = [
            os.path.join(current_dir, 'datasets', f'val_TSP{size}.npy')
            for size in [20, 50, 100, 200]
        ]
    elif mode == "test":
        paths = [
            os.path.join(current_dir, 'datasets', f'test_TSP{size}.npy')
            for size in [20, 50, 100, 200, 500]
        ]
    else:
        raise ValueError("Invalid mode. Choose 'train', 'val', or 'test'.")
    
    # Process all dataset files
    all_costs = []
    
    for path in paths:
        # Check if dataset file exists
        if not os.path.exists(path):
            print(f"Warning: Dataset file {path} not found. Skipping.")
            continue
        
        # Process dataset
        costs = process_file(path, PERTURBATION_MOVES, ITER_LIMIT)
        
        all_costs.extend(costs.tolist())
    
    # Print total cost (main metric)
    print(float(np.mean(all_costs)) if all_costs else float('inf'))

if __name__ == "__main__":
    # Get mode from command line argument
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(mode)
