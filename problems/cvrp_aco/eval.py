import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from aco import run_cvrp_aco

N_ANTS = 30
N_ITERATIONS = 100
# Test keeps the training settings and only doubles the iteration count.
N_ITERATIONS_TEST = 200
CAPACITY = 50

def eval_instance(coords, demands, capacity, n_ants, n_iter, seed):
    """
    Evaluate a single CVRP instance.
    
    Parameters
    ----------
    coords : np.ndarray, shape (n, 2)
        Node coordinates.
    demands : np.ndarray, shape (n,)
        Node demands.
    capacity : int
        Vehicle capacity.
    n_ants : int
        Number of ants.
    n_iter : int
        Number of iterations.
    seed : int
        Random seed.
        
    Returns
    -------
    float
        Best cost found.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create distance matrix
    dist_mat = distance_matrix(coords, coords)
    
    # Run CVRP ACO
    return run_cvrp_aco(dist_mat, demands, coords, capacity, n_ants, n_iter)

def process_file(path, n_ants, n_iter):
    """
    Process a dataset file.
    
    Parameters
    ----------
    path : str
        Path to dataset file.
    n_ants : int
        Number of ants.
    n_iter : int
        Number of iterations.
        
    Returns
    -------
    np.ndarray
        Results for all instances.
    """
    # Load dataset: shape (n_instances, n_nodes, 3) where last dim is [demand, x, y]
    dataset = np.load(path)
    demands, node_positions = dataset[:, :, 0], dataset[:, :, 1:]
    
    n_instances = node_positions.shape[0]
    seeds = np.arange(n_instances)
    
    results = []
    for i in range(n_instances):
        results.append(eval_instance(node_positions[i], demands[i], CAPACITY, n_ants, n_iter, int(seeds[i])))
    
    return np.array(results)

def main(mode="train"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if mode == "train":
        paths = [os.path.join(current_dir, 'datasets', f'train_CVRP{size}.npy') 
                 for size in [50]]
    elif mode == "test":
        paths = [os.path.join(current_dir, 'datasets', f'test_CVRP{size}.npy')
                 for size in [20, 50, 100, 200, 300, 500]]
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'test'.")
    
    n_iter = N_ITERATIONS_TEST if mode == "test" else N_ITERATIONS

    total_cost = 0
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: File {path} not found. Skipping.")
            continue

        costs = process_file(path, n_ants=N_ANTS, n_iter=n_iter)
        total_cost += costs.sum()
    
    print(total_cost)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(mode)
