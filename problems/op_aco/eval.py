import os
import sys
import numpy as np
from aco import run_op_aco

def gen_prizes(coordinates):
    """
    Generate prize values based on distance from depot.
    
    Parameters
    ----------
    coordinates : np.ndarray, shape (n, 2)
        Node coordinates.
        
    Returns
    -------
    np.ndarray
        Prize values.
    """
    depot_coor = coordinates[0]
    distances = np.sqrt(np.sum((coordinates - depot_coor)**2, axis=1))
    prizes = 1 + np.floor(99 * distances / distances.max())
    prizes = prizes / prizes.max()
    return prizes

def gen_distance_matrix(coordinates):
    """
    Generate distance matrix from coordinates.
    
    Parameters
    ----------
    coordinates : np.ndarray, shape (n, 2)
        Node coordinates.
        
    Returns
    -------
    np.ndarray
        Distance matrix.
    """
    n_nodes = len(coordinates)
    distances = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                distances[i, j] = np.sqrt(np.sum((coordinates[i] - coordinates[j])**2))
    
    # Set diagonal to high value to prevent self-loops
    np.fill_diagonal(distances, 1e9)
    
    return distances

def get_max_len(n):
    """
    Get maximum tour length based on problem size.
    
    Parameters
    ----------
    n : int
        Problem size.
        
    Returns
    -------
    float
        Maximum tour length.
    """
    threshold_list = [50, 100, 200, 300]
    maxlen = [3.0, 4.0, 5.0, 6.0]
    for threshold, result in zip(threshold_list, maxlen):
        if n <= threshold:
            return result
    return 7.0

def create_instance(coordinates, idx=0):
    """
    Create an OP instance from coordinates.
    
    Parameters
    ----------
    coordinates : np.ndarray, shape (n, 2)
        Node coordinates.
    idx : int
        Instance index.
        
    Returns
    -------
    tuple
        (prizes, distances, maxlen)
    """
    n = coordinates.shape[0]
    
    # Create distance matrix
    distances = gen_distance_matrix(coordinates)
    
    # Generate prizes
    prizes = gen_prizes(coordinates)
    
    # Get maximum tour length
    maxlen = get_max_len(n)
    
    return prizes, distances, maxlen

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
        Result objectives.
    """
    # Load the dataset
    data = np.load(path)
    coordinates = data['coordinates']
    n_instances = coordinates.shape[0]
    
    # Generate seeds for reproducibility
    seeds = np.arange(n_instances)
    
    # Process all instances in parallel
    results = []
    for i in range(n_instances):
        prizes, distances, maxlen = create_instance(coordinates[i])
        results.append(eval_instance(prizes, distances, maxlen, n_ants, n_iter, int(seeds[i])))
    
    return np.array(results)

def eval_instance(prizes, distances, maxlen, n_ants, n_iter, seed=0):
    """
    Evaluate a single instance.
    
    Parameters
    ----------
    prizes : np.ndarray
        Prize values.
    distances : np.ndarray
        Distance matrix.
    maxlen : float
        Maximum tour length.
    n_ants : int
        Number of ants.
    n_iter : int
        Number of iterations.
    seed : int, optional
        Random seed.
        
    Returns
    -------
    float
        Best objective value.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Run ACO and return best objective value
    obj, _ = run_op_aco(prizes, distances, maxlen, n_ants, n_iter)
    return obj

def main(mode="train"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check and generate datasets if needed
    if not os.path.exists(os.path.join(current_dir, "datasets")):
        print("Datasets not found.")
    
    N_ANTS = 20
    N_ITER = 100
    
    if mode == "train":
        problem_sizes = [50]
    elif mode == "test":
        problem_sizes = [50, 100, 200, 300, 500]
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'test'.")
    
    # Process all files
    total_obj = 0
    
    for size in problem_sizes:
        path = os.path.join(current_dir, 'datasets', f'{mode}_OP{size}.npz')
        
        # Check if file exists
        if not os.path.exists(path):
            print(f"Warning: File {path} not found. Skipping.")
            continue

        objs = process_file(path, n_ants=N_ANTS, n_iter=N_ITER)
        total_obj += objs.sum()
        mean_obj = objs.mean()
        
        if mode == "test":
            print(-mean_obj)
            
    print(-total_obj)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(mode)
