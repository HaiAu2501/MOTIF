import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from aco import run_tsp_aco

def eval_instance(coords, n_ants, n_iter, seed):
    D = distance_matrix(coords, coords)
    return run_tsp_aco(D, n_ants, n_iter, seed)

def process_file(path, n_ants, n_iter):
    data = np.load(path)
    n_instances = data.shape[0]
    seeds = np.arange(n_instances)
    results = []
    for i in range(n_instances):
        np.random.seed(seeds[i])
        results.append(eval_instance(data[i], n_ants, n_iter, int(seeds[i])))
    return np.array(results)

def main(mode):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if mode == "train":
        paths = [os.path.join(current_dir, 'datasets', 'train_TSP50.npy')]
    elif mode == "test":
        paths = [
            os.path.join(current_dir, 'datasets', f'test_TSP{size}.npy')
            for size in [20, 50, 100, 200, 300, 500, 1000]
        ]
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'test'.")

    if mode == "test":
        N_ANTS = 30
        N_ITER = 200
    else:
        N_ANTS = 30
        N_ITER = 100

    total = 0
    for path in paths:
        costs = process_file(path, n_ants=N_ANTS, n_iter=N_ITER)
        total += costs.sum()
        mean = costs.mean()
        if mode == "test":
            print(mean)
    print(total)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(mode)
