import os
import sys
import numpy as np
from aco import run_bpp_aco

N_ANTS = 20
N_ITERATIONS = 50

def eval_instance(instance_data, n_ants, n_iter, seed):
    capacity = int(instance_data[0])
    demands = instance_data[1:].astype(int)
    
    return run_bpp_aco(demands, capacity, n_ants, n_iter, seed)

def process_file(path, n_ants, n_iter):
    data = np.load(path)
    instances = data['instances']
    
    n_instances = instances.shape[0]
    seeds = np.arange(n_instances)
    
    results = []
    for i in range(n_instances):
        result = eval_instance(instances[i], n_ants, n_iter, int(seeds[i]))
        results.append(result)
    
    return np.array(results)

def main(mode="train"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if mode == "train":
        paths = [os.path.join(current_dir, 'datasets', 'train_BPP200.npz')]
    elif mode == "test":
        paths = [os.path.join(current_dir, 'datasets', f'test_BPP{size}.npz') 
                 for size in [200, 400, 600, 800, 1000]]
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'test'.")
    
    total_cost = 0
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: File {path} not found. Skipping.")
            continue

        costs = process_file(path, n_ants=N_ANTS, n_iter=N_ITERATIONS)
        total_cost += costs.sum()
    
    print(total_cost)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(mode)
