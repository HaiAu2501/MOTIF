import os
import sys
import numpy as np
from aco import run_mkp_aco

def eval_instance(prize, weight, n_ants, n_iter, seed=0):
    if seed is not None:
        np.random.seed(seed)
    
    return run_mkp_aco(prize, weight, n_ants, n_iter)

def process_file(path, n_ants, n_iter):
    data = np.load(path)
    prizes, weights = data['prizes'], data['weights']
    n_instances = prizes.shape[0]
    
    seeds = np.arange(n_instances)
    
    results = []
    for i in range(n_instances):
        result = eval_instance(prizes[i], weights[i], n_ants, n_iter, int(seeds[i]))
        results.append(result)
    
    return np.array(results)

def main(mode="train"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    N_ANTS = 10
    # Test keeps the training settings and only doubles the iteration count.
    N_ITER = 100 if mode == "test" else 50

    if mode == "train":
        problem_sizes = [100]
    elif mode == "test":
        problem_sizes = [50, 100, 200, 300, 500]
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'test'.")
    
    total_obj = 0
    
    for size in problem_sizes:
        path = os.path.join(current_dir, 'datasets', f'{mode}_MKP{size}.npz')
        
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
