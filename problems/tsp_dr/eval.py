import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from dr import run_tsp_dr

def eval_instance(coords, seed=None):
    distances = distance_matrix(coords, coords)
    return run_tsp_dr(
        distances=distances,
        use_2opt=False,
        max_workers=20,
        seed=seed,
    )

def process_file(path):
    data = np.load(path)
    n_instances = data.shape[0]
    
    seeds = np.arange(n_instances)
    
    results = []
    for i in range(n_instances):
        coordinates = data[i]
        avg_cost = eval_instance(coordinates, seed=int(seeds[i]))
        results.append(avg_cost)
    
    return np.array(results)

def main(mode="train"):
    if mode != "train":
        raise ValueError("Only train mode is supported.")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, 'datasets', 'train_TSP100.npy')
        
    costs = process_file(path)
    print(costs.mean())

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(mode)
