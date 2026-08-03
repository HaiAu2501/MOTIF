import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
from dr import run_cvrp_dr

CAPACITY = 50

def eval_instance(coords, demands, capacity, seed=None):
    distances = distance_matrix(coords, coords)
    return run_cvrp_dr(
        distances=distances,
        demands=demands,
        capacity=capacity,
        destruction_rate=0.3,
        use_2opt=False,
        max_workers=20,
        seed=seed
    )

def process_file(path):
    dataset = np.load(path)
    demands, node_positions = dataset[:, :, 0], dataset[:, :, 1:]
    
    n_instances = node_positions.shape[0]
    
    seeds = np.arange(n_instances)
    
    results = []
    for i in range(n_instances):
        coordinates = node_positions[i]
        instance_demands = demands[i]
        
        cost = eval_instance(coordinates, instance_demands, CAPACITY, seed=int(seeds[i]))
        results.append(cost)
    
    return np.array(results)

def main(mode="train"):
    if mode != "train":
        raise ValueError("Only train mode is supported.")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    path = os.path.join(current_dir, 'datasets', 'train_CVRP50.npy')
    
    costs = process_file(path)
    print(costs.mean())

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(mode)
