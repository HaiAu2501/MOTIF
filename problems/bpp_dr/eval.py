import os
import sys
import numpy as np
from dr import run_bpp_dr

N_WORKERS = 20
DESTRUCTION_RATE = 0.3

def eval_instance(instance_data, max_workers, destruction_rate, seed):
    capacity = int(instance_data[0])
    demands = instance_data[1:].astype(int)
    
    return run_bpp_dr(
        demands=demands,
        capacity=capacity,
        destruction_rate=destruction_rate,
        max_workers=max_workers,
        seed=seed
    )

def process_file(path, max_workers, destruction_rate):
    data = np.load(path)
    instances = data['instances']
    
    n_instances = instances.shape[0]
    seeds = np.arange(n_instances)
    
    results = []
    for i in range(n_instances):
        result = eval_instance(instances[i], max_workers, destruction_rate, int(seeds[i]))
        results.append(result)
    
    return np.array(results)

def main(mode="train"):
    if mode != "train":
        raise ValueError("Only train mode is supported.")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    path = os.path.join(current_dir, 'datasets', 'train_BPP100.npz')

    bins_used = process_file(path, max_workers=N_WORKERS, destruction_rate=DESTRUCTION_RATE)

    print(bins_used.mean())

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    main(mode)
