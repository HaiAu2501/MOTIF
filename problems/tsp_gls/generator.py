import numpy as np
import os

def generate_tsp_datasets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    dataset_dir = os.path.join(current_dir, "datasets")

    os.makedirs(dataset_dir, exist_ok=True)
    
    splits = {
        "train": ([200], 10),
        "val": ([20, 50, 100, 200], 64),
        "test": ([20, 50, 100, 200, 500], 64),
    }

    for split, (sizes, n_instances) in splits.items():
        # Match the deterministic protocol used by ReEvo/MCTS-AHD.
        np.random.seed(len(split))
        for size in sizes:
            batch = np.random.random((n_instances, size, 2))
            filename = os.path.join(dataset_dir, f"{split}_TSP{size}.npy")
            np.save(filename, batch)
            print(f"Generated {n_instances} {split} instances of size {size}")

if __name__ == "__main__":
    generate_tsp_datasets()
