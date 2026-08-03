import numpy as np
import os

def generate_tsp_datasets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)

    for mood, seed, sizes, n_instances in [
        ("train", 1234, [100], 5),
        ("test", 4567, [50, 100, 200], 64),
    ]:
        np.random.seed(seed)
        for size in sizes:
            batch = np.random.rand(n_instances, size, 2)
            np.save(os.path.join(dataset_dir, f"{mood}_TSP{size}.npy"), batch)
            print(f"Generated {n_instances} {mood} instances of size {size}")

if __name__ == "__main__":
    generate_tsp_datasets()
