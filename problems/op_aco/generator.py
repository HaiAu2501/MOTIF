import numpy as np
import os

def generate_op_datasets(basepath: str = None):
    basepath = basepath or os.path.join(os.path.dirname(__file__), "datasets")
    os.makedirs(basepath, exist_ok=True)

    for mood, seed, problem_sizes in [
        ('train', 1234, (50,)),
        ('test',  4567, (50, 100, 200, 300, 500)),
    ]:
        np.random.seed(seed)
        batch_size = 5 if mood == 'train' else 64

        for n in problem_sizes:
            coords = np.random.rand(batch_size, n, 2)

            filename = os.path.join(basepath, f"{mood}_OP{n}.npz")
            np.savez(filename, coordinates=coords)

            print(f"Generated {batch_size} {mood} instances of size {n}")

if __name__ == "__main__":
    generate_op_datasets()
