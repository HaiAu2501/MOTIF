import numpy as np
import os

DEMAND_LOW = 20
DEMAND_HIGH = 100
CAPACITY = 150

def gen_instance(n):
    demands = np.random.randint(DEMAND_LOW, DEMAND_HIGH + 1, size=n)
    return np.concatenate([[CAPACITY], demands])

def generate_bpp_datasets(basepath: str = None):
    basepath = basepath or os.path.join(os.path.dirname(__file__), "datasets")
    os.makedirs(basepath, exist_ok=True)

    for mood, seed, problem_sizes in [
        ('train', 1234, (100,)),
        ('test',  4567, (100, 200, 300)),
    ]:
        np.random.seed(seed)
        batch_size = 5 if mood == 'train' else 64

        for n in problem_sizes:
            batch_data = np.stack([gen_instance(n) for _ in range(batch_size)])
            filename = os.path.join(basepath, f"{mood}_BPP{n}.npz")
            np.savez(filename, instances=batch_data)
            
            print(f"Generated {batch_size} {mood} instances of size {n}")

if __name__ == "__main__":
    generate_bpp_datasets()
