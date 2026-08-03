import os
import numpy as np

CAPACITY = 50
DEMAND_LOW = 1
DEMAND_HIGH = 9
DEPOT_COOR = [0.5, 0.5]

def gen_instance(n):
    locations = np.random.rand(n, 2)
    demands   = np.random.randint(DEMAND_LOW, DEMAND_HIGH + 1, size=n)
    depot       = np.array([DEPOT_COOR])
    all_locs    = np.concatenate((depot, locations), axis=0)
    all_demands = np.concatenate((np.zeros(1,), demands))
    return np.concatenate((all_demands.reshape(-1, 1), all_locs), axis=1)

def generate_cvrp_datasets():
    basepath    = os.path.dirname(__file__)
    dataset_dir = os.path.join(basepath, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)

    for mood, seed, problem_sizes in [
        ('train', 1234, (50,)),
        ('test',  4567, (20, 50, 100, 200, 300, 500)),
    ]:
        np.random.seed(seed)
        batch_size = 10 if mood == 'train' else 64

        for n in problem_sizes:
            instances = [gen_instance(n) for _ in range(batch_size)]
            data = np.stack(instances)

            filename = os.path.join(dataset_dir, f"{mood}_CVRP{n}.npy")
            np.save(filename, data)
            print(f"Generated {batch_size} {mood} instances of size {n}")

if __name__ == "__main__":
    generate_cvrp_datasets()
