import numpy as np
import os

def gen_instance(n, m):
    prize = np.random.rand(n)
    weight_matrix = np.random.rand(n, m)
    constraints = np.random.uniform(low=weight_matrix.max(axis=0),
                                    high=weight_matrix.sum(axis=0))
    weight_matrix = weight_matrix / constraints.reshape(1, m)
    return prize, weight_matrix

def generate_mkp_datasets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "datasets")
    os.makedirs(dataset_dir, exist_ok=True)

    m = 5 

    for mood, seed, problem_sizes in [
        ('train', 1234, (100,)),
        ('test',  4567, (50, 100, 200, 300, 500)),
    ]:
        np.random.seed(seed)
        batch_size = 5 if mood == 'train' else 64

        for n in problem_sizes:
            prizes = []
            weights = []
            for _ in range(batch_size):
                prize, weight = gen_instance(n, m)
                prizes.append(prize)
                weights.append(weight)

            prizes = np.stack(prizes)
            weights = np.stack(weights)

            filename = os.path.join(dataset_dir, f"{mood}_MKP{n}.npz")
            np.savez(filename, prizes=prizes, weights=weights)
            print(f"Generated {batch_size} {mood} instances of size {n}")

if __name__ == "__main__":
    generate_mkp_datasets()
