import numpy as np

def update_pheromone(pheromone, solutions, costs, iteration, n_iterations):
    # Evaporation
    pheromone = pheromone * 0.9

    # Elitist deposit: only the best ant of this iteration reinforces its edges
    best = int(np.argmin(costs))
    deposit = 5.0 / costs[best]
    for route in solutions[best]:
        for j in range(len(route) - 1):
            pheromone[route[j], route[j + 1]] += deposit

    return np.maximum(pheromone, 1e-10)
