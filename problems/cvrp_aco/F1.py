import numpy as np

def initialize(distances, demands, coords, capacity):
    # Uninformative prior: every transition looks equally good, so the ants have to
    # learn route structure from pheromone alone.
    heuristic = np.ones_like(distances)
    pheromone = np.ones_like(distances)
    return heuristic, pheromone
