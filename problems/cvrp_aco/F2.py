import numpy as np

def compute_probabilities(pheromone, heuristic, iteration, n_iterations):
    # Basic ACO probability computation
    alpha = 1.0  # Pheromone influence
    beta = 1.0   # Heuristic influence
    
    # Compute basic probabilities
    probabilities = np.power(pheromone, alpha) * np.power(heuristic, beta)
    
    return probabilities