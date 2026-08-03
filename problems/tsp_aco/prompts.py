SYSTEM_PROMPT = (
    "You are an expert in the domain of optimization heuristics. "
    "Your task is to design heuristics that can effectively solve optimization problems."
)

PROBLEM = """PROBLEM: Traveling Salesman Problem.
OBJECTIVE: Minimize the length of a tour that visits every city exactly once and returns to the start.
"""

RULES = """RULES:
- Keep the exact function signature.
- Use only inputs passed to the function.
- You may define simple hyperparameters inside the function.
- Do not include docstrings or long comments.
- Handle edge cases and return numerically stable outputs.
---
"""

F1 = f"""{PROBLEM}
TASK: Implement an initialization heuristic.

SIGNATURE:
```python
import numpy as np

def initialize(distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pass
```
- `distances`: pairwise city distances.
- `heuristic`: returned prior indicators of promising edges.
- `pheromone`: returned initial search memory over edges.

HINT: Try several edge-score formulas that mix inverse distance, neighborhood rank, local density, and global distance scale. The heuristic should separate edges likely to appear in short tours, while pheromone should start smooth enough for exploration.

{RULES}
"""

F2 = f"""{PROBLEM}
TASK: Implement a transition-weight heuristic.

SIGNATURE:
```python
import numpy as np

def compute_probabilities(
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    iteration: int,
    n_iterations: int
) -> np.ndarray:
    pass
```
- `pheromone`: current edge memory.
- `heuristic`: edge desirability prior.
- `iteration`: current search iteration.
- `n_iterations`: total search iterations.
- `return`: unnormalized transition weights.

HINT: Try power, log, rank, or temperature-style formulas that combine pheromone and heuristic differently across iterations. The weights should amplify promising edges without collapsing exploration too early.

{RULES}
"""

F3 = f"""{PROBLEM}
TASK: Implement a pheromone update heuristic.

SIGNATURE:
```python
import numpy as np

def update_pheromone(
    pheromone: np.ndarray,
    paths: np.ndarray,
    costs: np.ndarray,
    iteration: int,
    n_iterations: int,
) -> np.ndarray:
    pass
```
- `pheromone`: current edge memory.
- `paths`: tours constructed by ants.
- `costs`: tour cost for each ant.
- `iteration`: current search iteration.
- `n_iterations`: total search iterations.
- `return`: updated pheromone matrix.

HINT: Try deposit formulas based on tour cost rank, elite tours, relative cost gaps, or repeated edge usage, with evaporation that changes over time. Reinforce useful edges strongly enough to guide search but clip or smooth values to avoid saturation.

{RULES}
"""
