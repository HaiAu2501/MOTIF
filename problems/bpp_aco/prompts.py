SYSTEM_PROMPT = (
    "You are an expert in the domain of optimization heuristics. "
    "Your task is to design heuristics that can effectively solve optimization problems."
)

PROBLEM = """PROBLEM: Bin Packing Problem.
OBJECTIVE: Minimize the number of bins used to pack all items without exceeding capacity.
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

def initialize(demands: np.ndarray, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    pass
```
- `demands`: item sizes.
- `capacity`: bin capacity.
- `heuristic`: returned item-pair compatibility prior.
- `pheromone`: returned initial search memory over item pairs.

HINT: Try pair-score formulas using size complementarity, residual capacity fit, large-small balance, and near-full-bin utilization. The heuristic should make item pairs that pack tightly more attractive, while pheromone starts smooth enough to explore alternatives.

{RULES}
"""

F2 = f"""{PROBLEM}
TASK: Implement a pheromone update heuristic.

SIGNATURE:
```python
import numpy as np

def update_pheromone(pheromone: np.ndarray, paths: list, fitnesses: np.ndarray, iteration: int, n_iterations: int) -> np.ndarray:
    pass
```
- `pheromone`: current item-pair memory.
- `paths`: item orderings constructed by ants.
- `fitnesses`: packing quality for each path.
- `iteration`: current search iteration.
- `n_iterations`: total search iterations.
- `return`: updated pheromone matrix.

HINT: Try pheromone deposits based on bin utilization, pair co-occurrence in good packings, fitness rank, or residual-waste reduction. Reinforce stable item groupings without saturating the matrix or destroying exploration.

{RULES}
"""
