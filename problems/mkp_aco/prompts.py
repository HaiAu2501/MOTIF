SYSTEM_PROMPT = (
    "You are an expert in the domain of optimization heuristics. "
    "Your task is to design heuristics that can effectively solve optimization problems."
)

PROBLEM = """PROBLEM: Multiple Knapsack Problem.
OBJECTIVE: Maximize total prize under multiple normalized capacity constraints.
"""

RULES = """RULES:
- Keep the exact function signature; do not add optional parameters.
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

def initialize(prize: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pass
```
- `prize`: reward for selecting each item.
- `weight`: resource usage of each item.
- `heuristic`: returned item desirability prior.
- `pheromone`: returned initial item memory.

HINT: Try item-score formulas combining prize, normalized resource use, bottleneck scarcity, dominance, and prize-to-burden ratios. The heuristic should prefer high-value items that leave scarce capacities flexible, while pheromone should not overcommit initially.

{RULES}
"""

F2 = f"""{PROBLEM}
TASK: Implement an item-selection weight heuristic.

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
- `pheromone`: current item memory, including dummy entry.
- `heuristic`: item desirability prior, including dummy entry.
- `iteration`: current search iteration.
- `n_iterations`: total search iterations.
- `return`: unnormalized item-selection weights.

HINT: Try power, rank, temperature, or iteration-scheduled formulas that combine pheromone memory with item desirability. Keep dummy and infeasible entries numerically valid while making high-value feasible items stand out.

{RULES}
"""

F3 = f"""{PROBLEM}
TASK: Implement a pheromone update heuristic.

SIGNATURE:
```python
import numpy as np

def update_pheromone(pheromone: np.ndarray, sols: np.ndarray, objs: np.ndarray, it: int, n_iterations: int) -> np.ndarray:
    pass
```
- `pheromone`: current item memory.
- `sols`: item selections constructed by ants.
- `objs`: objective value for each solution.
- `it`: current search iteration.
- `n_iterations`: total search iterations.
- `return`: updated item pheromone vector.

HINT: Try deposit formulas based on objective rank, normalized prize gaps, selected-item frequency, or elite feasible solutions. Evaporate conservatively so good item sets persist, but clip or smooth pheromone to avoid locking onto one pattern.

{RULES}
"""
