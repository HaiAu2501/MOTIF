SYSTEM_PROMPT = (
    "You are an expert in the domain of optimization heuristics. "
    "Your task is to design heuristics that can effectively solve optimization problems."
)

PROBLEM = """PROBLEM: Capacitated Vehicle Routing Problem.
OBJECTIVE: Minimize total route cost while serving every customer within vehicle capacity.
"""

RULES = """RULES:
- Keep the exact function signature.
- Use only inputs passed to the function.
- You may define simple hyperparameters inside the function.
- Node 0 is the depot; customers are nodes 1..n-1.
- Do not include docstrings or long comments.
- Handle edge cases and return numerically stable outputs.
---
"""

F1 = f"""{PROBLEM}
TASK: Implement an initialization heuristic.

SIGNATURE:
```python
import numpy as np

def initialize(distances, demands, coords, capacity):
    pass
```
- `distances`: pairwise node distances.
- `demands`: demand of each node.
- `coords`: node coordinates.
- `capacity`: vehicle capacity.
- `heuristic`: returned transition desirability prior.
- `pheromone`: returned initial search memory.

HINT: Score each transition by how much it helps build short, compact routes. Start pheromone uniform.

{RULES}
"""

F2 = f"""{PROBLEM}
TASK: Implement a transition-weight heuristic.

SIGNATURE:
```python
import numpy as np

def compute_probabilities(pheromone, heuristic, iteration, n_iterations):
    pass
```
- `pheromone`: current transition memory.
- `heuristic`: transition desirability prior.
- `iteration`: current search iteration.
- `n_iterations`: total search iterations.
- `return`: transition weights.

HINT: Combine pheromone and heuristic so promising transitions dominate later without collapsing early.

{RULES}
"""

F3 = f"""{PROBLEM}
TASK: Implement a pheromone update heuristic.

SIGNATURE:
```python
import numpy as np

def update_pheromone(pheromone, solutions, costs, iteration, n_iterations):
    pass
```
- `pheromone`: current transition memory.
- `solutions`: a Python list of one solution per ant. Each solution is a list of routes,
  and each route is a list of node indices starting and ending at the depot 0. Ants have
  different numbers of routes and routes have different lengths, so `solutions` is ragged
  and cannot be converted to a rectangular array or indexed with numpy fancy indexing
  across ants or routes. Iterate over it with plain Python loops.
- `costs`: a Python list of one float per ant, the total cost of that ant's solution.
- `iteration`: current search iteration.
- `n_iterations`: total search iterations.
- `return`: updated pheromone matrix.

HINT: Reward transitions used by cheap solutions, and evaporate so old choices fade.

{RULES}
"""
