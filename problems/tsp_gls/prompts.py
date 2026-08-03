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
TASK: Implement a guide-matrix heuristic.

SIGNATURE:
```python
import numpy as np

def generate_guide_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    pass
```
- `distance_matrix`: pairwise city distances.
- `return`: finite nonnegative guide matrix; larger values are penalized first.

HINT: Only relative guide order selects which tour edge is penalized. Design prior indicators of how bad it is to keep each edge; larger means worse. Return a finite matrix with the same shape.

{RULES}
"""
