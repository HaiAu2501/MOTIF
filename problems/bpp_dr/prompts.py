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
TASK: Implement an item compatibility heuristic.

SIGNATURE:
```python
import numpy as np

def item_compatibility(i: int, j: int, demands: np.ndarray, capacity: int) -> float:
    pass
```
- `i`: first item.
- `j`: next item.
- `demands`: item sizes.
- `capacity`: bin capacity.
- `return`: item compatibility score; higher is better.

HINT: Try compatibility formulas using size complementarity, near-capacity fit, large-small pairing, and residual waste. Higher scores should place items together when their sizes help form dense bins rather than merely being similar.

{RULES}
"""

F2 = f"""{PROBLEM}
TASK: Implement a removal badness heuristic.

SIGNATURE:
```python
import numpy as np

def item_badness(item_idx: int, permutation: list[int], demands: np.ndarray, capacity: int) -> float:
    pass
```
- `item_idx`: item position in `permutation`.
- `permutation`: current item ordering.
- `demands`: item sizes.
- `capacity`: bin capacity.
- `return`: removal badness; higher is removed earlier.

HINT: Try badness formulas using item size, local pair compatibility, expected residual waste, and disruption to nearby items. Higher scores should select items whose removal creates more repair freedom or fixes poor bin structure.

{RULES}
"""

F3 = f"""{PROBLEM}
TASK: Implement an insertion-position heuristic.

SIGNATURE:
```python
import numpy as np

def insert_position(item: int, permutation: list[int], demands: np.ndarray, capacity: int) -> int:
    pass
```
- `item`: item to insert.
- `permutation`: current item ordering.
- `demands`: item sizes.
- `capacity`: bin capacity.
- `return`: insertion index.

HINT: Try insertion formulas using local compatibility, residual-capacity fit, waste reduction, and regret between candidate positions. Return the index that makes the ordering more likely to form full bins after greedy packing.

{RULES}
"""
