import math
import random
import subprocess
import sys

import numpy as np

from src.node import Node
from src.operators import Operators


class CompetitiveMCTS:
    """
    Competitive 2-player MCTS for strategy optimization.
    
    Key design:
    - Tree search is driven entirely by node.active_player (not a global current_player).
    - Root is always P1's turn. Children alternate: depth 1 is P2, depth 2 is P1, etc.
    - Each non-root node stores one W/Q pair for the player who created that node
      (the active player of its parent). This makes UCT selection at a parent use
      child values from the parent's perspective.
    """
    
    def __init__(self, function_name: str, strategy_id: str, initial_code: str,
                 client, prompts: dict, problem_config, baseline_cost: float, controller):
        self.function_name = function_name
        self.strategy_id = strategy_id
        self.client = client
        self.prompts = prompts
        self.problem_config = problem_config
        self.baseline_cost = baseline_cost
        self.controller = controller
        
        # Root node: both players start with identical code, P1's turn
        self.root = Node(
            p1_code=initial_code,
            p2_code=initial_code,
            function_name=function_name,
            depth=0,
            active_player="P1"
        )
        self.root.p1_cost = baseline_cost
        self.root.p2_cost = baseline_cost
        self.root.p1_improvement = 0.0
        self.root.p2_improvement = 0.0
        
        # Iteration counter (for logging only, not for tree logic)
        self.turn_count = 0
        
        # Best tracking per player: only genuine improvements replace incumbents.
        self.p1_best_cost = baseline_cost
        self.p2_best_cost = baseline_cost
        self.p1_best_code = initial_code
        self.p2_best_code = initial_code
        self.p1_best_improvement = 0.0
        self.p2_best_improvement = 0.0
        
        # For logging
        self.latest_generated_code = ""
        self.latest_reflection = ""
        self.recent_attempts = []
        self.reflection_memory = []
        
        # Q-value parameters
        self.lambda_factor = 0.7
        self.k_factor = 1.0
        self.ucb_c = 0.15

        # Exploration decays as the global budget is consumed; the controller
        # refreshes this before each block of inner iterations.
        self.exploration_scale = 1.0

        # Observed improvements. Both the Q scale and the viability gate are derived
        # from this rather than from constants, because the reachable span differs by
        # orders of magnitude across problem-solvers: about 0.4% for tsp_gls against
        # over 300% for tsp_aco. Any fixed percentage is either never or always hit.
        self.improvement_history = []

        # Robust scale is measured in MADs, which is unit-free, so these two stay
        # meaningful for every problem-solver.
        self.viability_mad_multiplier = 5.0
        self.viability_min_history = 8
        self.history_window = 40
    
    def run_iteration(self) -> dict:
        """
        Run one MCTS iteration: Selection -> Expansion -> Simulation -> Backpropagation.
        Tree search is driven by node.active_player, NOT a global player toggle.
        """
        # 1. Selection: traverse from root, find node to expand
        node, operator = self._select()
        
        if node is None or operator is None:
            self.turn_count += 1
            return {"success": False, "reason": "selection_failed"}
        
        # The player who will expand is node.active_player
        expanding_player = node.active_player
        
        # 2. Expansion: create child using the operator
        child = self._expand(node, operator)
        
        if child is None:
            self.turn_count += 1
            return {"success": False, "reason": "expansion_failed"}
        
        # 3. Simulation: evaluate new code, compute Q1 and Q2
        q1, q2 = self._simulate(child, expanding_player)
        
        # 4. Backpropagation: assign each edge/node value to the player who created it
        self._backpropagate(child, q1, q2)
        
        # 5. Update best tracking
        improvement = self._update_best_if_improved(child, expanding_player)
        
        self.turn_count += 1
        
        return {
            "success": True,
            "player": expanding_player,
            "improvement": improvement,
            "operator": operator,
            "summary": child.summary,
            "code": self.latest_generated_code,
            "reflection": self.latest_reflection,
            "cost": child.get_cost(expanding_player)
        }
    
    def _select(self):
        """
        Selection phase: traverse from root using UCT.
        
        At each node:
        - If node has unused operators, return (node, operator) for expansion.
        - Otherwise, select child with highest UCT value from perspective of node.active_player.
        
        UCT(child) = Q(child) + C * sqrt(log(N) / n)
        where Q is the Q-value stored at child from the perspective of
        node.active_player, i.e. the player choosing among node's children.
        """
        node = self.root
        
        while True:
            # Check for unused operators at this node
            unvisited = node.get_unvisited_operators(Operators.AVAILABLE)
            
            if unvisited:
                # Expand this node with an unused operator
                operator = random.choice(unvisited)
                return node, operator
            
            # All operators used - must select a child
            if not node.children:
                # Leaf with all operators used - revisit with random operator
                operator = random.choice(Operators.AVAILABLE)
                return node, operator
            
            # Select best child using UCT from perspective of node.active_player
            best_child = self._select_best_child_uct(node)
            
            if best_child is None:
                return node, random.choice(Operators.AVAILABLE)
            
            node = best_child
    
    def _select_best_child_uct(self, node):
        """
        Select best child using UCT formula.
        
        The Q-value at each child represents the value from the perspective of
        the player who created that child, i.e. node.active_player.
        """
        if not node.children:
            return None
        
        parent_visits = node.visits if node.visits > 0 else 1
        best_child = None
        best_uct = float('-inf')

        # Unvisited children get priority, but pick among them at random. Returning
        # the first one made selection collapse onto the leftmost branch every time
        # statistics were cleared after a baseline change.
        unvisited = [c for c in node.children if c.visits == 0]
        if unvisited:
            return random.choice(unvisited)

        exploration_c = self.ucb_c * max(self.exploration_scale, 0.1)

        for child in node.children:
            q_value = child.get_q_value()
            exploration = exploration_c * math.sqrt(math.log(parent_visits) / child.visits)
            uct_value = q_value + exploration
            
            if uct_value > best_uct:
                best_uct = uct_value
                best_child = child
        
        return best_child
    
    def _expand(self, node, operator):
        """
        Expansion phase: create child node using operator on node.active_player's code.
        
        - The active player at 'node' generates new code.
        - The opponent's code is copied unchanged.
        - The child's active_player is the opponent.
        """
        expanding_player = node.active_player
        opponent = "P2" if expanding_player == "P1" else "P1"

        try:
            # Apply operator to generate new code. If this player's code at this node
            # failed or was catastrophically bad, fall back to its own best code so
            # the subtree is not built on top of a broken implementation.
            seed_code = self.get_seed_code(node, expanding_player)

            new_code, summary = Operators.apply(
                operator, node, self, self.client, self.prompts,
                self.strategy_id, self.baseline_cost, seed_code
            )
            
            self.latest_generated_code = new_code
            
            # Create child: expanding_player gets new code, opponent keeps old code
            if expanding_player == "P1":
                child = Node(
                    p1_code=new_code,
                    p2_code=node.p2_code,
                    function_name=self.function_name,
                    parent=node,
                    depth=node.depth + 1,
                    active_player=opponent,  # Next turn is opponent's
                    operator=operator,
                    summary=summary
                )
            else:
                child = Node(
                    p1_code=node.p1_code,
                    p2_code=new_code,
                    function_name=self.function_name,
                    parent=node,
                    depth=node.depth + 1,
                    active_player=opponent,  # Next turn is opponent's
                    operator=operator,
                    summary=summary
                )
            
            node.add_child(child)
            node.mark_operator_used(operator)
            return child
            
        except Exception as e:
            print(f"[EXPANSION ERROR] {e}")
            return None
    
    def _simulate(self, child, expanding_player):
        """
        Simulation phase: evaluate the new code and compute Q1, Q2.
        
        - Only the expanding_player's code is new and needs evaluation.
        - The opponent's improvement is inherited from parent.
        - Compute Q1 and Q2 based on improvements.
        
        Returns: (q1, q2) tuple
        """
        opponent = "P2" if expanding_player == "P1" else "P1"

        # Evaluate expanding_player's new code
        new_cost, error_text = self._evaluate_code(child, expanding_player)

        if new_cost is None or new_cost == float('inf'):
            # Evaluation failed: assign a strong negative improvement so invalid
            # implementations do not receive neutral sigmoid reward.
            child.set_cost(expanding_player, float('inf'))
            child.set_improvement(expanding_player, -100.0)
            child.set_viable(expanding_player, False)
        else:
            child.set_cost(expanding_player, new_cost)
            improvement = (self.baseline_cost - new_cost) / abs(self.baseline_cost) * 100
            child.set_improvement(expanding_player, improvement)
            # Runs, but is a gross outlier on the bad side of this tree's own
            # distribution: keep the measurement, refuse to let it seed expansions.
            # The floor is computed before appending so a blow-up cannot widen the
            # scale that judges it.
            floor = self._viability_floor()
            child.set_viable(expanding_player, floor is None or improvement >= floor)
            self.improvement_history.append(improvement)

        self.recent_attempts.append({
            "player": expanding_player,
            "operator": child.operator,
            "summary": child.summary,
            "code": child.get_code(expanding_player),
            "improvement": child.get_improvement(expanding_player),
            "success": new_cost is not None and new_cost != float('inf'),
            "viable": child.is_viable(expanding_player),
            "error": error_text
        })
        
        # Opponent's values inherited from parent
        if child.parent:
            child.set_cost(opponent, child.parent.get_cost(opponent))
            child.set_improvement(opponent, child.parent.get_improvement(opponent))
        
        # Compute Q1 and Q2
        i1 = child.p1_improvement
        i2 = child.p2_improvement
        
        q1 = self._compute_q_for_player(i1, i2)
        q2 = self._compute_q_for_player(i2, i1)
        
        return q1, q2
    
    def _robust_scale(self):
        """
        (median, MAD-based sigma) of recent improvements, or None when there is not
        enough history yet.

        MAD is used rather than the standard deviation because blow-ups (-300% on
        tsp_aco) would otherwise dominate the estimate, and because it needs no
        arbitrary cutoff to exclude them.
        """
        recent = [i for i in self.improvement_history[-self.history_window:]
                  if math.isfinite(i)]
        if len(recent) < self.viability_min_history:
            return None

        arr = np.asarray(recent, dtype=float)
        median = float(np.median(arr))
        sigma = float(np.median(np.abs(arr - median))) * 1.4826
        if sigma < 1e-12:
            return None
        return median, sigma

    def _current_k(self):
        """
        Scale of the sigmoid, adapted to the spread of improvements this problem
        actually produces.

        With a fixed k = 1.0 and improvements of a few tenths of a percent (tsp_gls),
        sigmoid(k * I) stays within ~0.03 of 0.5 for every candidate, so Q carries no
        signal and UCT degenerates into visit counting. Setting k ~ 1 / sigma keeps the
        useful part of the sigmoid aligned with the observed range.
        """
        scale = self._robust_scale()
        if scale is None:
            return self.k_factor
        return float(np.clip(1.0 / scale[1], 0.2, 50.0))

    def _viability_floor(self):
        """
        Improvement below which a candidate counts as a blow-up and may not seed
        further expansions.

        Expressed in MADs below the median of what this tree has actually produced,
        so it self-scales per problem-solver. Returns None while history is too short,
        in which case only hard evaluation failures are treated as non-viable.
        """
        scale = self._robust_scale()
        if scale is None:
            return None
        median, sigma = scale
        return median - self.viability_mad_multiplier * sigma

    def _compute_q_for_player(self, my_improvement, opponent_improvement):
        """
        Q = lambda * sigmoid(k * I) + (1 - lambda) * sigmoid(k * (I - I_opponent))
        """
        k = self._current_k()
        # Clip to avoid overflow
        baseline_term = 1 / (1 + math.exp(np.clip(-k * my_improvement, -50, 50)))
        relative_term = 1 / (1 + math.exp(np.clip(-k * (my_improvement - opponent_improvement), -50, 50)))

        return self.lambda_factor * baseline_term + (1 - self.lambda_factor) * relative_term
    
    def _evaluate_code(self, child, player):
        """
        Evaluate player's code by writing to file, running eval, and reverting.

        Returns (cost, error_text). error_text is the tail of the interpreter's
        stderr when the run produced no usable objective, so the next prompt can
        show the LLM why its code failed instead of silently repeating the mistake.
        """
        return self._run_evaluation(child.get_code(player))

    def _run_evaluation(self, code_to_eval):
        try:
            strategy_path = self._get_strategy_path()

            # Backup original
            with open(strategy_path, 'r', encoding="utf-8") as f:
                original_code = f.read()

            try:
                with open(strategy_path, 'w', encoding="utf-8") as f:
                    f.write(code_to_eval)

                # Run evaluation
                result = subprocess.run(
                    [sys.executable, self.problem_config.eval_script, "train"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=120
                )

                stderr = result.stderr.strip()
                if stderr:
                    print(f"[EVAL WARNING] {stderr}")

                if not result.stdout.strip():
                    return None, self._trim_error(stderr) or "No objective printed by eval script."

                lines = result.stdout.strip().split('\n')
                try:
                    return float(lines[-1]), ""
                except ValueError:
                    return None, self._trim_error(stderr) or f"Unparsable objective: {lines[-1][:120]}"

            finally:
                # Always revert
                with open(strategy_path, 'w', encoding="utf-8") as f:
                    f.write(original_code)

        except subprocess.TimeoutExpired:
            print("[EVALUATION ERROR] timeout")
            return None, "Evaluation exceeded the 120s time limit; the implementation is too slow."
        except Exception as e:
            print(f"[EVALUATION ERROR] {e}")
            return None, self._trim_error(str(e))

    @staticmethod
    def _trim_error(text, limit=400):
        if not text:
            return ""
        text = text.strip()
        return text if len(text) <= limit else "..." + text[-limit:]
    
    def _get_strategy_path(self):
        for func in self.problem_config.functions:
            if func.id == self.strategy_id:
                return func.path
        raise ValueError(f"Strategy path not found for {self.strategy_id}")
    
    def _backpropagate(self, leaf, q1, q2):
        """
        Backpropagation: propagate Q-values up the tree.

        Selection at a node compares its children, so each child must store the
        utility of the player who created it (the parent's active player).
        The root has no creator; its value is only used for visit accounting.
        """
        node = leaf
        while node is not None:
            value_player = node.parent.active_player if node.parent else node.active_player
            node.update_stats(q1 if value_player == "P1" else q2)
            node = node.parent
    
    def _update_best_if_improved(self, child, expanding_player):
        """Update best implementation using best-latest logic."""
        cost = child.get_cost(expanding_player)
        improvement = child.get_improvement(expanding_player)
        
        if cost is None or cost == float('inf'):
            return 0.0
        
        if expanding_player == "P1":
            if cost < self.p1_best_cost:
                self.p1_best_cost = cost
                self.p1_best_code = child.p1_code
                self.p1_best_improvement = improvement
        else:
            if cost < self.p2_best_cost:
                self.p2_best_cost = cost
                self.p2_best_code = child.p2_code
                self.p2_best_improvement = improvement
        
        return improvement
    
    def update_baseline(self, new_baseline: float, remeasure_incumbents: bool = True):
        """
        Update baseline cost and recalculate all improvements in the tree.

        A baseline change means some other strategy was replaced, so every cached
        cost in this tree was measured against a system that no longer exists. The
        two incumbents are re-measured under the new system; without that, a fresh
        candidate only has to beat a stale number from a worse system to be crowned.
        """
        self.baseline_cost = new_baseline

        if remeasure_incumbents:
            self._remeasure_incumbents()

        self._recalculate_improvements(self.root)
        self._decay_tree_statistics(self.root)

        # Recalculate best improvements
        if self.p1_best_cost < float('inf'):
            self.p1_best_improvement = (new_baseline - self.p1_best_cost) / abs(new_baseline) * 100
        else:
            self.p1_best_improvement = 0.0

        if self.p2_best_cost < float('inf'):
            self.p2_best_improvement = (new_baseline - self.p2_best_cost) / abs(new_baseline) * 100
        else:
            self.p2_best_improvement = 0.0

    def _remeasure_incumbents(self):
        """Re-evaluate both players' best code under the current system."""
        for player in ("P1", "P2"):
            code = self.p1_best_code if player == "P1" else self.p2_best_code
            old = self.p1_best_cost if player == "P1" else self.p2_best_cost
            if not code or old == float('inf'):
                continue

            cost, _ = self._run_evaluation(code)
            if cost is None:
                cost = float('inf')

            if player == "P1":
                self.p1_best_cost = cost
            else:
                self.p2_best_cost = cost

            if abs(cost - old) > 1e-9:
                print(f"[REMEASURE] {self.strategy_id} {player}: {old:.6f} -> {cost:.6f}")
    
    def _recalculate_improvements(self, node):
        """
        Recursively recalculate improvements for all nodes.

        Failed implementations keep their -100 penalty: rewriting it to 0.0 made
        broken nodes look neutral (Q ~ 0.5) after every baseline change.
        """
        if node.p1_cost is not None:
            if node.p1_cost == float('inf'):
                node.p1_improvement = -100.0
            else:
                node.p1_improvement = (self.baseline_cost - node.p1_cost) / abs(self.baseline_cost) * 100

        if node.p2_cost is not None:
            if node.p2_cost == float('inf'):
                node.p2_improvement = -100.0
            else:
                node.p2_improvement = (self.baseline_cost - node.p2_cost) / abs(self.baseline_cost) * 100

        for child in node.children:
            self._recalculate_improvements(child)

    def _decay_tree_statistics(self, node, factor: float = 0.5):
        """
        Discount UCT statistics after a baseline change while keeping explored code.

        Clearing them outright set every visit count to zero, and selection then
        always returned the first unvisited child, walking the same leftmost branch
        after every baseline update. Discounting keeps the ordering information and
        just lowers its weight against fresh evidence.
        """
        if node.visits > 0:
            q = node.get_q_value()
            node.visits = max(1, int(node.visits * factor))
            node.total_value = q * node.visits  # keep Q, drop only the confidence
        for child in node.children:
            self._decay_tree_statistics(child, factor)
    
    def get_seed_code(self, node, player: str) -> str:
        """Code the operator should start from: the node's, unless it is not viable."""
        if node.is_viable(player):
            return node.get_code(player)
        fallback = self.p1_best_code if player == "P1" else self.p2_best_code
        print(f"[REPAIR] {self.strategy_id} {player}: node code not viable, "
              f"restarting from this player's best")
        return fallback or node.get_code(player)

    def get_opponent_best_code(self, player: str) -> str:
        """Get opponent's best code (used by operators for context)."""
        if player == "P1":
            return self.p2_best_code
        return self.p1_best_code
    
    def get_opponent_best_improvement(self, player: str) -> float:
        """Get opponent's best improvement (used by operators for context)."""
        if player == "P1":
            return self.p2_best_improvement
        return self.p1_best_improvement

    def get_own_best_improvement(self, player: str) -> float:
        """Get this player's own best improvement."""
        if player == "P1":
            return self.p1_best_improvement
        return self.p2_best_improvement
    
    def get_best_improvement(self) -> float:
        return max(self.p1_best_improvement, self.p2_best_improvement)

    def get_recent_attempts(self, limit: int = 4) -> list:
        return self.recent_attempts[-limit:]

    def get_reflections(self, limit: int = 3) -> list:
        return self.reflection_memory[-limit:]

    def add_reflection(self, reflection: str) -> None:
        if reflection and reflection not in self.reflection_memory:
            self.reflection_memory.append(reflection)
    
    def get_winning_code(self) -> str:
        if self.p1_best_cost <= self.p2_best_cost:
            return self.p1_best_code
        return self.p2_best_code
    
    def get_best_cost(self) -> float:
        return min(self.p1_best_cost, self.p2_best_cost)
    
    def save_best_implementation(self):
        try:
            strategy_path = self._get_strategy_path()
            best_code = self.get_winning_code()
            
            with open(strategy_path, 'w', encoding="utf-8") as f:
                f.write(best_code)
                
        except Exception as e:
            print(f"[SAVE ERROR] Failed to save for {self.strategy_id}: {e}")
