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
        self.recent_attempts = []
        
        # Q-value parameters
        self.lambda_factor = 0.7
        self.k_factor = 1.0
        self.ucb_c = 1.414
    
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
        
        for child in node.children:
            if child.visits == 0:
                # Unvisited child gets priority
                return child
            
            q_value = child.get_q_value()
            exploration = self.ucb_c * math.sqrt(math.log(parent_visits) / child.visits)
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
            # Apply operator to generate new code
            new_code, summary = Operators.apply(
                operator, node, self, self.client, self.prompts,
                self.strategy_id, self.baseline_cost
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
        new_cost = self._evaluate_code(child, expanding_player)
        
        if new_cost is None or new_cost == float('inf'):
            # Evaluation failed: assign a strong negative improvement so invalid
            # implementations do not receive neutral sigmoid reward.
            child.set_cost(expanding_player, float('inf'))
            child.set_improvement(expanding_player, -100.0)
        else:
            child.set_cost(expanding_player, new_cost)
            improvement = (self.baseline_cost - new_cost) / abs(self.baseline_cost) * 100
            child.set_improvement(expanding_player, improvement)

        self.recent_attempts.append({
            "player": expanding_player,
            "operator": child.operator,
            "summary": child.summary,
            "code": child.get_code(expanding_player),
            "improvement": child.get_improvement(expanding_player),
            "success": new_cost is not None and new_cost != float('inf')
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
    
    def _compute_q_for_player(self, my_improvement, opponent_improvement):
        """
        Q = lambda * sigmoid(k * I) + (1 - lambda) * sigmoid(k * (I - I_opponent))
        """
        # Clip to avoid overflow
        baseline_term = 1 / (1 + math.exp(np.clip(-self.k_factor * my_improvement, -50, 50)))
        relative_term = 1 / (1 + math.exp(np.clip(-self.k_factor * (my_improvement - opponent_improvement), -50, 50)))
        
        return self.lambda_factor * baseline_term + (1 - self.lambda_factor) * relative_term
    
    def _evaluate_code(self, child, player):
        """Evaluate player's code by writing to file, running eval, and reverting."""
        try:
            strategy_path = self._get_strategy_path()
            
            # Backup original
            with open(strategy_path, 'r', encoding="utf-8") as f:
                original_code = f.read()
            
            try:
                # Write player's new code
                code_to_eval = child.get_code(player)
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
                
                if result.stderr.strip():
                    print(f"[EVAL WARNING] {result.stderr.strip()}")
                
                if not result.stdout.strip():
                    return None
                
                lines = result.stdout.strip().split('\n')
                return float(lines[-1])
                
            finally:
                # Always revert
                with open(strategy_path, 'w', encoding="utf-8") as f:
                    f.write(original_code)
                    
        except Exception as e:
            print(f"[EVALUATION ERROR] {e}")
            return None
    
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
    
    def update_baseline(self, new_baseline: float):
        """Update baseline cost and recalculate all improvements in the tree."""
        self.baseline_cost = new_baseline
        self._recalculate_improvements(self.root)
        self._reset_tree_statistics(self.root)
        
        # Recalculate best improvements
        if self.p1_best_cost < float('inf'):
            self.p1_best_improvement = (new_baseline - self.p1_best_cost) / abs(new_baseline) * 100
        else:
            self.p1_best_improvement = 0.0
            
        if self.p2_best_cost < float('inf'):
            self.p2_best_improvement = (new_baseline - self.p2_best_cost) / abs(new_baseline) * 100
        else:
            self.p2_best_improvement = 0.0
    
    def _recalculate_improvements(self, node):
        """Recursively recalculate improvements for all nodes."""
        if node.p1_cost is not None:
            if node.p1_cost == float('inf'):
                node.p1_improvement = 0.0
            else:
                node.p1_improvement = (self.baseline_cost - node.p1_cost) / abs(self.baseline_cost) * 100
        
        if node.p2_cost is not None:
            if node.p2_cost == float('inf'):
                node.p2_improvement = 0.0
            else:
                node.p2_improvement = (self.baseline_cost - node.p2_cost) / abs(self.baseline_cost) * 100
        
        for child in node.children:
            self._recalculate_improvements(child)

    def _reset_tree_statistics(self, node):
        """Reset UCT statistics after a baseline change while keeping explored code."""
        node.visits = 0
        node.total_value = 0.0
        for child in node.children:
            self._reset_tree_statistics(child)
    
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
    
    def get_best_improvement(self) -> float:
        return max(self.p1_best_improvement, self.p2_best_improvement)

    def get_recent_attempts(self, limit: int = 4) -> list:
        return self.recent_attempts[-limit:]
    
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
