import math

class Node:
    """
    Node for competitive 2-player MCTS.
    Each node stores two code versions (P1 and P2) and tracks which player's turn it is.
    Only one W/Q value is stored per node. For non-root nodes, this value is from
    the perspective of the player who created the node (the parent's active player).
    """
    
    def __init__(self, p1_code: str, p2_code: str, function_name: str, parent=None, 
                 depth: int = 0, active_player: str = "P1", operator: str = None, summary: str = None):
        self.p1_code = p1_code
        self.p2_code = p2_code
        self.function_name = function_name
        self.parent = parent
        self.children = []
        self.depth = depth
        self.active_player = active_player  # Player whose turn it is at this node
        self.operator = operator  # Operator used to create this node
        self.summary = summary
        
        # Performance metrics for each player's code
        self.p1_cost = None
        self.p2_cost = None
        self.p1_improvement = 0.0
        self.p2_improvement = 0.0

        # Viability: False once a player's code here failed or was catastrophically
        # worse than the baseline. Non-viable code must not seed further expansions.
        self.p1_viable = True
        self.p2_viable = True
        
        # MCTS statistics - single W and n per node
        self.visits = 0
        self.total_value = 0.0
        
        # Operator tracking for expansion
        self.used_operators = set()
        
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        
    def get_q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits
    
    def update_stats(self, q_value: float):
        self.visits += 1
        self.total_value += q_value
    
    def get_unvisited_operators(self, available_operators: list) -> list:
        return [op for op in available_operators if op not in self.used_operators]
    
    def mark_operator_used(self, operator: str):
        self.used_operators.add(operator)
    
    def get_code(self, player: str) -> str:
        return self.p1_code if player == "P1" else self.p2_code
    
    def get_opponent_code(self, player: str) -> str:
        return self.p2_code if player == "P1" else self.p1_code
    
    def get_improvement(self, player: str) -> float:
        return self.p1_improvement if player == "P1" else self.p2_improvement
    
    def get_cost(self, player: str) -> float:
        return self.p1_cost if player == "P1" else self.p2_cost
    
    def set_cost(self, player: str, cost: float):
        if player == "P1":
            self.p1_cost = cost
        else:
            self.p2_cost = cost
    
    def set_improvement(self, player: str, improvement: float):
        if player == "P1":
            self.p1_improvement = improvement
        else:
            self.p2_improvement = improvement

    def is_viable(self, player: str) -> bool:
        return self.p1_viable if player == "P1" else self.p2_viable

    def set_viable(self, player: str, viable: bool):
        if player == "P1":
            self.p1_viable = viable
        else:
            self.p2_viable = viable

    def get_path_summaries(self, max_depth: int = 3) -> list:
        """Get summaries from this node back to root, limited to max_depth."""
        summaries = []
        node = self
        while node is not None and len(summaries) < max_depth:
            if node.summary:
                summaries.append(node.summary)
            node = node.parent
        return summaries[::-1]
