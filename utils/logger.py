import json
import os
from datetime import datetime
from typing import Optional
import threading


class CompetitiveLogger:
    """Logger for competitive MCTS rounds with JSON output. Thread-safe."""
    
    def __init__(self, results_dir: str, solver_name: str):
        self.results_dir = results_dir
        self.solver_name = solver_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.round1_lock = threading.Lock()
        self.round2_lock = threading.Lock()
        
        self.round1_file = os.path.join(
            results_dir, f"{self.timestamp}_{solver_name}_round_1.json"
        )
        self.round2_file = os.path.join(
            results_dir, f"{self.timestamp}_{solver_name}_round_2.json"
        )
        
        self.round1_data = {"experiment_metadata": {}, "turns": []}
        self.round2_data = {"experiment_metadata": {}, "turns": []}
        
        os.makedirs(results_dir, exist_ok=True)
    
    def init_round1(self, problem: str, initial_baseline: float,
                    outer_iterations: int, inner_iterations: int):
        with self.round1_lock:
            self.round1_data["experiment_metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "problem": problem,
                "initial_baseline": initial_baseline,
                "outer_iterations": outer_iterations,
                "inner_iterations": inner_iterations
            }
            self._save_round1()
    
    def init_round2(self, problem: str, macro_baseline: float, final_iterations: int):
        with self.round2_lock:
            self.round2_data["experiment_metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "problem": problem,
                "macro_baseline": macro_baseline,
                "final_iterations": final_iterations
            }
            self._save_round2()
    
    def log_round1_turn(self, outer_iteration: int, inner_turn: int,
                        selected_strategy: str, current_baseline: float,
                        player: str, operator: str, llm_code: str,
                        llm_summary: str, eval_cost: Optional[float],
                        eval_improvement: float, eval_success: bool,
                        llm_reflection: str = "",
                        baseline_updated: bool = False,
                        new_baseline: Optional[float] = None):
        turn_data = {
            "outer_iteration": outer_iteration,
            "inner_turn": inner_turn,
            "selected_strategy": selected_strategy,
            "current_baseline": current_baseline,
            "player": player,
            "operator": operator,
            "llm_code": llm_code,
            "llm_summary": llm_summary,
            "llm_reflection": llm_reflection,
            "eval_cost": eval_cost,
            "eval_improvement": eval_improvement,
            "eval_success": eval_success,
            "baseline_updated": baseline_updated
        }
        
        if new_baseline is not None:
            turn_data["new_baseline"] = new_baseline
        
        with self.round1_lock:
            self.round1_data["turns"].append(turn_data)
            self._save_round1()
    
    def log_round2_turn(self, strategy: str, strategy_baseline: float,
                        turn: int, player: str, llm_code: str,
                        llm_summary: str, eval_cost: Optional[float],
                        eval_improvement: float, eval_success: bool,
                        llm_reflection: str = ""):
        turn_data = {
            "strategy": strategy,
            "strategy_baseline": strategy_baseline,
            "turn": turn,
            "player": player,
            "llm_code": llm_code,
            "llm_summary": llm_summary,
            "llm_reflection": llm_reflection,
            "eval_cost": eval_cost,
            "eval_improvement": eval_improvement,
            "eval_success": eval_success
        }
        
        with self.round2_lock:
            self.round2_data["turns"].append(turn_data)
            self._save_round2()
    
    def _save_round1(self):
        try:
            with open(self.round1_file, 'w', encoding='utf-8') as f:
                json.dump(self.round1_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[LOGGER ERROR] Failed to save Round 1 log: {e}")
    
    def _save_round2(self):
        try:
            with open(self.round2_file, 'w', encoding='utf-8') as f:
                json.dump(self.round2_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[LOGGER ERROR] Failed to save Round 2 log: {e}")
    
    def get_round1_file_path(self) -> str:
        return self.round1_file
    
    def get_round2_file_path(self) -> str:
        return self.round2_file
