import os
import sys
import subprocess
import math

import numpy as np

from src.mcts import CompetitiveMCTS


class Controller:
    """
    Controller for Round 1: manages multiple CompetitiveMCTS trees (one per strategy).
    Uses UCB to select which tree to run in each outer iteration.
    Maintains a shared baseline that updates when any tree finds improvement.
    """
    
    def __init__(self, client, prompts, cfg, logger=None):
        self.client = client
        self.prompts = prompts
        self.cfg = cfg
        self.logger = logger
        
        self.solver_config = cfg.solver
        self.outer_iterations = cfg.mcts.outer_iterations
        self.inner_iterations = cfg.mcts.inner_iterations
        
        # UCB parameters for tree selection
        self.ucb_c = 1.414
        
        # Tree management
        self.trees = {}           # {strategy_id: CompetitiveMCTS}
        self.tree_visits = {}     # {strategy_id: visit_count}
        self.tree_rewards = {}    # {strategy_id: total_reward}
        
        # Baseline management
        self.current_baseline = None
        self.initial_baseline = None
        self.best_implementations = {}  # {strategy_id: best_code}
        
        self._initialize_baseline()
        self._initialize_trees()
        
        if self.logger:
            problem_name = f"{self.solver_config.problem}_{self.solver_config.algorithm}"
            self.logger.init_round1(
                problem=problem_name,
                initial_baseline=self.initial_baseline,
                outer_iterations=self.outer_iterations,
                inner_iterations=self.inner_iterations
            )
        
        print("Initialized Controller with dynamic baseline")
    
    def _initialize_baseline(self):
        print("[BASELINE] Evaluating initial implementation...")
        
        result = subprocess.run(
            [sys.executable, self.solver_config.eval_script, "train"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120
        )
        
        if result.stderr.strip():
            print(f"[BASELINE ERROR] {result.stderr.strip()}")
        
        if not result.stdout.strip():
            print("[BASELINE ERROR] No output from evaluation")
            return
        
        lines = result.stdout.strip().split('\n')
        self.current_baseline = float(lines[-1])
        self.initial_baseline = self.current_baseline
        
        print(f"[BASELINE] Initial cost: {self.current_baseline}")
    
    def _initialize_trees(self):
        print("\n" + "=" * 50)
        print("INITIALIZING COMPETITIVE STRATEGY TREES")
        print("=" * 50)
        
        for strategy in self.solver_config.functions:
            strategy_id = strategy.id
            strategy_name = strategy.name
            strategy_path = strategy.path
            
            with open(strategy_path, 'r', encoding="utf-8") as f:
                initial_code = f.read()
            
            self.best_implementations[strategy_id] = initial_code
            
            mcts = CompetitiveMCTS(
                function_name=strategy_name,
                strategy_id=strategy_id,
                initial_code=initial_code,
                client=self.client,
                prompts=self.prompts,
                problem_config=self.solver_config,
                baseline_cost=self.current_baseline,
                controller=self
            )
            
            self.trees[strategy_id] = mcts
            self.tree_visits[strategy_id] = 0
            self.tree_rewards[strategy_id] = 0.0
            
            print(f"  Initialized tree for {strategy_id} ({strategy_name})")
        
        print("=" * 50 + "\n")
    
    def _select_tree_ucb(self):
        if not self.trees:
            return None
        
        strategy_ids = list(self.trees.keys())
        total_visits = sum(self.tree_visits.values()) + 1
        
        best_strategy = None
        best_ucb = float('-inf')
        
        for strategy_id in strategy_ids:
            visits = self.tree_visits[strategy_id]
            
            if visits == 0:
                return strategy_id
            
            exploitation = self.tree_rewards[strategy_id] / visits
            exploration = self.ucb_c * math.sqrt(math.log(total_visits) / visits)
            ucb_value = exploitation + exploration
            
            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_strategy = strategy_id
        
        return best_strategy
    
    def _run_inner_iterations(self, strategy_id, outer_iteration):
        mcts = self.trees[strategy_id]
        best_improvement = 0.0
        
        print(f"\n[STRATEGY] Running {self.inner_iterations} iterations on {strategy_id}")
        print(f"  Current baseline: {self.current_baseline:.6f}")
        
        for i in range(self.inner_iterations):
            result = mcts.run_iteration()
            inner_turn = i + 1
            
            if result["success"]:
                improvement = result["improvement"]
                if improvement > best_improvement:
                    best_improvement = improvement
                
                print(f"  Turn {mcts.turn_count}: {result['player']} used {result['operator']}")
                print(f"    Summary: {result['summary']}")
                print(f"    Improvement: {improvement:.2f}%")
                
                if self.logger:
                    player = result['player']
                    eval_cost = result.get('cost')
                    eval_success = eval_cost is not None and eval_cost != float('inf')
                    
                    self.logger.log_round1_turn(
                        outer_iteration=outer_iteration,
                        inner_turn=inner_turn,
                        selected_strategy=strategy_id,
                        current_baseline=self.current_baseline,
                        player=player,
                        operator=result['operator'],
                        llm_code=result.get('code', ''),
                        llm_summary=result['summary'],
                        eval_cost=eval_cost if eval_success else None,
                        eval_improvement=improvement,
                        eval_success=eval_success
                    )
            else:
                print(f"  Turn {mcts.turn_count}: failed ({result.get('reason', 'unknown')})")
                
                if self.logger:
                    self.logger.log_round1_turn(
                        outer_iteration=outer_iteration,
                        inner_turn=inner_turn,
                        selected_strategy=strategy_id,
                        current_baseline=self.current_baseline,
                        player="unknown",
                        operator="failed",
                        llm_code="",
                        llm_summary=f"Turn failed: {result.get('reason', 'unknown')}",
                        eval_cost=None,
                        eval_improvement=0.0,
                        eval_success=False
                    )
        
        return best_improvement
    
    def _check_and_update_baseline(self, strategy_id):
        """Check if strategy improved and update baseline if so."""
        print(f"[BASELINE] Checking {strategy_id} results...")
        
        mcts = self.trees[strategy_id]
        best_cost = mcts.get_best_cost()
        
        if best_cost < self.current_baseline:
            improvement_pct = (self.current_baseline - best_cost) / abs(self.current_baseline) * 100
            
            print(f"[BASELINE] Found improvement!")
            print(f"  Previous: {self.current_baseline:.6f}")
            print(f"  New: {best_cost:.6f}")
            print(f"  Improvement: {improvement_pct:.2f}%")
            
            # Update best implementation
            best_code = mcts.get_winning_code()
            self.best_implementations[strategy_id] = best_code
            
            # Save to file
            self._save_implementation(strategy_id, best_code)
            
            # Update baseline in all trees
            old_baseline = self.current_baseline
            self.current_baseline = best_cost
            self._propagate_baseline_update()
            
            return True, best_cost, improvement_pct
        else:
            print(f"[BASELINE] No improvement")
            print(f"  Current baseline: {self.current_baseline:.6f}")
            print(f"  Best achieved: {best_cost:.6f}")
            
            # Revert file to previous best
            self._revert_implementation(strategy_id)
            
            return False, self.current_baseline, 0.0
    
    def _save_implementation(self, strategy_id, code):
        strategy_path = None
        for func in self.solver_config.functions:
            if func.id == strategy_id:
                strategy_path = func.path
                break
        
        if strategy_path:
            with open(strategy_path, 'w', encoding="utf-8") as f:
                f.write(code)
            print(f"[SAVE] Saved best implementation for {strategy_id}")
    
    def _revert_implementation(self, strategy_id):
        if strategy_id not in self.best_implementations:
            return
        
        best_impl = self.best_implementations[strategy_id]
        self._save_implementation(strategy_id, best_impl)
        print(f"[REVERT] Reverted {strategy_id} to previous best")
    
    def _propagate_baseline_update(self):
        """Update baseline in all trees and reset baseline-dependent UCT values."""
        print(f"[BASELINE] Propagating update to all trees...")
        
        for strategy_id, mcts in self.trees.items():
            mcts.update_baseline(self.current_baseline)
        
        print(f"[BASELINE] All trees updated")
    
    def get_current_best_implementation(self, strategy_id):
        return self.best_implementations.get(strategy_id, "")
    
    def _update_tree_stats(self, strategy_id, improved, improvement_pct):
        self.tree_visits[strategy_id] += 1
        
        if improved:
            reward = 1 / (1 + math.exp(np.clip(-improvement_pct * 10, -50, 50)))
        else:
            reward = 0.3  # Small reward to avoid complete avoidance
        
        self.tree_rewards[strategy_id] += reward
    
    def _print_status(self):
        print("\n" + "=" * 50)
        print("COMPETITIVE STATUS")
        print("=" * 50)
        
        for strategy_id, mcts in self.trees.items():
            winner = "P1" if mcts.p1_best_cost <= mcts.p2_best_cost else "P2"
            best_cost = mcts.get_best_cost()
            
            p1_cost_str = "FAILED" if mcts.p1_best_cost == float('inf') else f"{mcts.p1_best_cost:.6f}"
            p2_cost_str = "FAILED" if mcts.p2_best_cost == float('inf') else f"{mcts.p2_best_cost:.6f}"
            best_cost_str = "FAILED" if best_cost == float('inf') else f"{best_cost:.6f}"
            
            print(f"{strategy_id}:")
            print(f"  P1: {p1_cost_str} (improvement: {mcts.p1_best_improvement:.2f}%)")
            print(f"  P2: {p2_cost_str} (improvement: {mcts.p2_best_improvement:.2f}%)")
            print(f"  Winner: {winner} with cost {best_cost_str}")
            print(f"  Turns: {mcts.turn_count}")
            print()
        
        print(f"Current baseline: {self.current_baseline:.6f}")
        print("=" * 50)
    
    def _save_all_best_implementations(self):
        print("\nSaving all best implementations...")
        
        for strategy_id, mcts in self.trees.items():
            mcts.save_best_implementation()
            print(f"  Saved {strategy_id}")
        
        print("All implementations saved")
    
    def run(self):
        """Run Round 1 optimization."""
        print("\n" + "=" * 50)
        print("STARTING ROUND 1: MCTS OPTIMIZATION")
        print(f"Initial baseline: {self.current_baseline}")
        print(f"Outer iterations: {self.outer_iterations}")
        print("=" * 50)
        
        for iteration in range(self.outer_iterations):
            outer_iter = iteration + 1
            print(f"\n{'='*20} OUTER ITERATION {outer_iter}/{self.outer_iterations} {'='*20}")
            
            # 1. Select tree using UCB
            selected_strategy = self._select_tree_ucb()
            
            if not selected_strategy:
                print("[ERROR] No strategy selected")
                break
            
            print(f"[UCB] Selected: {selected_strategy}")
            
            # 2. Run inner iterations
            self._run_inner_iterations(selected_strategy, outer_iter)
            
            # 3. Check for improvement and update baseline
            improved, new_baseline, improvement_pct = self._check_and_update_baseline(selected_strategy)
            
            # 4. Log baseline update
            if self.logger and improved:
                if self.logger.round1_data["turns"]:
                    last_turn = self.logger.round1_data["turns"][-1]
                    last_turn["baseline_updated"] = True
                    last_turn["new_baseline"] = new_baseline
                    self.logger._save_round1()
            
            # 5. Update UCB stats
            self._update_tree_stats(selected_strategy, improved, improvement_pct)
            
            # Print status on last iteration
            if iteration == self.outer_iterations - 1:
                self._print_status()
        
        # Final evaluation
        print("\n" + "=" * 50)
        print("ROUND 1 COMPLETED")
        print("=" * 50)
        
        final_improvement = (self.initial_baseline - self.current_baseline) / abs(self.initial_baseline) * 100
        print(f"Initial baseline: {self.initial_baseline:.6f}")
        print(f"Final baseline: {self.current_baseline:.6f}")
        print(f"Total improvement: {final_improvement:.2f}%")
        
        self._print_status()
        self._save_all_best_implementations()
        
        # Collect results
        results = {}
        for strategy_id, mcts in self.trees.items():
            winner = "P1" if mcts.p1_best_cost <= mcts.p2_best_cost else "P2"
            
            results[strategy_id] = {
                "best_code": mcts.get_winning_code(),
                "best_cost": mcts.get_best_cost(),
                "p1_cost": mcts.p1_best_cost,
                "p2_cost": mcts.p2_best_cost,
                "winner": winner
            }
        
        return results
    
    @property
    def initial_baseline_cost(self):
        return self.initial_baseline
    
    @property
    def current_baseline_cost(self):
        return self.current_baseline
