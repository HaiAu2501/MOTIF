import subprocess
import sys

from src.final_operators import FinalOperators


class FinalRound:
    """
    Final Round: Sequential optimization with full system awareness.
    Not MCTS - simply iterates through each strategy with alternating players.
    """
    
    def __init__(self, initial_combination: dict, problem_config, client, 
                 macro_baseline: float, prompts=None, logger=None):
        self.problem_config = problem_config
        self.client = client
        self.macro_baseline = macro_baseline
        self.logger = logger
        self.prompts = prompts or {}
        
        self.strategy_ids = list(initial_combination.keys())
        
        # Global baseline (updated after each strategy completes)
        self.global_baseline_combination = initial_combination.copy()
        self.global_baseline_cost = macro_baseline
        
        # Strategy-level baseline (fixed during single strategy optimization)
        self.strategy_baseline_combination = None
        self.strategy_baseline_cost = None
        
        # Best tracking
        self.best_combination = initial_combination.copy()
        self.best_cost = macro_baseline
    
    def _evaluate(self, combination: dict) -> float:
        """Evaluate a strategy combination."""
        # Backup original files
        original_files = {}
        for strategy in self.problem_config.functions:
            with open(strategy.path, 'r', encoding='utf-8') as f:
                original_files[strategy.id] = f.read()
        
        try:
            # Write combination to files
            for strategy in self.problem_config.functions:
                if strategy.id in combination:
                    with open(strategy.path, 'w', encoding='utf-8') as f:
                        f.write(combination[strategy.id])
            
            # Run evaluation
            result = subprocess.run(
                [sys.executable, self.problem_config.eval_script, "train"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120
            )
            
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                return float(lines[-1])
            return float('inf')
            
        finally:
            # Restore original files
            for strategy in self.problem_config.functions:
                if strategy.id in original_files:
                    with open(strategy.path, 'w', encoding='utf-8') as f:
                        f.write(original_files[strategy.id])
    
    def _optimize_strategy(self, target_strategy: str, num_iterations: int) -> dict:
        """Run sequential optimization for one strategy."""
        print(f"\n{'='*50}")
        print(f"Final Round: Optimizing {target_strategy}")
        print(f"{'='*50}")
        
        # Set fixed baseline for this strategy
        self.strategy_baseline_combination = self.global_baseline_combination.copy()
        self.strategy_baseline_cost = self.global_baseline_cost
        
        print(f"  Strategy baseline: {self.strategy_baseline_cost:.6f}")
        
        # Track player results
        baseline_code = self.strategy_baseline_combination[target_strategy]
        p1_best_cost = self.strategy_baseline_cost
        p1_best_code = baseline_code
        p1_best_improvement = 0.0
        p2_best_cost = self.strategy_baseline_cost
        p2_best_code = baseline_code
        p2_best_improvement = 0.0
        
        attempt_feedback = []
        current_player = "P1"
        
        for turn in range(num_iterations):
            try:
                # Get opponent's best for context
                if current_player == "P1":
                    opponent_best_code = p2_best_code
                    opponent_best_improvement = p2_best_improvement
                else:
                    opponent_best_code = p1_best_code
                    opponent_best_improvement = p1_best_improvement
                
                # Each player refines its own incumbent while seeing the opponent's best.
                player_code = p1_best_code if current_player == "P1" else p2_best_code
                proposal_combination = self.global_baseline_combination.copy()
                if player_code is not None:
                    proposal_combination[target_strategy] = player_code

                # Generate new code
                new_code, summary = FinalOperators.apply(
                    current_combination=proposal_combination,
                    target_strategy=target_strategy,
                    client=self.client,
                    baseline_combination=self.strategy_baseline_combination,
                    baseline_cost=self.strategy_baseline_cost,
                    player=current_player,
                    opponent_best_code=opponent_best_code,
                    opponent_best_improvement=opponent_best_improvement,
                    attempt_feedback=attempt_feedback,
                    task_prompt=self.prompts.get(target_strategy, "")
                )
                
                # Create new combination
                new_combination = self.global_baseline_combination.copy()
                new_combination[target_strategy] = new_code
                
                # Evaluate
                cost = self._evaluate(new_combination)
                
                if cost == float('inf'):
                    improvement = -100.0
                else:
                    improvement = (self.strategy_baseline_cost - cost) / abs(self.strategy_baseline_cost) * 100
                
                # Update player's best
                if current_player == "P1":
                    if cost < p1_best_cost:
                        p1_best_cost = cost
                        p1_best_code = new_code
                        p1_best_improvement = improvement
                else:
                    if cost < p2_best_cost:
                        p2_best_cost = cost
                        p2_best_code = new_code
                        p2_best_improvement = improvement
                
                status = f"{improvement:+.2f}%" if cost != float('inf') else "invalid"
                attempt_feedback.append({
                    "player": current_player,
                    "status": status,
                    "summary": summary,
                    "code": new_code
                })
                
                # Update global best if improved
                if cost < self.best_cost:
                    self.best_combination = new_combination.copy()
                    self.best_cost = cost
                
                # Log
                vs_baseline = "BEAT" if improvement > 0 else "BELOW"
                print(f"  Turn {turn + 1}: {current_player} - {summary}")
                print(f"    Cost: {cost:.6f}, Improvement: {improvement:.2f}% ({vs_baseline})")
                
                if self.logger:
                    self.logger.log_round2_turn(
                        strategy=target_strategy,
                        strategy_baseline=self.strategy_baseline_cost,
                        turn=turn + 1,
                        player=current_player,
                        llm_code=new_code,
                        llm_summary=summary,
                        eval_cost=cost if cost != float('inf') else None,
                        eval_improvement=improvement,
                        eval_success=cost != float('inf')
                    )
                
            except Exception as e:
                print(f"  Turn {turn + 1}: {current_player} failed - {e}")
                
                if self.logger:
                    self.logger.log_round2_turn(
                        strategy=target_strategy,
                        strategy_baseline=self.strategy_baseline_cost,
                        turn=turn + 1,
                        player=current_player,
                        llm_code="",
                        llm_summary="Failed",
                        eval_cost=None,
                        eval_improvement=-100.0,
                        eval_success=False
                    )
            
            # Switch player
            current_player = "P2" if current_player == "P1" else "P1"
        
        # Determine winner
        winner = "P1" if p1_best_cost <= p2_best_cost else "P2"
        winner_cost = min(p1_best_cost, p2_best_cost)
        winner_code = p1_best_code if winner == "P1" else p2_best_code
        
        print(f"\n  {target_strategy} Results:")
        print(f"    P1 best: {p1_best_cost:.6f}")
        print(f"    P2 best: {p2_best_cost:.6f}")
        print(f"    Winner: {winner}")
        
        # Update global baseline if improved
        if winner_cost < self.global_baseline_cost and winner_code:
            old_baseline = self.global_baseline_cost
            self.global_baseline_combination[target_strategy] = winner_code
            self.global_baseline_cost = winner_cost
            
            improvement_pct = (old_baseline - winner_cost) / abs(old_baseline) * 100
            print(f"    Global baseline updated: {old_baseline:.6f} -> {winner_cost:.6f} ({improvement_pct:.2f}%)")
        else:
            print(f"    No improvement to global baseline")
        
        return {
            "strategy": target_strategy,
            "winner": winner,
            "winner_cost": winner_cost,
            "p1_best": p1_best_cost,
            "p2_best": p2_best_cost,
            "improved": winner_cost < self.strategy_baseline_cost
        }
    
    def run(self, final_iterations: int) -> dict:
        """Run complete Final Round through all strategies."""
        print("\n" + "=" * 60)
        print("STARTING FINAL ROUND: SEQUENTIAL OPTIMIZATION")
        print("=" * 60)
        print("Goal: System-aware strategy refinement")
        print("Two players compete per strategy with fixed baseline")
        print("Strategies optimized sequentially with baseline updates")
        
        if self.logger:
            problem_name = f"{self.problem_config.problem}_{self.problem_config.algorithm}"
            self.logger.init_round2(
                problem=problem_name,
                macro_baseline=self.macro_baseline,
                final_iterations=final_iterations
            )
        
        initial_cost = self.global_baseline_cost
        strategy_results = []
        
        # Optimize each strategy sequentially
        for strategy_id in self.strategy_ids:
            result = self._optimize_strategy(strategy_id, final_iterations)
            strategy_results.append(result)
        
        # Final summary
        total_improvement = (initial_cost - self.best_cost) / abs(initial_cost) * 100
        
        print("\n" + "=" * 60)
        print("FINAL ROUND COMPLETED")
        print("=" * 60)
        print(f"Initial cost: {initial_cost:.6f}")
        print(f"Final cost: {self.best_cost:.6f}")
        print(f"Total improvement: {total_improvement:.2f}%")
        
        print("\nStrategy Winners:")
        for result in strategy_results:
            print(f"  {result['strategy']}: {result['winner']} with {result['winner_cost']:.6f}")
        
        return {
            "best_combination": self.best_combination,
            "best_cost": self.best_cost,
            "total_improvement": total_improvement,
            "strategy_results": strategy_results
        }
    
    def save_best_combination(self):
        """Save best combination to strategy files."""
        try:
            for strategy in self.problem_config.functions:
                if strategy.id in self.best_combination:
                    with open(strategy.path, 'w', encoding='utf-8') as f:
                        f.write(self.best_combination[strategy.id])
            print("Best combination saved to files")
        except Exception as e:
            print(f"[SAVE ERROR] {e}")
