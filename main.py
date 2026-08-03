import os
import sys
import atexit
import hydra
import importlib.util
from omegaconf import DictConfig
from dotenv import load_dotenv

from utils.logger import CompetitiveLogger
from utils.openai import LLMClient
from utils.solver_config import normalize_solver_config
from src.controller import Controller
from src.final_round import FinalRound

load_dotenv()


def snapshot_strategy_sources(solver_config):
    snapshot = {}
    for strategy in solver_config.functions:
        with open(strategy.path, 'r', encoding='utf-8') as f:
            snapshot[strategy.path] = f.read()
    return snapshot


def restore_strategy_sources(snapshot):
    for path, code in snapshot.items():
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code)


def load_prompts(solver_path):
    """Load prompts from solver's prompts.py file."""
    prompts = {}
    prompts_file = os.path.join(solver_path, "prompts.py")
    
    if not os.path.exists(prompts_file):
        print(f"[ERROR] Required prompts file not found: {prompts_file}")
        sys.exit(1)
    
    spec = importlib.util.spec_from_file_location("prompts", prompts_file)
    prompts_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts_module)
    
    for attr_name in dir(prompts_module):
        if not attr_name.startswith('_'):
            attr_value = getattr(prompts_module, attr_name)
            if isinstance(attr_value, str):
                prompts[attr_name] = attr_value
    
    if not prompts:
        print(f"[ERROR] No prompts found in {prompts_file}")
        sys.exit(1)
    
    return prompts


def save_best_strategies(results, solver_config, results_dir):
    """Save best strategies from Round 1 to results directory."""
    print(f"\nSaving Round 1 strategies to {results_dir}...")
    
    os.makedirs(results_dir, exist_ok=True)
    saved_files = []
    
    for strategy_id, result in results.items():
        strategy_info = None
        for func in solver_config.functions:
            if func.id == strategy_id:
                strategy_info = func
                break
        
        if not strategy_info:
            print(f"  Warning: Strategy info not found for {strategy_id}")
            continue
        
        filename = f"{strategy_id}_round1_best.py"
        filepath = os.path.join(results_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Best Round 1 implementation for {strategy_info.name}\n")
                f.write(f"# Strategy ID: {strategy_id}\n")
                f.write(f"# Winner: {result['winner']}\n")
                f.write(f"# Best cost: {result['best_cost']:.6f}\n")
                f.write(f"# P1 cost: {result['p1_cost']:.6f}\n")
                f.write(f"# P2 cost: {result['p2_cost']:.6f}\n\n")
                f.write(result['best_code'])
            
            saved_files.append(filepath)
            print(f"  Saved {strategy_id} to {filename}")
            
        except Exception as e:
            print(f"  Failed to save {strategy_id}: {e}")
    
    print(f"Saved {len(saved_files)} Round 1 strategy files")
    return saved_files


def save_final_strategies(final_combination, solver_config, results_dir):
    """Save Final Round optimized strategies."""
    print(f"\nSaving Final Round strategies to {results_dir}...")
    
    os.makedirs(results_dir, exist_ok=True)
    saved_files = []
    
    for strategy_id, code in final_combination.items():
        strategy_info = None
        for func in solver_config.functions:
            if func.id == strategy_id:
                strategy_info = func
                break
        
        if not strategy_info:
            print(f"  Warning: Strategy info not found for {strategy_id}")
            continue
        
        filename = f"{strategy_id}_final_best.py"
        filepath = os.path.join(results_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Final Round optimized implementation for {strategy_info.name}\n")
                f.write(f"# Strategy ID: {strategy_id}\n")
                f.write(f"# Phase: Final Round (system-aware)\n\n")
                f.write(code)
            
            saved_files.append(filepath)
            print(f"  Saved {strategy_id} to {filename}")
            
        except Exception as e:
            print(f"  Failed to save {strategy_id}: {e}")
    
    print(f"Saved {len(saved_files)} Final Round strategy files")
    return saved_files


@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    """
    Main entry point for Two-Phase Competitive Optimization:
    1. Round 1: MCTS-based individual strategy optimization
    2. Final Round: Sequential system-aware optimization
    """
    solver_config = normalize_solver_config(cfg.solver)
    source_snapshot = snapshot_strategy_sources(solver_config)
    restore_on_exit = lambda: restore_strategy_sources(source_snapshot)
    atexit.register(restore_on_exit)
    problem_name = solver_config.problem
    algorithm_name = solver_config.algorithm
    final_iterations = cfg.mcts.final_iterations
    
    print("Starting Two-Phase Competitive Optimization")
    print(f"Problem: {problem_name} using {algorithm_name}")
    print(f"Round 1: MCTS optimization (individual strategies)")
    print(f"Final Round: Sequential optimization (system-aware)")
    print("Two AI players compete in both phases")
    
    # Initialize LLM client
    client = LLMClient(
        model=cfg.llm.model,
        temperature=cfg.llm.temperature
    )
    
    # Load prompts
    solver_path = solver_config.base_path
    prompts = load_prompts(solver_path)
    
    # Create results directory
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    
    # Initialize logger
    solver_name = f"{problem_name.lower()}_{algorithm_name.lower()}"
    logger = CompetitiveLogger(cfg.paths.results_dir, solver_name)
    
    print(f"\nLogging enabled:")
    print(f"  Round 1: {logger.get_round1_file_path()}")
    print(f"  Final Round: {logger.get_round2_file_path()}")
    
    # ============= ROUND 1: MCTS OPTIMIZATION =============
    print("\n" + "=" * 60)
    print("PHASE 1: ROUND 1 - MCTS OPTIMIZATION")
    print("=" * 60)
    
    controller = Controller(
        client=client,
        prompts=prompts,
        cfg=cfg,
        logger=logger
    )
    
    initial_baseline = controller.initial_baseline_cost
    round1_results = controller.run()
    round1_baseline = controller.current_baseline_cost
    
    print("\nRound 1 completed!")
    print("Round 1 Results Summary:")
    
    round1_improvement = (initial_baseline - round1_baseline) / abs(initial_baseline) * 100
    print(f"  Initial baseline: {initial_baseline:.6f}")
    print(f"  Round 1 baseline: {round1_baseline:.6f}")
    print(f"  Round 1 improvement: {round1_improvement:.2f}%")
    print("  Strategy winners:")
    
    for strategy_id, result in round1_results.items():
        winner = result["winner"]
        best_cost = result["best_cost"]
        print(f"    {strategy_id}: {winner} wins with cost {best_cost:.6f}")
    
    # Save Round 1 strategies
    round1_files = save_best_strategies(round1_results, solver_config, cfg.paths.results_dir)
    
    # ============= FINAL ROUND: SEQUENTIAL OPTIMIZATION =============
    print("\n" + "=" * 60)
    print("PHASE 2: FINAL ROUND - SEQUENTIAL OPTIMIZATION")
    print("=" * 60)
    
    # Extract best implementations from Round 1
    initial_combination = {}
    for strategy_id, result in round1_results.items():
        initial_combination[strategy_id] = result['best_code']
    
    final_round = FinalRound(
        initial_combination=initial_combination,
        problem_config=solver_config,
        client=client,
        macro_baseline=round1_baseline,
        prompts=prompts,
        logger=logger
    )
    
    final_results = final_round.run(final_iterations)
    final_round.save_best_combination()
    
    # Save Final Round strategies
    final_files = save_final_strategies(
        final_results["best_combination"],
        solver_config,
        cfg.paths.results_dir
    )
    
    # ============= SUMMARY =============
    print("\n" + "=" * 60)
    print("TWO-PHASE OPTIMIZATION COMPLETED")
    print("=" * 60)
    
    final_round_improvement = final_results['total_improvement']
    overall_improvement = (
        (initial_baseline - final_results['best_cost']) / abs(initial_baseline) * 100
    )
    
    print("Phase Summary:")
    print(f"  Round 1: {round1_improvement:.2f}% improvement")
    print(f"  Final Round: {final_round_improvement:.2f}% additional improvement")
    print(f"  Overall: {overall_improvement:.2f}% improvement")
    print(f"  Results saved to: {cfg.paths.results_dir}")
    print("  Log files:")
    print(f"    Round 1: {logger.get_round1_file_path()}")
    print(f"    Final Round: {logger.get_round2_file_path()}")
    restore_strategy_sources(source_snapshot)
    atexit.unregister(restore_on_exit)
    print("\nOptimization completed successfully!")


if __name__ == "__main__":
    main()
