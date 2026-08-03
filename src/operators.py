from src.reflector import ContrastiveReflector


class Operators:
    """Three competitive operators for 2-player MCTS: counter, learning, innovation."""
    
    AVAILABLE = ["counter", "learning", "innovation"]
    
    @staticmethod
    def apply(operator: str, node, mcts, client, prompts, strategy_id, baseline_cost):
        """
        Apply operator to generate new code for node.active_player.
        Uses node.active_player (NOT a global current_player).
        """
        active_player = node.active_player
        current_impl = node.get_code(active_player)
        
        system_prompt = prompts.get(
            "SYSTEM_PROMPT",
            (
                "You are an expert in the domain of optimization heuristics. "
                "Your task is to design heuristics that can effectively solve optimization problems."
            )
        )
        task_prompt = prompts.get(strategy_id, "")
        baseline_impl = mcts.controller.get_current_best_implementation(strategy_id)
        
        context = Operators._build_context(
            node, mcts, baseline_cost, current_impl, operator, baseline_impl,
            active_player, task_prompt
        )

        attempts = mcts.get_recent_attempts(1)
        reflection = ""
        mcts.latest_reflection = ""
        if attempts:
            try:
                reflection = ContrastiveReflector.reflect(
                client=client,
                task_prompt=task_prompt,
                challenger=attempts[-1],
                reference_code=mcts.get_winning_code(),
                reference_improvement=max(0.0, mcts.get_best_improvement()),
                    prior_reflections=mcts.get_reflections()
                )
                mcts.add_reflection(reflection)
                mcts.latest_reflection = reflection
                context += f"\n\nCONTRASTIVE REFLECTION:\n{reflection}"
            except Exception as e:
                print(f"[REFLECTION WARNING] {e}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        
        _, code, summary = client.get_code(messages, function_id=strategy_id)
        return code, summary
    
    @staticmethod
    def _build_context(node, mcts, baseline_cost, current_impl, operator_type, baseline_impl, active_player, task_prompt):
        task_section = task_prompt

        baseline_section = f"""BASELINE IMPLEMENTATION:
```python
{baseline_impl}
```"""
        
        current_cost = node.get_cost(active_player)
        current_improvement = node.get_improvement(active_player)
        
        if current_cost == float('inf') or current_cost is None:
            status_info = "FAILED - Implementation has errors"
            improvement_info = f"Improvement: {current_improvement:.2f}% (failed)"
        else:
            status_info = f"Cost: {current_cost:.6f}"
            improvement_info = f"Improvement: {current_improvement:.2f}%"
        
        current_section = f"""CURRENT SOLUTION ({active_player}):
- Status: {status_info}
- {improvement_info}

IMPLEMENTATION:
```python
{current_impl}
```"""
        
        # Get opponent's best (using active_player to determine opponent)
        opponent_best_code = mcts.get_opponent_best_code(active_player)
        opponent_best_improvement = mcts.get_opponent_best_improvement(active_player)
        
        opponent_section = f"""OPPONENT BEST:
- Improvement over baseline: {opponent_best_improvement:.2f}%

```python
{opponent_best_code}
```"""
        
        path_summaries = node.get_path_summaries(max_depth=3)
        history_text = "\n".join(f"- {s}" for s in path_summaries) if path_summaries else "- No moves yet"
        
        history_section = f"""CURRENT SEARCH PATH:
{history_text}
"""

        attempts = mcts.get_recent_attempts(4)
        if attempts:
            attempt_text = "\n".join(
                f"- {a['player']} {a['operator']}: {a['improvement']:+.2f}% — {a['summary']}"
                if a["success"] else f"- {a['player']} {a['operator']}: invalid — {a['summary']}"
                for a in attempts
            )
        else:
            attempt_text = "- No evaluated attempts yet"
        feedback_section = f"""RECENT EVALUATOR FEEDBACK:
{attempt_text}
Avoid repeating failed or non-improving formula families."""

        contrast_section = ""
        if attempts:
            last = attempts[-1]
            last_status = f"{last['improvement']:+.2f}%" if last["success"] else "invalid"
            contrast_section = f"""LAST CHALLENGER ({last_status}):
```python
{last['code']}
```
Compare its concrete formula with the better baseline/opponent before proposing the next code."""

        reflections = mcts.get_reflections(3)
        reflection_text = "\n".join(f"- {x}" for x in reflections) if reflections else "- No reflection yet"
        reflection_section = f"""ACCUMULATED DESIGN INSIGHTS:
{reflection_text}"""
        
        instructions = Operators._get_operator_instructions(operator_type)
        
        instructions_section = f"""---
INSTRUCTION:
{instructions}

GOAL:
Create an implementation that beats both baseline cost ({baseline_cost:.6f}) and the opponent.
Change one decision rule at a time; when feedback is negative, return to the baseline or best positive idea.
Keep reasoning concise (50 words max)."""
        
        return f"{task_section}\n\n{baseline_section}\n\n{current_section}\n\n{opponent_section}\n\n{history_section}\n\n{feedback_section}\n\n{contrast_section}\n\n{reflection_section}\n\n{instructions_section}"
    
    @staticmethod
    def _get_operator_instructions(operator_type):
        if operator_type == "counter":
            return "Counter: target one weakness in the opponent or its evaluator score with a distinct stable formula."
        
        elif operator_type == "learning":
            return "Learning: retain one useful opponent pattern, then change or combine it using evaluator feedback."
        
        elif operator_type == "innovation":
            return "Innovation: try a compact formula family not used by the baseline, opponent, or recent attempts."
        
        return (
            "Optimize the implementation with a concrete heuristic formula that preserves the exact "
            "signature, uses only available inputs, and improves the evaluator cost."
        )
