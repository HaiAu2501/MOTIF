class ContrastiveReflector:
    """Produce a short verbal gradient from evaluated heuristic code."""

    @staticmethod
    def reflect(client, task_prompt: str, challenger: dict, reference_code: str,
                reference_improvement: float = 0.0, prior_reflections: list = None) -> str:
        challenger_improvement = challenger.get("improvement", -100.0)
        challenger_code = challenger.get("code", "")

        if challenger_improvement > reference_improvement:
            worse_code, worse_score = reference_code, reference_improvement
            better_code, better_score = challenger_code, challenger_improvement
        else:
            worse_code, worse_score = challenger_code, challenger_improvement
            better_code, better_score = reference_code, reference_improvement

        memory = "\n".join(f"- {x}" for x in (prior_reflections or [])[-3:])
        memory_section = f"\nPrior insights:\n{memory}" if memory else ""
        user = f"""{task_prompt}

Compare two evaluated implementations.

WORSE ({worse_score:+.2f}%):
```python
{worse_code}
```

BETTER ({better_score:+.2f}%):
```python
{better_code}
```
{memory_section}

Give one actionable design insight explaining the performance difference. Use at most 25 words."""
        messages = [
            {"role": "system", "content": "You reflect on evaluated optimization heuristics and give precise design hints."},
            {"role": "user", "content": user}
        ]
        reflection = client.get_response(messages, temperature=0.3).strip()
        return " ".join(reflection.split()[:25])
