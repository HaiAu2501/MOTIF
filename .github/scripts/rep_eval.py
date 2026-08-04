"""Evaluate one problem at one instance size, for the reproduction workflows.

The per-problem ``eval.py`` scripts hard-code their full test size list and print an
aggregate, which cannot be sharded across parallel CI jobs. This calls the same
``process_file`` they use, for a single size, and emits one JSON line.

    python .github/scripts/rep_eval.py --problem tsp_aco --size 100

MKP and OP are maximisation problems: their ``process_file`` returns collected value,
and their ``eval.py`` negates it so MOTIF can minimise. ``objective`` below is always
reported in the problem's natural direction, and ``better`` says which way is good.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

# pattern: test dataset filename; kwargs: what this problem's process_file expects
CONFIG = {
    "tsp_aco":  dict(pattern="test_TSP{size}.npy",  kwargs=dict(n_ants=50, n_iter=200), maximize=False),
    "cvrp_aco": dict(pattern="test_CVRP{size}.npy", kwargs=dict(n_ants=30, n_iter=100), maximize=False),
    "op_aco":   dict(pattern="test_OP{size}.npz",   kwargs=dict(n_ants=20, n_iter=100), maximize=True),
    "mkp_aco":  dict(pattern="test_MKP{size}.npz",  kwargs=dict(n_ants=10, n_iter=50),  maximize=True),
    "bpp_aco":  dict(pattern="test_BPP{size}.npz",  kwargs=dict(n_ants=20, n_iter=50),  maximize=False),
    "tsp_gls":  dict(pattern="test_TSP{size}.npy",
                     kwargs=dict(perturbation_moves=30, iter_limit=3000), maximize=False),
    "tsp_dr":   dict(pattern="test_TSP{size}.npy",  kwargs={}, maximize=False),
    "cvrp_dr":  dict(pattern="test_CVRP{size}.npy", kwargs={}, maximize=False),
    "bpp_dr":   dict(pattern="test_BPP{size}.npz",
                     kwargs=dict(max_workers=20, destruction_rate=0.3), maximize=False),
}


def load_eval_module(problem_dir):
    """Import the problem's eval.py with its own directory on sys.path.

    Every solver module does a bare ``from F1 import ...`` at import time, so the
    strategy files must already be in place before this is called.
    """
    path = os.path.join(problem_dir, "eval.py")
    spec = importlib.util.spec_from_file_location("problem_eval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["problem_eval"] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=sorted(CONFIG))
    ap.add_argument("--size", required=True, type=int)
    ap.add_argument("--repo-root", default=os.getcwd())
    ap.add_argument("--label", default="run")
    # tsp_gls only: the paper trains at (30, 1200) and we test at a larger budget
    ap.add_argument("--moves", type=int)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = CONFIG[args.problem]
    problem_dir = os.path.join(args.repo_root, "problems", args.problem)
    dataset = os.path.join(problem_dir, "datasets", cfg["pattern"].format(size=args.size))

    if not os.path.exists(dataset):
        raise SystemExit(f"[rep_eval] dataset not found: {dataset}")

    kwargs = dict(cfg["kwargs"])
    if args.moves is not None and "perturbation_moves" in kwargs:
        kwargs["perturbation_moves"] = args.moves
    if args.iters is not None and "iter_limit" in kwargs:
        kwargs["iter_limit"] = args.iters

    os.chdir(problem_dir)
    sys.path.insert(0, problem_dir)
    module = load_eval_module(problem_dir)

    t0 = time.time()
    values = np.asarray(module.process_file(dataset, **kwargs), dtype=float)
    elapsed = time.time() - t0

    objective = float(values.mean())
    record = {
        "problem": args.problem,
        "size": args.size,
        "label": args.label,
        "objective": objective,
        "better": "higher" if cfg["maximize"] else "lower",
        "n_instances": int(values.size),
        "seconds": round(elapsed, 1),
        "params": kwargs,
    }

    print(json.dumps(record))
    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(args.repo_root, args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
