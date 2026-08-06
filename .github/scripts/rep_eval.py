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

# `train` holds the settings MOTIF searches under, taken from each problem's eval.py.
# Test keeps those settings unchanged and only scales the iteration count, by
# --iter-multiplier (default 2). `scale` names the key to multiply; None means the
# solver has no iteration parameter at all.
#
# tsp_aco is the one problem whose eval.py already encodes this (100 -> 200, ant count
# unchanged), so the default multiplier reproduces it exactly. The other ACO problems
# reuse their train constants for test in eval.py, and are scaled here instead.
#
# The DR solvers take destruction_rate / use_2opt / max_workers / seed and nothing else:
# each run does one deconstruct-repair pass per starting item, and the number of starting
# items is fixed by the instance size. There is no iteration count to scale.
CONFIG = {
    "tsp_aco":  dict(pattern="test_TSP{size}.npy",  maximize=False, scale="n_iter",
                     train=dict(n_ants=30, n_iter=100)),
    "cvrp_aco": dict(pattern="test_CVRP{size}.npy", maximize=False, scale="n_iter",
                     train=dict(n_ants=30, n_iter=100)),
    "op_aco":   dict(pattern="test_OP{size}.npz",   maximize=True,  scale="n_iter",
                     train=dict(n_ants=20, n_iter=100)),
    "mkp_aco":  dict(pattern="test_MKP{size}.npz",  maximize=True,  scale="n_iter",
                     train=dict(n_ants=10, n_iter=50)),
    "bpp_aco":  dict(pattern="test_BPP{size}.npz",  maximize=False, scale="n_iter",
                     train=dict(n_ants=20, n_iter=50)),
    "tsp_gls":  dict(pattern="test_TSP{size}.npy",  maximize=False, scale="iter_limit",
                     train=dict(perturbation_moves=30, iter_limit=1200)),
    "tsp_dr":   dict(pattern="test_TSP{size}.npy",  maximize=False, scale=None, train={}),
    "cvrp_dr":  dict(pattern="test_CVRP{size}.npy", maximize=False, scale=None, train={}),
    "bpp_dr":   dict(pattern="test_BPP{size}.npz",  maximize=False, scale=None,
                     train=dict(max_workers=20, destruction_rate=0.3)),
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
    ap.add_argument("--iter-multiplier", type=float, default=2.0,
                    help="test iterations = train iterations x this (ignored when the "
                         "solver has no iteration parameter)")
    # explicit overrides, used by rep_gls where the test budget is stated outright
    ap.add_argument("--moves", type=int)
    ap.add_argument("--iters", type=int)
    ap.add_argument("--out")
    args = ap.parse_args()

    cfg = CONFIG[args.problem]
    problem_dir = os.path.join(args.repo_root, "problems", args.problem)
    dataset = os.path.join(problem_dir, "datasets", cfg["pattern"].format(size=args.size))

    if not os.path.exists(dataset):
        raise SystemExit(f"[rep_eval] dataset not found: {dataset}")

    # same settings as training, iterations scaled
    kwargs = dict(cfg["train"])
    scale_key = cfg["scale"]
    if scale_key:
        kwargs[scale_key] = int(round(kwargs[scale_key] * args.iter_multiplier))

    if args.moves is not None and "perturbation_moves" in kwargs:
        kwargs["perturbation_moves"] = args.moves
    if args.iters is not None and scale_key:
        kwargs[scale_key] = args.iters

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
        "train_params": cfg["train"],
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
