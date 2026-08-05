"""Build the single reproduction report: the results table plus the discovered
heuristics inlined as fenced code blocks.

Two modes:
  --mode gap          compare against hard-coded LKH reference values (tsp_gls)
  --mode improvement  compare the evolved run against the baseline run
"""
import argparse
import glob
import json
import os
import re

# LKH-3 (via elkai) on problems/tsp_gls/datasets/test_TSP{size}.npy, 64 instances each.
# TSP50 used runs=200 and matches the best tour GLS itself finds, so it is optimal;
# TSP100 runs=50; TSP200 and TSP500 runs=10, i.e. strong references but not proven optima,
# so a small negative gap at those sizes is possible.
LKH_TSP_GLS = {
    50: 5.684580,
    100: 7.778580,
    200: 10.711932,
    500: 16.499827,
}


def load(pattern):
    out = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def emit(lines, out=None):
    text = "\n".join(lines)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


def mode_gap(records):
    lines = ["## rep_gls — optimality gap vs LKH-3", "",
             "| size | LKH-3 | MOTIF | gap | instances | seconds |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(records, key=lambda x: x["size"]):
        opt = LKH_TSP_GLS.get(r["size"])
        if opt is None:
            lines.append(f"| {r['size']} | n/a | {r['objective']:.6f} | n/a | "
                         f"{r['n_instances']} | {r['seconds']} |")
            continue
        gap = (r["objective"] - opt) / opt * 100
        lines.append(f"| {r['size']} | {opt:.6f} | {r['objective']:.6f} | {gap:+.3f}% | "
                     f"{r['n_instances']} | {r['seconds']} |")
    params = records[0]["params"] if records else {}
    lines += ["", f"Test solver budget: `{params}`.",
              "LKH-3 references are hard-coded in `.github/scripts/rep_report.py`."]
    return lines


def mode_improvement(records):
    pairs = {}
    for r in records:
        pairs.setdefault((r["problem"], r["size"]), {})[r["label"]] = r

    lines = ["## Improvement over baseline", "",
             "| problem | size | baseline | MOTIF | improvement | better | seconds |",
             "|---|---:|---:|---:|---:|:--:|---:|"]
    for (problem, size) in sorted(pairs):
        got = pairs[(problem, size)]
        base, motif = got.get("baseline"), got.get("motif")
        if not base or not motif:
            missing = "baseline" if not base else "motif"
            lines.append(f"| {problem} | {size} | — | — | missing `{missing}` run | | |")
            continue
        b, m = base["objective"], motif["objective"]
        # improvement is always "how much better MOTIF is", in the natural direction
        impr = (m - b) / abs(b) * 100 if motif["better"] == "higher" else (b - m) / abs(b) * 100
        lines.append(f"| {problem} | {size} | {b:.6f} | {m:.6f} | {impr:+.2f}% | "
                     f"{motif['better']} | {motif['seconds']} |")
    return lines


def heuristic_sections(root):
    """Inline every discovered F*.py as a fenced code block, grouped by problem.

    Artifacts land either flat (single-problem workflow) or one directory per
    downloaded artifact named `<workflow>-heuristic-<problem>`.
    """
    files = sorted(glob.glob(os.path.join(root, "**", "F*_final_best.py"), recursive=True))
    if not files:
        return ["", "## Discovered heuristics", "", "_none found_"]

    by_problem = {}
    for path in files:
        parent = os.path.basename(os.path.dirname(path))
        m = re.search(r"heuristic-(.+)$", parent)
        problem = m.group(1) if m else ""
        by_problem.setdefault(problem, []).append(path)

    lines = ["", "## Discovered heuristics"]
    for problem in sorted(by_problem):
        if problem:
            lines += ["", f"### {problem}"]
        for path in sorted(by_problem[problem]):
            strategy = os.path.basename(path).split("_")[0]
            with open(path, encoding="utf-8") as f:
                # drop the provenance header MOTIF writes, it is repeated in the table
                code = "\n".join(
                    ln for ln in f.read().splitlines()
                    if not ln.startswith(("# Final Round optimized", "# Strategy ID:", "# Phase:"))
                ).strip()
            lines += ["", f"**{strategy}**", "", "```python", code, "```"]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gap", "improvement"])
    ap.add_argument("--heuristics", help="directory holding the F*_final_best.py artifacts")
    ap.add_argument("--glob", default="collected/**/*.json")
    ap.add_argument("--out", help="also write the markdown here, for archiving")
    ap.add_argument("--title", default="", help="line prepended to the report")
    args = ap.parse_args()

    records = load(args.glob)
    missing = not records
    lines = ([f"No result files matched `{args.glob}`."] if missing
             else mode_gap(records) if args.mode == "gap"
             else mode_improvement(records))

    if args.heuristics:
        lines += heuristic_sections(args.heuristics)
    if args.title:
        lines = [args.title, ""] + lines

    emit(lines, args.out)
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
