"""Collect the per-size JSON records into one markdown table.

Two modes:
  --mode gap          compare against hard-coded LKH reference values (tsp_gls)
  --mode improvement  compare the evolved run against the baseline run
"""
import argparse
import glob
import json
import os

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


def emit(lines):
    text = "\n".join(lines)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gap", "improvement"])
    ap.add_argument("--glob", default="collected/**/*.json")
    args = ap.parse_args()

    records = load(args.glob)
    if not records:
        emit([f"No result files matched `{args.glob}`."])
        raise SystemExit(1)

    emit(mode_gap(records) if args.mode == "gap" else mode_improvement(records))


if __name__ == "__main__":
    main()
