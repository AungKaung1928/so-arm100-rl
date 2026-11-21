"""DR A/B table from the runs tools/dr_ab.sh leaves behind.

Arm A = --dr full, arm B = --dr centred (DR centred on the real-fitted servo). For every training seed that
has both runs/gap_ab_full_s<seed>.json and runs/gap_ab_centred_s<seed>.json: success per physics cell, the
paired difference B - A, and the sim-to-real gap of each arm (nominal - real_fit success). Spread = standard
deviation over training seeds; with fewer than 3 seeds read the per-seed rows, not the mean.

    python dr_ab.py            # prints the table, writes runs/dr_ab.json
"""
import glob
import json
import re
import statistics
import sys


def load(runs="runs"):
    """{seed: {arm: eval_gap result}} for seeds where both arms exist."""
    by_seed = {}
    for path in glob.glob(f"{runs}/gap_ab_*_s*.json"):
        arm, seed = re.search(r"gap_ab_(full|centred)_s(\d+)\.json$", path).groups()
        with open(path) as fh:
            by_seed.setdefault(int(seed), {})[arm] = json.load(fh)
    return {s: r for s, r in sorted(by_seed.items()) if len(r) == 2}


def mean_sd(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def summarise(by_seed):
    seeds = list(by_seed)
    cells = [c for c in by_seed[seeds[0]]["full"]["cells"] if all(c in r[a]["cells"] for r in by_seed.values() for a in r)]
    rate = {a: {c: [by_seed[s][a]["cells"][c]["success_rate"] for s in seeds] for c in cells} for a in ("full", "centred")}
    out = {"train_seeds": seeds, "cells": {}}
    for c in cells:
        diff = [b - a for a, b in zip(rate["full"][c], rate["centred"][c])]
        out["cells"][c] = {"full": rate["full"][c], "centred": rate["centred"][c], "diff": diff,
                           "full_mean_sd": mean_sd(rate["full"][c]), "centred_mean_sd": mean_sd(rate["centred"][c]),
                           "diff_mean_sd": mean_sd(diff)}
    if "nominal" in cells and "real_fit" in cells:
        gap = {a: [n - r for n, r in zip(rate[a]["nominal"], rate[a]["real_fit"])] for a in rate}
        out["gap_nominal_minus_real_fit"] = {a: {"per_seed": g, "mean_sd": mean_sd(g)} for a, g in gap.items()}
    return out


def table(s):
    lines = ["| physics | A: --dr full | B: --dr centred | B - A (paired) |", "|---|---|---|---|"]
    for c, r in s["cells"].items():
        lines.append("| {} | {:.3f} +- {:.3f} | {:.3f} +- {:.3f} | {:+.3f} +- {:.3f} |".format(
            c, *r["full_mean_sd"], *r["centred_mean_sd"], *r["diff_mean_sd"]))
    g = s.get("gap_nominal_minus_real_fit")
    if g:
        lines.append("| **gap: nominal - real_fit** | {:.3f} +- {:.3f} | {:.3f} +- {:.3f} | |".format(
            *g["full"]["mean_sd"], *g["centred"]["mean_sd"]))
    return "\n".join(lines)


if __name__ == "__main__":
    runs = sys.argv[1] if len(sys.argv) > 1 else "runs"
    by_seed = load(runs)
    if not by_seed:
        raise SystemExit(f"no seed with both arms in {runs}/ (run tools/dr_ab.sh first)")
    s = summarise(by_seed)
    print(f"training seeds {s['train_seeds']}; mean +- sd over training seeds\n")
    print(table(s))
    with open(f"{runs}/dr_ab.json", "w") as fh:
        json.dump(s, fh, indent=1)
