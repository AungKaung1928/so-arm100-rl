"""DR A/B table from the runs tools/dr_ab.sh leaves behind.

Arm A = --dr full, arm B = --dr centred (DR centred on the real-fitted servo) by default, or any other
--dr arm via --b (e.g. --b both). For every training seed that has both runs/gap_ab_full_s<seed>.json and
runs/gap_ab_<b>_s<seed>.json: success per physics cell, the
paired difference B - A, and the sim-to-real gap of each arm (nominal - real_fit success). Spread = standard
deviation over training seeds; with fewer than 3 seeds read the per-seed rows, not the mean.

    python dr_ab.py            # prints the table, writes runs/dr_ab.json
    python dr_ab.py --b both   # arm C vs A, writes runs/dr_ab_both.json
    python dr_ab.py --b both --prefix gap_ab gap_abx   # arm A's weak/heavy/... cells come from gap_abx
"""
import argparse
import glob
import json
import re
import statistics


def load(runs="runs", b="centred", prefixes=("gap_ab",)):
    """{seed: {arm: eval_gap result}} for seeds where both arm A (full) and arm b exist.

    Each prefix is one eval file per arm and seed (e.g. gap_ab, gap_abx); their cells are merged, and
    files merged for one arm and seed must come from the same checkpoint."""
    by_seed = {}
    for prefix in prefixes:
        for arm in ("full", b):
            for path in glob.glob(f"{runs}/{prefix}_{arm}_s*.json"):
                seed = int(re.search(r"_s(\d+)\.json$", path).group(1))
                with open(path) as fh:
                    r = json.load(fh)
                have = by_seed.setdefault(seed, {}).get(arm)
                if have is None:
                    by_seed[seed][arm] = r
                    continue
                if have.get("ckpt") != r.get("ckpt"):
                    raise ValueError(f"{path}: ckpt {r.get('ckpt')} != {have.get('ckpt')}")
                have["cells"] = {**have["cells"], **r["cells"]}
    return {s: r for s, r in sorted(by_seed.items()) if len(r) == 2}


def mean_sd(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def summarise(by_seed, b="centred"):
    seeds = list(by_seed)
    cells = [c for c in by_seed[seeds[0]]["full"]["cells"] if all(c in r[a]["cells"] for r in by_seed.values() for a in r)]
    rate = {a: {c: [by_seed[s][a]["cells"][c]["success_rate"] for s in seeds] for c in cells} for a in ("full", b)}
    every = {c for r in by_seed.values() for a in r for c in r[a]["cells"]}
    out = {"train_seeds": seeds, "b": b, "cells": {}, "dropped_cells": sorted(every - set(cells))}
    for c in cells:
        diff = [y - x for x, y in zip(rate["full"][c], rate[b][c])]
        out["cells"][c] = {"full": rate["full"][c], b: rate[b][c], "diff": diff,
                           "full_mean_sd": mean_sd(rate["full"][c]), f"{b}_mean_sd": mean_sd(rate[b][c]),
                           "diff_mean_sd": mean_sd(diff)}
    if "nominal" in cells and "real_fit" in cells:
        gap = {a: [n - r for n, r in zip(rate[a]["nominal"], rate[a]["real_fit"])] for a in rate}
        out["gap_nominal_minus_real_fit"] = {a: {"per_seed": g, "mean_sd": mean_sd(g)} for a, g in gap.items()}
    return out


def table(s):
    b = s.get("b", "centred")
    lines = [f"| physics | A: --dr full | B: --dr {b} | B - A (paired) |", "|---|---|---|---|"]
    for c, r in s["cells"].items():
        lines.append("| {} | {:.3f} +- {:.3f} | {:.3f} +- {:.3f} | {:+.3f} +- {:.3f} |".format(
            c, *r["full_mean_sd"], *r[f"{b}_mean_sd"], *r["diff_mean_sd"]))
    g = s.get("gap_nominal_minus_real_fit")
    if g:
        lines.append("| **gap: nominal - real_fit** | {:.3f} +- {:.3f} | {:.3f} +- {:.3f} | |".format(
            *g["full"]["mean_sd"], *g[b]["mean_sd"]))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--b", default="centred", help="arm compared with --dr full")
    ap.add_argument("--prefix", nargs="+", default=["gap_ab"], help="eval files merged per arm and seed")
    a = ap.parse_args()
    by_seed = load(a.runs, a.b, a.prefix)
    if not by_seed:
        raise SystemExit(f"no seed with both arms in {a.runs}/ (run tools/dr_ab.sh first)")
    s = summarise(by_seed, a.b)
    print(f"training seeds {s['train_seeds']}; mean +- sd over training seeds\n")
    print(table(s))
    if s["dropped_cells"]:
        print(f"\nnot in every arm and seed, left out: {', '.join(s['dropped_cells'])}")
    name = "dr_ab.json" if a.b == "centred" else f"dr_ab_{a.b}.json"
    with open(f"{a.runs}/{name}", "w") as fh:
        json.dump(s, fh, indent=1)
