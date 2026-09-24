"""Evaluate a trained policy under the bench protocol on every held-out
physics cell, plus DR sampling. This is the number that matters.

    python eval_gap.py --ckpt runs/lift_scr_nominal/policy.pt --task lift --tag nominal --jobs 8

100 episodes x 5 seeds per cell by default (~40 min single core for lift);
`--jobs 8` runs the cells in parallel processes with identical results;
`--episodes 20 --seeds 1` for a quick look. Writes runs/gap_<tag>.json and
prints the markdown table.
"""
import argparse
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from so_arm100_sim.dr import DRConfig, EVAL_PHYSICS                   # noqa: E402
from so_arm100_sim.evaluate import evaluate, format_gap_table, save_json, provenance   # noqa: E402

from so_arm100_rl.ppo import FrozenPolicy                             # noqa: E402


def run_cell(ckpt, task, color, phys, episodes, seeds):
    """One physics cell. Top-level so a worker process can run it; every cell
    builds its own env and policy from the seed, so the result does not depend
    on which process ran it or in what order."""
    policy = FrozenPolicy.from_checkpoint(ckpt)
    factory = lambda env: policy      # noqa: E731 -- the policy ignores the env
    kw = dict(n_episodes=episodes, seeds=tuple(range(seeds)),
              env_kwargs={"obs_mode": "state", "action_mode": "delta"})
    if phys == "dr":
        return evaluate(factory, f"{task}:{color}", dr=DRConfig(), **kw)
    return evaluate(factory, f"{task}:{color}", physics=phys, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="lift")
    ap.add_argument("--color", default="red")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--physics", nargs="*", default=list(EVAL_PHYSICS) + ["dr"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--jobs", type=int, default=1, help="cells evaluated in parallel processes")
    a = ap.parse_args()

    args = [(a.ckpt, a.task, a.color, phys, a.episodes, a.seeds) for phys in a.physics]
    if a.jobs > 1:
        with mp.get_context("spawn").Pool(min(a.jobs, len(args))) as pool:
            rows = pool.starmap(run_cell, args)
    else:
        rows = [run_cell(*x) for x in args]
    results = {}
    for phys, r in zip(a.physics, rows):
        results[phys] = r
        print(f"  {phys:9s} success {r['success_rate']:.3f} +- {r['success_std']:.3f}"
              f"  ({r['wall_s']:.0f} s)", flush=True)
    print("\n" + format_gap_table(results))
    out = {"ckpt": a.ckpt, "task": a.task, "episodes": a.episodes, "seeds": a.seeds,
           "cells": results, **provenance()}
    path = save_json(f"runs/gap_{a.tag}.json", out)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
