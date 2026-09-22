"""Evaluate a trained policy under the bench protocol on every held-out
physics cell, plus DR sampling. This is the number that matters.

    python eval_gap.py --ckpt runs/lift_cur_nominal/policy.pt --task lift --tag nominal

100 episodes x 5 seeds per cell by default (~40 min single core for lift);
`--episodes 20 --seeds 1` for a quick look. Writes runs/gap_<tag>.json and
prints the markdown table.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from so_arm100_sim.dr import DRConfig, EVAL_PHYSICS                   # noqa: E402
from so_arm100_sim.evaluate import evaluate, format_gap_table, save_json, provenance   # noqa: E402

from so_arm100_rl.ppo import FrozenPolicy                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="lift")
    ap.add_argument("--color", default="red")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--physics", nargs="*", default=list(EVAL_PHYSICS) + ["dr"])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    policy = FrozenPolicy.from_checkpoint(a.ckpt)
    factory = lambda env: policy      # noqa: E731 -- the policy ignores the env
    results = {}
    for phys in a.physics:
        kw = dict(n_episodes=a.episodes, seeds=tuple(range(a.seeds)),
                  env_kwargs={"obs_mode": "state", "action_mode": "delta"})
        if phys == "dr":
            r = evaluate(factory, f"{a.task}:{a.color}", dr=DRConfig(), **kw)
        else:
            r = evaluate(factory, f"{a.task}:{a.color}", physics=phys, **kw)
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
