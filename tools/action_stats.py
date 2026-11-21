"""Do the DR A/B policies differ in how hard they drive the arm? (candidate mechanism for B's lag tolerance)

For each of the six A/B policies, on `nominal` and on `laggy`: 50 lift episodes (eval seed 0, same episode seeds for every policy),
mean |action| per step, mean |a_t - a_{t-1}| per step (jerkiness), and the share of action components at the
+-1 clip. Actions are joint-target deltas (action_mode delta), so |action| is the commanded speed.

    python tools/action_stats.py      # writes runs/action_stats.json, about 30 s on one core
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")
from so_arm100_sim.env import ArmEnv, TaskSpec   # noqa: E402
from so_arm100_rl.ppo import FrozenPolicy        # noqa: E402

EPISODES = 50
out = {}
for phys in ("nominal", "laggy"):
    for arm in ("full", "centred"):
        for seed in (0, 1, 2):
            policy = FrozenPolicy.from_checkpoint(os.path.join(ROOT, f"runs/ab_{arm}_s{seed}/policy.pt"))
            env = ArmEnv(TaskSpec.parse("lift:red"), seed=0, physics=phys, obs_mode="state", action_mode="delta")
            mag, jerk, sat = [], [], []
            for j in range(EPISODES):
                obs, prev, done = env.reset(seed=j), None, False
                while not done:
                    a = policy.act(obs)
                    mag.append(np.abs(a).mean())
                    sat.append((np.abs(a) >= 0.999).mean())
                    if prev is not None:
                        jerk.append(np.abs(a - prev).mean())
                    prev = a
                    obs, _, done, _ = env.step(a)
            env.close()
            out[f"{phys}/{arm}_s{seed}"] = {"mean_abs_action": float(np.mean(mag)), "mean_abs_delta": float(np.mean(jerk)),
                                                 "saturated_share": float(np.mean(sat)), "steps": len(mag)}
            print(phys, arm, seed, {k: round(v, 4) for k, v in out[f"{phys}/{arm}_s{seed}"].items()}, flush=True)
with open(os.path.join(ROOT, "runs/action_stats.json"), "w") as fh:
    json.dump({"episodes": EPISODES, "policies": out}, fh, indent=1)
