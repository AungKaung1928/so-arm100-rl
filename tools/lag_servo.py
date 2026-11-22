"""Is the DR A/B 'laggy' lead an eval-time servo effect or a trained-policy effect?

Evaluates the eight A/B policies (runs/ab_{full,centred}_s{0,1,2,3}) on one extra cell: 3-step latency (as
`laggy`) with the servo at the centre of --dr centred (kp 0.71x, frictionloss 2.5x). If arm A recovers there,
a slow servo absorbs the lag at evaluation time; if not, the lead comes from how B was trained.

    python tools/lag_servo.py      # writes runs/gap_lagsoft_{arm}_s{seed}.json, about 3 min on one core
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")
from so_arm100_sim.evaluate import evaluate, save_json   # noqa: E402
from so_arm100_rl.ppo import FrozenPolicy               # noqa: E402

CELL = {"latency": 3, "kp": 0.71, "frictionloss": 2.5}

for seed in (0, 1, 2, 3):
    for arm in ("full", "centred"):
        out = os.path.join(ROOT, f"runs/gap_lagsoft_{arm}_s{seed}.json")
        if os.path.exists(out):
            continue
        policy = FrozenPolicy.from_checkpoint(os.path.join(ROOT, f"runs/ab_{arm}_s{seed}/policy.pt"))
        r = evaluate(lambda env: policy, "lift:red", physics=CELL, n_episodes=100, seeds=tuple(range(5)),
                     env_kwargs={"obs_mode": "state", "action_mode": "delta"})
        print(arm, seed, round(r["success_rate"], 3), flush=True)
        save_json(out, {"cells": {"lagsoft": r}, "cell": CELL})
