"""Where do the DR A/B policies fail under 3-step lag: before the grasp or after it?

Each of the six A/B policies, 100 `laggy` lift episodes (eval seed 0): share of episodes that succeed, that end
with the cube grasped but not lifted, and that end without a grasp.

    python tools/lag_failmode.py      # writes runs/lag_failmode.json, about 20 s on one core
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")
from so_arm100_sim.env import ArmEnv, TaskSpec   # noqa: E402
from so_arm100_sim.evaluate import run_episode   # noqa: E402
from so_arm100_rl.ppo import FrozenPolicy        # noqa: E402

EPISODES = 100
out = {}
for arm in ("full", "centred"):
    for seed in (0, 1, 2):
        policy = FrozenPolicy.from_checkpoint(os.path.join(ROOT, f"runs/ab_{arm}_s{seed}/policy.pt"))
        env = ArmEnv(TaskSpec.parse("lift:red"), seed=0, physics="laggy", obs_mode="state", action_mode="delta")
        eps = [run_episode(env, policy, j) for j in range(EPISODES)]
        env.close()
        ok = sum(e["success"] for e in eps)
        held = sum(e["grasped"] and not e["success"] for e in eps)
        out[f"{arm}_s{seed}"] = {"success": ok / EPISODES, "grasped_not_lifted": held / EPISODES,
                                 "no_grasp": (EPISODES - ok - held) / EPISODES}
        print(arm, seed, out[f"{arm}_s{seed}"], flush=True)
with open(os.path.join(ROOT, "runs/lag_failmode.json"), "w") as fh:
    json.dump({"episodes": EPISODES, "physics": "laggy", "policies": out}, fh, indent=1)
