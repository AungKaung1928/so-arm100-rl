"""Where do the DR A/B policies fail under 3-step lag: before any grasp, or after grasping and losing the cube?

Each of the six A/B policies, 100 `laggy` lift episodes (eval seed 0). Per episode, from the env's per-step
`grasped` flag: success; failed after the cube was grasped at some step; failed without any grasp. Also counts
grasp onsets and losses (flag on -> off) over all episodes, and the share of losses with the cube on the table
(less than 1 cm above its start height).

    python tools/lag_failmode.py      # writes runs/lag_failmode.json, about 20 s on one core
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")
from so_arm100_sim.env import ArmEnv, TaskSpec   # noqa: E402
from so_arm100_rl.ppo import FrozenPolicy        # noqa: E402

EPISODES = 100
out = {}
for arm in ("full", "centred"):
    for seed in (0, 1, 2):
        policy = FrozenPolicy.from_checkpoint(os.path.join(ROOT, f"runs/ab_{arm}_s{seed}/policy.pt"))
        env = ArmEnv(TaskSpec.parse("lift:red"), seed=0, physics="laggy", obs_mode="state", action_mode="delta")
        n = {"success": 0, "failed_after_grasp": 0, "failed_no_grasp": 0}
        onsets = losses = losses_on_table = 0
        for j in range(EPISODES):
            obs, done, grasped_ever, prev, z0 = env.reset(seed=j), False, False, False, None
            while not done:
                obs, _, done, info = env.step(policy.act(obs))
                g, z = bool(info["grasped"]), float(info["cube_pos"][2])
                z0 = z if z0 is None else z0
                onsets += g and not prev
                if prev and not g:
                    losses += 1
                    losses_on_table += z - z0 < 0.01
                prev = g
                grasped_ever |= g
            key = "success" if info["success_ever"] else "failed_after_grasp" if grasped_ever else "failed_no_grasp"
            n[key] += 1
        env.close()
        out[f"{arm}_s{seed}"] = {**{k: v / EPISODES for k, v in n.items()}, "grasp_onsets": int(onsets),
                                 "grasp_losses": int(losses), "losses_on_table": losses_on_table / max(losses, 1)}
        print(arm, seed, out[f"{arm}_s{seed}"], flush=True)
with open(os.path.join(ROOT, "runs/lag_failmode.json"), "w") as fh:
    json.dump({"episodes": EPISODES, "physics": "laggy", "policies": out}, fh, indent=1)
