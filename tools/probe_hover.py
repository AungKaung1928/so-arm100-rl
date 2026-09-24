"""Probe a lift policy for the hover exploit (README, "Three training failures", 2).

    python tools/probe_hover.py runs/<tag>/policy.pt

20 episodes of 150 steps with success termination off, first with the mean
action, then with Gaussian noise (sigma 0.5) on it. Prints the median number
of steps the cube was grasped, the peak cube height per episode, the median
height while grasped, and how many episodes ever crossed the success line.
"""
import sys, numpy as np, torch
torch.set_num_threads(1)
from so_arm100_rl.ppo import FrozenPolicy
from so_arm100_sim.env import ArmEnv, LIFT_HEIGHT
pol = FrozenPolicy.from_checkpoint(sys.argv[1])
for noise in (False, True):
  rows=[]
  for s in range(20):
    env = ArmEnv("lift", obs_mode="state", action_mode="delta", seed=1000+s, terminate_on_success=False)
    o = env.reset()
    hs=[]; g=0; succ=0
    for t in range(150):
        a = pol.act(o)
        if noise: a = np.clip(a + np.random.randn(6)*0.5, -1, 1)
        o, r, d, info = env.step(a)
        h = info["cube_pos"][2] - env.physics.cube_half
        if info["grasped"]: g+=1; hs.append(h)
        succ += info["success"]
    rows.append((g, max(hs) if hs else 0, np.median(hs) if hs else 0, succ))
    env.close()
  r=np.array(rows)
  print("noise" if noise else "mean", "grasped steps med", np.median(r[:,0]), "| max h (cm) per ep", np.round(r[:,1]*100,1), "| median grasped h cm", np.round(np.median(r[:,2])*100,2), "| eps with any success", int((r[:,3]>0).sum()), "/20; line =", LIFT_HEIGHT*100)
