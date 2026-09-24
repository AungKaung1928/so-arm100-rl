import multiprocessing as mp
import os

import torch

import eval_gap as E
from so_arm100_rl.ppo import PPO, Config


def test_parallel_cells_equal_serial(tmp_path):
    """--jobs only changes which process runs a cell, never its numbers."""
    torch.manual_seed(0)
    ckpt = os.path.join(tmp_path, "p.pt")
    PPO(Config(hidden=32), 25, 6).save(ckpt)
    args = [(ckpt, "lift", "red", phys, 2, 1) for phys in ("nominal", "heavy")]
    serial = [E.run_cell(*x) for x in args]
    with mp.get_context("spawn").Pool(2) as pool:
        parallel = pool.starmap(E.run_cell, args)
    for s, p in zip(serial, parallel):
        for k in ("success_rate", "per_seed", "return_mean", "steps_mean"):
            assert s[k] == p[k], k
