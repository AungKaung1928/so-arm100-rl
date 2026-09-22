import os

import numpy as np
import torch

import export_onnx as X
from so_arm100_rl.ppo import PPO, Config, FrozenPolicy


def test_export_round_trip_and_latency(tmp_path):
    torch.manual_seed(0)
    ppo = PPO(Config(hidden=32), 25, 6)
    ppo.norm.update(np.random.default_rng(0).normal(3.0, 2.0, (200, 25)))
    pol = FrozenPolicy(ppo.agent, ppo.norm)
    path = os.path.join(tmp_path, "p.onnx")
    m, nbytes = X.export(pol.agent, pol.norm, path, 25)
    assert nbytes > 1000
    diff, session = X.verify(m, path, 25, n=200)
    assert diff < 1e-5
    lat = X.latency(session, 25, n=100)
    assert lat["p50_ms"] > 0 and lat["p99_ms"] >= lat["p50_ms"]
    # the graph output equals the frozen policy's action on a raw observation
    obs = {"state": np.random.default_rng(1).normal(3.0, 2.0, 25).astype(np.float32)}
    a = session.run(None, {"obs": obs["state"][None]})[0][0]
    assert np.allclose(a, pol.act(obs), atol=1e-6)
