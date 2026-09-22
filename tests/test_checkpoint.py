"""A checkpoint restores everything the next update depends on: weights,
optimizer moments, normaliser, RNG. Two more updates after a reload must
equal two more updates without one, bit for bit."""
import os

import numpy as np
import torch

from so_arm100_rl.ppo import PPO, Config, compute_gae


def synthetic_update(ppo, rng, T=8, N=2, O=25, A=6):
    obs = rng.normal(size=(T * N, O)).astype(np.float32)
    x, action, logp, value = ppo.policy_step(obs)
    b_obs, b_act = x.view(T, N, O), action.view(T, N, A)
    b_logp, b_val = logp.view(T, N), value.view(T, N)
    b_rew = torch.as_tensor(rng.normal(size=(T, N)).astype(np.float32))
    b_done = torch.zeros(T, N)
    adv, ret = compute_gae(b_rew, b_val, b_done, torch.zeros(N), 0.99, 0.95)
    return ppo.update(b_obs, b_act, b_logp, adv, ret, b_val)


def params(ppo):
    return [p.detach().clone() for p in ppo.agent.parameters()]


def test_resume_is_bit_exact(tmp_path):
    cfg = Config(hidden=16, num_envs=2, num_steps=8, update_epochs=2, num_minibatches=2, seed=3)
    torch.manual_seed(3)
    np.random.seed(3)
    rng = np.random.default_rng(3)
    ppo = PPO(cfg, 25, 6)
    for _ in range(2):
        synthetic_update(ppo, rng, N=2)
    path = os.path.join(tmp_path, "ckpt.pt")
    ppo.save(path, {"curriculum": {"note": "test"}})
    rng_state = rng.bit_generator.state

    # continue without reload
    for _ in range(2):
        synthetic_update(ppo, rng, N=2)
    straight = params(ppo)
    straight_norm = ppo.norm.state_dict()

    # reload and repeat the same two updates
    rng2 = np.random.default_rng(0)
    rng2.bit_generator.state = rng_state
    ppo2, ck = PPO.load(path)
    assert ck["curriculum"] == {"note": "test"} and ppo2.update_i == 2
    for _ in range(2):
        synthetic_update(ppo2, rng2, N=2)
    for a, b in zip(straight, params(ppo2)):
        assert torch.equal(a, b)
    assert np.array_equal(straight_norm["mean"], ppo2.norm.state_dict()["mean"])
    assert ppo2.update_i == 4


def test_saturation_is_tracked():
    cfg = Config(hidden=8, init_log_std=2.0)
    ppo = PPO(cfg, 25, 6)
    ppo.policy_step(np.zeros((64, 25), np.float32))
    assert 0.5 < ppo.saturated / ppo.total_actions <= 1.0      # sigma e^2: most samples clip
