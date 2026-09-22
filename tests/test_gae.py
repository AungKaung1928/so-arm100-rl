"""GAE by hand, with one termination and one truncation in four steps."""
import numpy as np
import torch

from so_arm100_rl.ppo import compute_gae


def test_gae_hand_computed_with_termination_and_truncation():
    gamma, lam = 0.9, 0.8
    # One env, four steps. Episode A ends by TERMINATION after step 1 (no
    # bootstrap). Episode B ends by TRUNCATION after step 2: the caller has
    # already added gamma * V(terminal) = 0.9 * 2.0 = 1.8 to r[2]. Step 3
    # belongs to episode C, still running, bootstrapped with last_value.
    r = torch.tensor([[1.0], [2.0], [0.5 + 1.8], [1.0]])
    v = torch.tensor([[0.5], [1.0], [0.7], [0.3]])
    d = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    last_value = torch.tensor([0.4])
    adv, ret = compute_gae(r, v, d, last_value, gamma, lam)

    # backward by hand
    # t=3: delta = 1.0 + 0.9*0.4*1 - 0.3 = 1.06 ; A3 = 1.06
    # t=2: done -> nonterminal 0: delta = 2.3 - 0.7 = 1.6 ; A2 = 1.6 (no flow from A3)
    # t=1: done -> delta = 2.0 - 1.0 = 1.0 ; A1 = 1.0
    # t=0: delta = 1.0 + 0.9*1.0 - 0.5 = 1.4 ; A0 = 1.4 + 0.9*0.8*A1 = 1.4 + 0.72 = 2.12
    expect = np.array([2.12, 1.0, 1.6, 1.06])
    assert np.allclose(adv.squeeze(1).numpy(), expect, atol=1e-6), adv.squeeze(1)
    assert np.allclose(ret.squeeze(1).numpy(), expect + v.squeeze(1).numpy(), atol=1e-6)


def test_gae_without_boundaries_matches_discounted_sum_when_lambda_is_one():
    gamma, lam = 0.5, 1.0
    r = torch.tensor([[1.0], [1.0], [1.0]])
    v = torch.zeros(3, 1)
    d = torch.zeros(3, 1)
    adv, ret = compute_gae(r, v, d, torch.tensor([0.0]), gamma, lam)
    assert np.allclose(ret.squeeze(1).numpy(), [1.75, 1.5, 1.0])


def test_gae_vectorised_over_envs_is_independent_per_env():
    gamma, lam = 0.9, 0.8
    r = torch.tensor([[1.0, 5.0], [2.0, 5.0], [2.3, 5.0], [1.0, 5.0]])
    v = torch.tensor([[0.5, 0.0], [1.0, 0.0], [0.7, 0.0], [0.3, 0.0]])
    d = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    adv, _ = compute_gae(r, v, d, torch.tensor([0.4, 0.0]), gamma, lam)
    single, _ = compute_gae(r[:, :1], v[:, :1], d[:, :1], torch.tensor([0.4]), gamma, lam)
    assert torch.allclose(adv[:, 0], single[:, 0])
    assert adv[0, 1] > adv[3, 1] > 0        # constant reward, no boundaries: monotone
