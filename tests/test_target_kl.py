"""target_kl stops the epoch loop early; 0 leaves it at update_epochs."""
import torch

from so_arm100_rl.ppo import PPO, Config


def batch(ppo, t=16, n=2):
    torch.manual_seed(0)
    obs = torch.randn(t, n, 25)
    with torch.no_grad():
        act, logp, _, val = ppo.agent.act(obs.reshape(-1, 25))
    return (obs, act.reshape(t, n, 6), logp.reshape(t, n), torch.randn(t, n),
            torch.randn(t, n), val.reshape(t, n))


def test_target_kl_early_stop():
    cfg = Config(hidden=32, num_envs=2, num_steps=16, update_epochs=6, num_minibatches=2)
    ppo = PPO(cfg, 25, 6)
    assert ppo.update(*batch(ppo))["epochs"] == 6
    cfg.target_kl = 1e-12
    ppo = PPO(cfg, 25, 6)
    assert ppo.update(*batch(ppo))["epochs"] == 1
