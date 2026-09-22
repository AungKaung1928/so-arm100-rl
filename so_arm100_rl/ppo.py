"""PPO for the SO-ARM100 tasks. Adapted from the author's ppo-from-scratch
(`ppo_continuous.py`): same clipped surrogate, GAE recursion, orthogonal
init and steps-to-threshold bookkeeping. What changed, and why:

  vectorised rollouts    the bench's `VecEnv` steps N forked MuJoCo
                         processes; the rollout loop is written against its
                         dict observations and autoreset semantics.

  two kinds of `done`    the cart-pole only truncated. These tasks
                         TERMINATE on success (no future to bootstrap) and
                         TRUNCATE on the step limit (bootstrap V of the state
                         the episode actually ended in, which `VecEnv`
                         carries as `info["terminal_obs"]`). Getting this
                         wrong corrupts the value target for ~1/(1-gamma)
                         steps before every boundary. `tests/test_gae.py`
                         has a hand-computed case with one of each.

  observation normaliser a running Welford estimate, updated from rollout
                         observations, frozen for evaluation and export.

  action handling        actions are clipped to [-1, 1] at the env boundary
                         while the log-probability is taken on the unclipped
                         sample. Biased, common, and the saturation fraction
                         is reported so the size of the bias is visible.

  checkpoints            model, optimizer, normaliser, RNG states, step
                         counters, curriculum state. `--resume` continues a
                         chunked run and `tests/test_checkpoint.py` asserts
                         two more updates after a resume equal two more
                         updates without one.
"""
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .normalizer import RunningNormalizer

torch.set_num_threads(1)


@dataclass
class Config:
    task: str = "lift"
    color: str = "red"
    curriculum: bool = True
    dr: str = "none"                # none | full
    total_steps: int = 20_000_000
    num_envs: int = 8
    num_steps: int = 128
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 8
    num_minibatches: int = 8
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    init_log_std: float = -0.7
    seed: int = 0
    eval_every: int = 50            # updates
    eval_episodes: int = 20
    ckpt_every: int = 50            # updates
    window: int = 100
    thresholds: dict = field(default_factory=dict)
    out: str = "runs/ppo"
    tag: str = "ppo"

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self):
        return max(1, self.batch_size // self.num_minibatches)


def layer_init(layer, std=math.sqrt(2.0), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class GaussianActorCritic(nn.Module):
    """Separate trunks. Sharing couples the value-loss scale to the policy
    gradient, which is one more thing to get wrong."""

    def __init__(self, obs_dim, act_dim, hidden=256, init_log_std=-0.7):
        super().__init__()
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0))
        self.mu = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01))
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def act(self, x, action=None):
        mean = self.mu(x)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.value(x)


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """rewards, values, dones: (T, N). `dones[t]` is True when the episode
    ended AFTER step t, so no value flows across it. Truncation bootstraps
    are folded into `rewards` by the caller, terminations get nothing.
    `last_value` is V(obs_{T}) for envs still running."""
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    running = torch.zeros(N)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
    return adv, adv + values


class PPO:
    """Owns the agent, optimizer, normaliser and the update rule. The
    rollout loop lives in `train.py` because it owns the env and curriculum."""

    def __init__(self, cfg: Config, obs_dim, act_dim):
        self.cfg = cfg
        self.agent = GaussianActorCritic(obs_dim, act_dim, cfg.hidden, cfg.init_log_std)
        self.opt = torch.optim.Adam(self.agent.parameters(), lr=cfg.lr, eps=1e-5)
        self.norm = RunningNormalizer(obs_dim)
        self.update_i = 0
        self.global_step = 0
        self.schedule_start_step = 0    # reset on curriculum promotion
        self.saturated = 0
        self.total_actions = 0

    # -- acting ----------------------------------------------------------

    def set_lr(self, total_steps):
        """Linear anneal from the schedule start to the end of the budget."""
        if not self.cfg.anneal_lr:
            return self.cfg.lr
        span = max(total_steps - self.schedule_start_step, 1)
        frac = 1.0 - (self.global_step - self.schedule_start_step) / span
        lr = self.cfg.lr * max(frac, 0.0)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    def policy_step(self, obs_raw):
        """Stochastic action for a batch of raw observations. Updates the normaliser."""
        self.norm.update(obs_raw)
        x = torch.as_tensor(self.norm.normalize(obs_raw))
        with torch.no_grad():
            action, logp, _, value = self.agent.act(x)
        a = action.numpy()
        self.saturated += int((np.abs(a) > 1.0).sum())
        self.total_actions += a.size
        return x, action, logp, value

    def mean_action(self, obs_raw):
        x = torch.as_tensor(self.norm.normalize(np.asarray(obs_raw, np.float32)))
        with torch.no_grad():
            return np.clip(self.agent.mu(x).numpy(), -1.0, 1.0)

    def value_of(self, obs_raw):
        x = torch.as_tensor(self.norm.normalize(np.asarray(obs_raw, np.float32)))
        with torch.no_grad():
            return self.agent.value(x)

    # -- update ----------------------------------------------------------

    def update(self, b_obs, b_act, b_logp, b_adv, b_ret, b_val):
        cfg = self.cfg
        O, A = self.agent.obs_dim, self.agent.act_dim
        f_obs, f_act = b_obs.reshape(-1, O), b_act.reshape(-1, A)
        f_logp, f_adv, f_ret, f_val = (b_logp.reshape(-1), b_adv.reshape(-1),
                                       b_ret.reshape(-1), b_val.reshape(-1))
        n = f_obs.shape[0]
        idx = np.arange(n)
        clipfracs, approx_kls, pg_losses, v_losses = [], [], [], []
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, n, cfg.minibatch_size):
                mb = idx[s:s + cfg.minibatch_size]
                _, newlogp, entropy, newval = self.agent.act(f_obs[mb], f_act[mb])
                logratio = newlogp - f_logp[mb]
                ratio = logratio.exp()
                mb_adv = f_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = -torch.min(ratio * mb_adv,
                                     torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv).mean()
                if cfg.clip_vloss:
                    v_unclipped = (newval - f_ret[mb]) ** 2
                    v_clipped = (f_val[mb] + torch.clamp(newval - f_val[mb], -cfg.clip_coef, cfg.clip_coef)
                                 - f_ret[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newval - f_ret[mb]) ** 2).mean()
                loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    clipfracs.append(((ratio - 1).abs() > cfg.clip_coef).float().mean().item())
                    approx_kls.append(((ratio - 1) - logratio).mean().item())
                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
        self.update_i += 1
        return {"clipfrac": float(np.mean(clipfracs)), "approx_kl": float(np.mean(approx_kls)),
                "pg_loss": float(np.mean(pg_losses)), "v_loss": float(np.mean(v_losses)),
                "sigma": float(self.agent.log_std.detach().exp().mean()),
                "saturation": self.saturated / max(self.total_actions, 1)}

    # -- checkpoints -----------------------------------------------------

    def state_dict(self, extra=None):
        d = {"config": asdict(self.cfg), "model": self.agent.state_dict(),
             "optimizer": self.opt.state_dict(), "normalizer": self.norm.state_dict(),
             "update_i": self.update_i, "global_step": self.global_step,
             "schedule_start_step": self.schedule_start_step,
             "saturated": self.saturated, "total_actions": self.total_actions,
             "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
             "obs_dim": self.agent.obs_dim, "act_dim": self.agent.act_dim}
        if extra:
            d.update(extra)
        return d

    def save(self, path, extra=None):
        torch.save(self.state_dict(extra), path)
        return path

    @classmethod
    def load(cls, path, restore_rng=True):
        d = torch.load(path, map_location="cpu", weights_only=False)
        cfg = Config(**d["config"])
        self = cls(cfg, d["obs_dim"], d["act_dim"])
        self.agent.load_state_dict(d["model"])
        self.opt.load_state_dict(d["optimizer"])
        self.norm.load_state_dict(d["normalizer"])
        self.update_i, self.global_step = d["update_i"], d["global_step"]
        self.schedule_start_step = d.get("schedule_start_step", 0)
        self.saturated, self.total_actions = d.get("saturated", 0), d.get("total_actions", 0)
        if restore_rng:
            torch.set_rng_state(d["rng"]["torch"])
            np.random.set_state(d["rng"]["numpy"])
        return self, d


class FrozenPolicy:
    """What evaluation and export see: a frozen normaliser and the mean action."""

    def __init__(self, agent, norm):
        self.agent = agent.eval()
        self.norm = RunningNormalizer.from_state_dict(norm.state_dict()).freeze()

    @classmethod
    def from_checkpoint(cls, path):
        ppo, _ = PPO.load(path, restore_rng=False)
        return cls(ppo.agent, ppo.norm)

    def reset(self):
        pass

    def act(self, obs):
        x = torch.as_tensor(self.norm.normalize(obs["state"]))
        with torch.no_grad():
            return np.clip(self.agent.mu(x).numpy(), -1.0, 1.0)


class Throughput:
    """env-steps/s per update, with the drop warning that stands in for a
    temperature sensor: the box cannot read its own, and contention looks
    the same, so the message says to check both."""

    def __init__(self, reference_updates=5, warn_drop=0.2):
        self.rates = []
        self.reference_updates = reference_updates
        self.warn_drop = warn_drop
        self.warned = False

    def record(self, steps, seconds):
        r = steps / max(seconds, 1e-9)
        self.rates.append(r)
        return r

    def check(self):
        if len(self.rates) < self.reference_updates + 3:
            return None
        ref = float(np.mean(self.rates[:self.reference_updates]))
        recent = float(np.mean(self.rates[-3:]))
        if recent < (1.0 - self.warn_drop) * ref:
            self.warned = True
            return (f"throughput {recent:,.0f} env-steps/s is {100 * (1 - recent / ref):.0f}% below "
                    f"the first {self.reference_updates} updates ({ref:,.0f}). Either the package "
                    f"is power limiting or something else is running: check the load average "
                    f"before reading this as thermal.")
        return None

    @property
    def elapsed_rate(self):
        return float(np.mean(self.rates)) if self.rates else 0.0
