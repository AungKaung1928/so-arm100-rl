"""Train PPO on the bench, with or without the curriculum and DR.

    nice -n 10 python train.py --task lift --curriculum --dr none --total-steps 20000000 \
        --num-envs 8 --out runs/lift_cur_nominal --chunk-steps 6000000

`--chunk-steps` stops after that many env steps in THIS invocation and
writes a checkpoint; rerun with `--resume runs/<tag>/ckpt.pt` to continue.
`--init-from <ckpt>` starts a new run from another run's weights and
normaliser (fine-tuning), with a fresh optimizer and step count.
Runs are chunked because the machine is shared and cannot be left
unattended; nothing about the algorithm needs it.

Everything measured lands in `runs/<tag>/log.jsonl` (one row per update)
and `runs/<tag>/summary.json`. The final policy is `runs/<tag>/policy.pt`.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

from so_arm100_sim.dr import DRConfig                # noqa: E402
from so_arm100_sim.env import ArmEnv                 # noqa: E402
from so_arm100_sim.evaluate import provenance        # noqa: E402
from so_arm100_sim.vec_env import VecEnv             # noqa: E402

from so_arm100_rl.boxcheck import require_quiet_box  # noqa: E402
from so_arm100_rl.curriculum import Curriculum       # noqa: E402
from so_arm100_rl.ppo import PPO, Config, Throughput, compute_gae, FrozenPolicy   # noqa: E402


def make_vec(cfg, task):
    dr = DRConfig() if cfg.dr == "full" else None
    return VecEnv(n=cfg.num_envs, seed=cfg.seed * 1000, task=task, obs_mode="state",
                  action_mode="delta", dr=dr,
                  terminate_on_success=cfg.terminate_on_success)


def evaluate_mean_action(ppo, task, episodes, seed=777_777):
    """Deterministic (mean action) episodes on nominal physics. Cheap, for the
    training log only; the bench protocol in eval_gap.py is the real number."""
    env = ArmEnv(task, obs_mode="state", action_mode="delta", seed=seed)
    succ, rets = [], []
    for j in range(episodes):
        obs = env.reset(seed=seed + j)
        total = 0.0
        while True:
            obs, r, done, info = env.step(ppo.mean_action(obs["state"]))
            total += r
            if done:
                succ.append(float(info["success_ever"]))
                rets.append(total)
                break
    env.close()
    return float(np.mean(succ)), float(np.mean(rets))


def train(cfg: Config, resume=None, chunk_steps=None, verbose=True):
    os.makedirs(cfg.out, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cur = Curriculum(final_task=cfg.task, color=cfg.color, window=cfg.window,
                     thresholds=cfg.thresholds, enabled=cfg.curriculum)
    probe = ArmEnv(cur.task, obs_mode="state", action_mode="delta")
    obs_dim, act_dim = probe.state_dim, 6
    probe.close()

    if resume:
        ppo, ck = PPO.load(resume)
        cur.load_state_dict(ck["curriculum"])
        cfg = ppo.cfg
        if verbose:
            print(f"resumed from {resume} at step {ppo.global_step:,}, stage {cur.stage}")
    else:
        ppo = PPO(cfg, obs_dim, act_dim)
        if cfg.init_from:
            src, _ = PPO.load(cfg.init_from, restore_rng=False)
            ppo.agent.load_state_dict(src.agent.state_dict())
            ppo.norm.load_state_dict(src.norm.state_dict())
            if verbose:
                print(f"initialised from {cfg.init_from} (weights + normaliser; optimizer, "
                      f"step count and LR schedule start fresh)")

    vec = make_vec(cfg, cur.task)
    obs = vec.reset()["state"]
    N, T = cfg.num_envs, cfg.num_steps
    O, A = obs_dim, act_dim
    b_obs = torch.zeros(T, N, O)
    b_act = torch.zeros(T, N, A)
    b_logp = torch.zeros(T, N)
    b_rew = torch.zeros(T, N)
    b_done = torch.zeros(T, N)
    b_val = torch.zeros(T, N)

    ep_ret = np.zeros(N)
    ep_returns_window = []
    thr = Throughput()
    log_path = os.path.join(cfg.out, "log.jsonl")
    log = open(log_path, "a")
    chunk_start = ppo.global_step
    t_start = time.perf_counter()
    last_eval = None

    while ppo.global_step < cfg.total_steps:
        if chunk_steps and ppo.global_step - chunk_start >= chunk_steps:
            break
        lr = ppo.set_lr(cfg.total_steps)
        t0 = time.perf_counter()
        promoted_to = None
        for t in range(T):
            x, action, logp, value = ppo.policy_step(obs)
            b_obs[t], b_act[t], b_logp[t], b_val[t] = x, action, logp, value
            nxt, rew, done, infos = vec.step(np.clip(action.numpy(), -1.0, 1.0))
            rew = rew.astype(np.float32)
            ep_ret += rew
            for i, info in enumerate(infos):
                if done[i]:
                    if info["truncated"] and not info["terminated"]:
                        # bootstrap the state the episode actually ended in
                        rew[i] += cfg.gamma * float(ppo.value_of(info["terminal_obs"]["state"]))
                    ep_returns_window.append(ep_ret[i])
                    ep_ret[i] = 0.0
                    if cur.record(info["success_ever"], ppo.global_step) and promoted_to is None:
                        promoted_to = cur.promote(ppo.global_step)
            b_rew[t] = torch.as_tensor(rew)
            b_done[t] = torch.as_tensor(done.astype(np.float32))
            obs = nxt["state"]
            ppo.global_step += N

        last_value = ppo.value_of(obs)
        adv, ret = compute_gae(b_rew, b_val, b_done, last_value, cfg.gamma, cfg.gae_lambda)
        stats = ppo.update(b_obs, b_act, b_logp, adv, ret, b_val)
        rate = thr.record(N * T, time.perf_counter() - t0)
        warn = thr.check()

        if promoted_to is not None:
            vec.close()
            vec = make_vec(cfg, promoted_to)
            obs = vec.reset()["state"]
            ep_ret[:] = 0.0
            ppo.schedule_start_step = ppo.global_step
            if verbose:
                print(f"  == promoted to {promoted_to} at step {ppo.global_step:,} "
                      f"(rolling success {cur.transitions[-1][3]:.2f})")

        row = {"update": ppo.update_i, "step": ppo.global_step, "stage": cur.stage, "lr": lr,
               "env_steps_per_s": rate, "rolling_success": cur.rolling_success,
               "rolling_return": float(np.mean(ep_returns_window[-100:])) if ep_returns_window else None,
               "episodes": len(ep_returns_window), **stats}
        if ppo.update_i % cfg.eval_every == 0:
            s, r = evaluate_mean_action(ppo, cur.task, cfg.eval_episodes)
            row["eval_success"], row["eval_return"] = s, r
            last_eval = (s, r)
            ppo.save(os.path.join(cfg.out, "policy.pt"), {"curriculum": cur.state_dict()})
        if ppo.update_i % cfg.ckpt_every == 0:
            ppo.save(os.path.join(cfg.out, "ckpt.pt"), {"curriculum": cur.state_dict()})
        log.write(json.dumps(row) + "\n")
        log.flush()
        if verbose and (ppo.update_i % 5 == 0 or ppo.update_i == 1):
            print(f"  upd {ppo.update_i:>5}  step {ppo.global_step:>10,}  {cur.stage:<10} "
                  f"succ {cur.rolling_success:5.2f}  ret {row['rolling_return'] or 0:8.2f}  "
                  f"{rate:7,.0f} env-steps/s  clip {stats['clipfrac']:.3f}  sigma {stats['sigma']:.3f}  kl {stats['approx_kl']:.4f}  ep {stats['epochs']}"
                  + (f"  eval {row['eval_success']:.2f}" if "eval_success" in row else ""))
        if warn and verbose:
            print("  WARNING", warn)

    vec.close()
    log.close()
    ppo.save(os.path.join(cfg.out, "ckpt.pt"), {"curriculum": cur.state_dict()})
    ppo.save(os.path.join(cfg.out, "policy.pt"), {"curriculum": cur.state_dict()})
    if last_eval is None or ppo.update_i % cfg.eval_every != 0:
        last_eval = evaluate_mean_action(ppo, cur.task, cfg.eval_episodes)
    summary = {
        "config": asdict(cfg), "steps": ppo.global_step, "updates": ppo.update_i,
        "final_stage": cur.stage, "transitions": cur.transitions,
        "reached_threshold_at": cur.reached_threshold_at,
        "final_eval_success": last_eval[0], "final_eval_return": last_eval[1],
        "saturation_fraction": ppo.saturated / max(ppo.total_actions, 1),
        "env_steps_per_s_mean": thr.elapsed_rate, "throughput_warning": thr.warned,
        "wall_s_this_invocation": time.perf_counter() - t_start,
        "finished": ppo.global_step >= cfg.total_steps, **provenance(),
    }
    with open(os.path.join(cfg.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    for k, v in asdict(d).items():
        if isinstance(v, bool) or isinstance(v, dict):
            continue
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--curriculum", dest="curriculum", action="store_true", default=True)
    p.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    p.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false", default=True)
    p.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false", default=True)
    p.add_argument("--no-terminate-on-success", dest="terminate_on_success",
                   action="store_false", default=True)
    p.add_argument("--threshold", action="append", default=[],
                   help="stage=value, e.g. --threshold lift=0.5 (repeatable)")
    p.add_argument("--chunk-steps", type=int, default=0)
    p.add_argument("--resume", default="")
    p.add_argument("--force", action="store_true", help="start even if the box is busy. Do not.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = parse()
    require_quiet_box(a.force, quiet=a.quiet)
    kw = {k: getattr(a, k) for k in asdict(Config()) if hasattr(a, k)}
    kw["thresholds"] = {s: float(v) for s, v in (t.split("=") for t in a.threshold)}
    if a.dr not in ("none", "full"):
        raise SystemExit("--dr must be none or full")
    cfg = Config(**kw)
    print(f"=== so-arm100-rl  task={cfg.task}  curriculum={cfg.curriculum}  dr={cfg.dr}  "
          f"seed={cfg.seed}  out={cfg.out} ===")
    s = train(cfg, resume=a.resume or None, chunk_steps=a.chunk_steps or None, verbose=not a.quiet)
    print(f"  steps {s['steps']:,}  final stage {s['final_stage']}  eval success {s['final_eval_success']:.2f}")
    print(f"  transitions {s['transitions']}")
    print(f"  saturation {s['saturation_fraction']:.1%}  mean {s['env_steps_per_s_mean']:,.0f} env-steps/s"
          f"  finished={s['finished']}")
