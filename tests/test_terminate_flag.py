import numpy as np

import train
from so_arm100_rl.ppo import Config
from so_arm100_sim.env import ArmEnv


def test_make_vec_forwards_terminate_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(train, "VecEnv", lambda **kw: seen.update(kw))
    train.make_vec(Config(terminate_on_success=False), "lift")
    assert seen["terminate_on_success"] is False
    train.make_vec(Config(), "lift")
    assert seen["terminate_on_success"] is True


def test_no_terminate_runs_to_step_limit():
    """With the flag off an episode only ever ends by truncation at the limit."""
    env = ArmEnv("lift", obs_mode="state", action_mode="delta", seed=0,
                 terminate_on_success=False)
    env.reset()
    for t in range(env.episode_steps):
        _, _, done, info = env.step(np.zeros(6))
        assert not info["terminated"]
    assert done and info["truncated"]
    env.close()
