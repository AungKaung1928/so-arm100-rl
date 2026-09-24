"""--init-from copies weights and normaliser, and nothing else."""
import torch

from so_arm100_rl.ppo import PPO, Config
import train as T


def test_init_from_copies_weights_not_optimizer(tmp_path):
    torch.manual_seed(0)
    src = PPO(Config(hidden=32), 25, 6)
    src.norm.update(torch.randn(64, 25).numpy() * 3 + 1)
    src.global_step, src.update_i = 12345, 12
    path = str(tmp_path / "src.pt")
    src.save(path, extra={"curriculum": {}})

    out = str(tmp_path / "run")
    cfg = Config(task="reach", hidden=32, total_steps=128, num_envs=2, num_steps=64,
                 update_epochs=1, num_minibatches=1, eval_every=100, ckpt_every=100,
                 out=out, init_from=path)
    first = {}
    orig = T.PPO.update
    def spy(self, *a, **k):
        if not first:
            first["w"] = {n: v.clone() for n, v in self.agent.state_dict().items()}
            first["step_before"] = self.global_step
        return orig(self, *a, **k)
    T.PPO.update = spy
    try:
        T.train(cfg, verbose=False)
    finally:
        T.PPO.update = orig
    for n, v in src.agent.state_dict().items():
        assert torch.equal(first["w"][n], v), n
    assert first["step_before"] < src.global_step
    import os
    run, _ = PPO.load(os.path.join(out, "ckpt.pt"), restore_rng=False)
    assert run.norm.count > src.norm.count          # normaliser carried over, then kept updating
    assert run.global_step == 128                   # step count started fresh
