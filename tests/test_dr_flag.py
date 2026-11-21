"""train.make_vec maps every --dr value to its so-arm100-sim DRConfig."""
import train
from so_arm100_rl.ppo import Config
from so_arm100_sim.dr import DRConfig


def test_make_vec_maps_each_dr_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(train, "VecEnv", lambda **kw: seen.update(dr=kw["dr"]))
    want = {"none": None, "full": DRConfig(), "real": DRConfig.real_informed(),
            "centred": DRConfig.real_centred(), "both": DRConfig.real_both()}
    for flag, cfg in want.items():
        train.make_vec(Config(dr=flag), "lift")
        assert seen["dr"] == cfg, flag
