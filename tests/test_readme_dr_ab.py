"""The README DR A/B table is exactly what dr_ab.py computes from the tracked runs/gap_ab_*.json."""
import os

import dr_ab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = open(os.path.join(ROOT, "README.md")).read()


def test_dr_ab_table_matches_runs():
    s = dr_ab.summarise(dr_ab.load(os.path.join(ROOT, "runs")))
    assert s["train_seeds"] == [0, 1, 2]
    for line in dr_ab.table(s).splitlines():
        assert line in README, line


def test_laggy_claim_matches_runs():
    s = dr_ab.summarise(dr_ab.load(os.path.join(ROOT, "runs")))
    lag = s["cells"]["laggy"]
    assert all(d > 0 for d in lag["diff"])
    assert "(" + ", ".join(f"{d:+.3f}" for d in lag["diff"]) + ")" in README
    assert f"(sd {lag['centred_mean_sd'][1]:.3f} vs {lag['full_mean_sd'][1]:.3f})" in README
    for c in ("nominal", "real_fit"):        # "smaller than their spread"
        m, sd = s["cells"][c]["diff_mean_sd"]
        assert abs(m) < sd, c


def test_other_cells_sentence_matches_runs(tmp_path):
    for f in os.listdir(os.path.join(ROOT, "runs")):     # gap_abx_* -> gap_ab_* so dr_ab.load pairs them
        if f.startswith("gap_abx_"):
            (tmp_path / f.replace("gap_abx_", "gap_ab_")).symlink_to(os.path.join(ROOT, "runs", f))
    s = dr_ab.summarise(dr_ab.load(str(tmp_path)))
    assert s["train_seeds"] == [0, 1, 2]
    text = " ".join(README.split())
    for c in ("heavy", "weak", "noisy", "small"):
        m, sd = s["cells"][c]["diff_mean_sd"]
        assert f"{c} {m:+.3f} ± {sd:.3f}" in text, c
        assert abs(m) < sd and not all(d > 0 for d in s["cells"][c]["diff"]), c
    # slippery is inert: identical to nominal for every policy
    assert s["cells"]["slippery"]["diff"] == dr_ab.summarise(dr_ab.load(os.path.join(ROOT, "runs")))["cells"]["nominal"]["diff"]
