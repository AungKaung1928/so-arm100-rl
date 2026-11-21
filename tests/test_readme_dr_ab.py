"""The README DR A/B table is exactly what dr_ab.py computes from the tracked runs/gap_ab_*.json."""
import json
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


def test_lag_servo_sentence_matches_runs(tmp_path):
    for f in os.listdir(os.path.join(ROOT, "runs")):
        if f.startswith("gap_lagsoft_"):
            (tmp_path / f.replace("gap_lagsoft_", "gap_ab_")).symlink_to(os.path.join(ROOT, "runs", f))
    s = dr_ab.summarise(dr_ab.load(str(tmp_path)))
    r = s["cells"]["lagsoft"]
    assert s["train_seeds"] == [0, 1, 2] and all(d > 0 for d in r["diff"])
    text = " ".join(README.split())
    assert "arm A does not recover ({:.3f} ± {:.3f})".format(*r["full_mean_sd"]) in text
    assert "B stays ahead in every seed ({:+.3f} ± {:.3f})".format(*r["diff_mean_sd"]) in text
    laggy = dr_ab.summarise(dr_ab.load(os.path.join(ROOT, "runs")))["cells"]["laggy"]
    assert r["full_mean_sd"][0] <= laggy["full_mean_sd"][0]     # "does not recover"


def test_action_stats_sentence_matches_runs():
    d = json.load(open(os.path.join(ROOT, "runs", "action_stats.json")))["policies"]
    assert len(d) == 12      # 2 physics x 2 arms x 3 seeds
    mag = [v["mean_abs_action"] for v in d.values()]
    sat = [v["saturated_share"] for v in d.values()]
    text = " ".join(README.split())
    assert f"mean |action| {min(mag):.3f}–{max(mag):.3f} with {100 * min(sat):.0f}–{100 * max(sat):.0f} %" in text
    for key in ("mean_abs_action", "saturated_share", "mean_abs_delta"):     # "the two arms' ranges overlap"
        for phys in ("nominal", "laggy"):
            a = [v[key] for k, v in d.items() if k.startswith(f"{phys}/full_")]
            b = [v[key] for k, v in d.items() if k.startswith(f"{phys}/centred_")]
            assert min(a) <= max(b) and min(b) <= max(a), (key, phys)


def test_lag_failmode_sentence_matches_runs():
    d = json.load(open(os.path.join(ROOT, "runs", "lag_failmode.json")))["policies"]
    assert len(d) == 6
    worst = max(v["grasped_not_lifted"] for v in d.values())
    assert f"(at most {100 * worst:.0f} % of episodes for any policy" in " ".join(README.split())
    # "that is where B is better": B's no-grasp share is lower than A's, paired by training seed
    assert all(d[f"centred_s{s}"]["no_grasp"] < d[f"full_s{s}"]["no_grasp"] for s in (0, 1, 2))
