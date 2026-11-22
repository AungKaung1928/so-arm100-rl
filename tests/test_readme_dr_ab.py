"""The README DR A/B table is exactly what dr_ab.py computes from the tracked runs/gap_ab_*.json."""
import json
import os

import dr_ab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = open(os.path.join(ROOT, "docs", "DETAILS.md")).read()

SEEDS = (0, 1, 2, 3)   # every A/B analysis covers the same training seeds

def test_dr_ab_table_matches_runs():
    s = dr_ab.summarise(dr_ab.load(os.path.join(ROOT, "runs")))
    assert s["train_seeds"] == [0, 1, 2, 3]
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


def test_other_cells_sentence_matches_runs():
    s = dr_ab.summarise(dr_ab.load(os.path.join(ROOT, "runs"), prefixes=("gap_abx",)))
    assert s["train_seeds"] == [0, 1, 2, 3]
    text = " ".join(README.split())
    for c in ("heavy", "weak", "noisy", "small"):
        m, sd = s["cells"][c]["diff_mean_sd"]
        assert f"{c} {m:+.3f} ± {sd:.3f}" in text, c
        assert abs(m) < sd and not all(d > 0 for d in s["cells"][c]["diff"]), c
    # slippery is inert: identical to nominal for every policy
    assert s["cells"]["slippery"]["diff"] == dr_ab.summarise(dr_ab.load(os.path.join(ROOT, "runs")))["cells"]["nominal"]["diff"]


def test_lag_servo_sentence_matches_runs():
    s = dr_ab.summarise(dr_ab.load(os.path.join(ROOT, "runs"), prefixes=("gap_lagsoft",)))
    r = s["cells"]["lagsoft"]
    assert s["train_seeds"] == list(SEEDS) and all(d > 0 for d in r["diff"])
    text = " ".join(README.split())
    assert "arm A does not recover ({:.3f} ± {:.3f})".format(*r["full_mean_sd"]) in text
    assert "B stays ahead in every seed ({:+.3f} ± {:.3f})".format(*r["diff_mean_sd"]) in text
    laggy = dr_ab.summarise(dr_ab.load(os.path.join(ROOT, "runs")))["cells"]["laggy"]
    assert r["full_mean_sd"][0] <= laggy["full_mean_sd"][0]     # "does not recover"


def test_action_stats_sentence_matches_runs():
    d = json.load(open(os.path.join(ROOT, "runs", "action_stats.json")))["policies"]
    assert len(d) == 4 * len(SEEDS)      # 2 physics x 2 arms
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
    assert len(d) == 2 * len(SEEDS)
    text = " ".join(README.split())

    def rng(arm, key):
        v = [d[f"{arm}_s{s}"][key] for s in SEEDS]
        return f"{100 * min(v):.0f}–{100 * max(v):.0f} %"
    assert (f"still fail are {rng('full', 'failed_after_grasp')} for A and "
            f"{rng('centred', 'failed_after_grasp')} for B, lower for B in every seed") in text
    assert all(d[f"centred_s{s}"]["failed_after_grasp"] < d[f"full_s{s}"]["failed_after_grasp"] for s in SEEDS)
    assert (f"no grasp at all are {rng('full', 'failed_no_grasp')} for A and "
            f"{rng('centred', 'failed_no_grasp')} for B") in text


def test_grasp_contact_sentence_matches_runs():
    d = json.load(open(os.path.join(ROOT, "runs", "lag_failmode.json")))["policies"]
    text = " ".join(README.split())

    def rng(arm, key, pct=False):
        v = [d[f"{arm}_s{s}"][key] for s in SEEDS]
        return f"{100 * min(v):.0f}–{100 * max(v):.0f}" if pct else f"{min(v)}–{max(v)}"
    assert (f"({rng('centred', 'grasp_onsets')} onsets per 100 episodes against "
            f"{rng('full', 'grasp_onsets')} for A, more in every seed)") in text
    assert all(d[f"centred_s{s}"]["grasp_onsets"] > d[f"full_s{s}"]["grasp_onsets"] for s in SEEDS)
    more = [s for s in SEEDS if d[f"centred_s{s}"]["grasp_losses"] > d[f"full_s{s}"]["grasp_losses"]]
    (k,) = [s for s in SEEDS if s not in more]
    assert len(more) == 3 and (f"loses more of them in three of four seeds ({rng('centred', 'grasp_losses')} against "
                               f"{rng('full', 'grasp_losses')}; seed {k}: {d[f'centred_s{k}']['grasp_losses']} against "
                               f"{d[f'full_s{k}']['grasp_losses']})") in text
    lo = min(v["losses_on_table"] for v in d.values()); hi = max(v["losses_on_table"] for v in d.values())
    assert f"{100 * lo:.0f}–{100 * hi:.0f} % of losses with the cube still on the table" in text
