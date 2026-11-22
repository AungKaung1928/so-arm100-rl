"""DETAILS training rows (throughput per chunk, stalled DR-from-scratch, curriculum outcome) match runs/training_summary.json."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETAILS = open(os.path.join(ROOT, "docs", "DETAILS.md")).read()
LINES = DETAILS.splitlines()
T = json.load(open(os.path.join(ROOT, "runs", "training_summary.json")))


def test_throughput_per_chunk_and_dr_finetune():
    c1, c2 = T["lift_scr_nominal"]["chunk_env_steps_per_s"]
    pair = f"{c1:,.0f} (chunk 1), {c2:,.0f} (chunk 2)"
    ft = T["lift_ft_dr"]["env_steps_per_s_mean"]
    assert f"| env-steps/s | {pair} | chunk 2 on a shared host; DR fine-tune {ft:,.0f} |" in DETAILS
    assert f"(`train.py` log, mean over updates) | {pair} |" in DETAILS


def test_dr_from_scratch_stalled():
    r = T["lift_scr_dr_stalled"]
    assert f"| rolling success | ≤ {r['max_rolling_success']:.2f} through {r['last_step'] / 1e6:.1f} M | stalled, stopped |" in DETAILS


def test_curriculum_never_reached_lift():
    r = T["lift_cur_nominal"]
    assert "lift" not in r["stages_seen"] and r["final_stage"] == "push"
    n = f"{r['steps'] / 1e6:.0f} M"
    assert f"| not reached in {n} | stuck on push; {r['final_eval_success']:.2f} on push, lift never started |" in DETAILS
    assert f"| curriculum | not reached in {n} (stuck on push) | {r['final_eval_success']:.2f} on push; lift never started |" in DETAILS
