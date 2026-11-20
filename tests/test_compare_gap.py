"""compare_gap.table: row deltas, shared cells only, mean drop over held-out cells."""
import json
import os

from compare_gap import table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _res(rates):
    return {"cells": {k: {"success_rate": v, "success_std": 0.01} for k, v in rates.items()}}


def _last_row(md):
    return [c.strip() for c in md.splitlines()[-1].strip("|").split("|")]


def test_hand_computed_drop_skips_nominal_dr_and_unshared_cells():
    a = _res({"nominal": 1.0, "dr": 0.9, "x": 0.8, "y": 0.6, "only_a": 0.0})
    b = _res({"nominal": 0.9, "dr": 0.95, "x": 0.9, "y": 0.7, "only_b": 0.0})
    md = table(a, b)
    assert "only_a" not in md and "only_b" not in md
    assert "| x | 0.800 +- 0.010 | 0.900 +- 0.010 | +0.100 |" in md
    # held-out = x, y. a: (0.2 + 0.4) / 2 = 0.3 ; b: (0.0 + 0.2) / 2 = 0.1
    # (with dr counted, a would be 0.233 -> wrong)
    assert _last_row(md)[1:] == ["0.300", "0.100", "-0.200"]


def test_no_drop_row_without_nominal_cell():
    md = table(_res({"x": 0.5}), _res({"x": 0.7}))
    assert "mean drop" not in md


def test_saved_runs_reproduce_readme_headline():
    # README.md:41 -> mean drop 0.120 (nominal-trained) vs 0.056 (DR), -0.064
    a, b = (json.load(open(os.path.join(ROOT, "runs", f))) for f in ("gap_nominal.json", "gap_dr.json"))
    assert _last_row(table(a, b))[1:] == ["0.120", "0.056", "-0.064"]
