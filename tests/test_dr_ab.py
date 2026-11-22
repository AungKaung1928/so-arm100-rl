"""dr_ab.py: pairing by training seed, the paired difference and the nominal - real_fit gap."""
import json

import pytest

import dr_ab


def _write(tmp_path, arm, seed, **cells):
    (tmp_path / f"gap_ab_{arm}_s{seed}.json").write_text(
        json.dumps({"cells": {c: {"success_rate": v} for c, v in cells.items()}}))


def test_paired_difference_and_gap(tmp_path):
    _write(tmp_path, "full", 0, nominal=0.90, real_fit=0.80)
    _write(tmp_path, "centred", 0, nominal=0.85, real_fit=0.84)
    _write(tmp_path, "full", 1, nominal=1.00, real_fit=0.70)
    _write(tmp_path, "centred", 1, nominal=0.95, real_fit=0.90)
    _write(tmp_path, "full", 2, nominal=1.00, real_fit=1.00)      # no centred run: seed 2 is dropped
    s = dr_ab.summarise(dr_ab.load(str(tmp_path)))
    assert s["train_seeds"] == [0, 1]
    assert s["cells"]["real_fit"]["diff"] == pytest.approx([0.04, 0.20])
    assert s["cells"]["real_fit"]["diff_mean_sd"][0] == pytest.approx(0.12)
    gap = s["gap_nominal_minus_real_fit"]
    assert gap["full"]["per_seed"] == pytest.approx([0.10, 0.30])
    assert gap["centred"]["mean_sd"][0] == pytest.approx(0.03)
    assert "| real_fit | 0.750 +- 0.071 | 0.870 +- 0.042 | +0.120 +- 0.113 |" in dr_ab.table(s)


def test_one_seed_has_zero_spread(tmp_path):
    _write(tmp_path, "full", 9, nominal=0.5, real_fit=0.5)
    _write(tmp_path, "centred", 9, nominal=0.6, real_fit=0.7)
    s = dr_ab.summarise(dr_ab.load(str(tmp_path)))
    assert s["cells"]["nominal"]["diff_mean_sd"] == pytest.approx((0.1, 0.0))


def test_other_b_arm_pairs_with_full(tmp_path):
    _write(tmp_path, "full", 0, nominal=0.90, real_fit=0.80)
    _write(tmp_path, "centred", 0, nominal=0.10, real_fit=0.10)    # ignored when b="both"
    _write(tmp_path, "both", 0, nominal=0.95, real_fit=0.90)
    s = dr_ab.summarise(dr_ab.load(str(tmp_path), b="both"), b="both")
    assert s["b"] == "both" and s["cells"]["real_fit"]["diff"] == pytest.approx([0.10])
    t = dr_ab.table(s)
    assert "B: --dr both" in t and "| real_fit | 0.800 +- 0.000 | 0.900 +- 0.000 | +0.100 +- 0.000 |" in t


def test_prefixes_merge_cells_and_dropped_cells_are_listed(tmp_path):
    for arm in ("full", "both"):
        _write(tmp_path, arm, 0, nominal=0.9, real_fit=0.8)
    (tmp_path / "gap_abx_full_s0.json").write_text(json.dumps({"cells": {"weak": {"success_rate": 0.5}}}))
    _write(tmp_path, "both", 0, nominal=0.9, real_fit=0.8, weak=0.7)
    s = dr_ab.summarise(dr_ab.load(str(tmp_path), b="both"), b="both")
    assert "weak" not in s["cells"] and s["dropped_cells"] == ["weak"]
    s = dr_ab.summarise(dr_ab.load(str(tmp_path), b="both", prefixes=("gap_ab", "gap_abx")), b="both")
    assert s["cells"]["weak"]["diff"] == pytest.approx([0.2]) and s["dropped_cells"] == []


def test_prefixes_refuse_different_checkpoints(tmp_path):
    for prefix, ckpt in (("gap_ab", "a.pt"), ("gap_abx", "b.pt")):
        (tmp_path / f"{prefix}_full_s0.json").write_text(json.dumps({"ckpt": ckpt, "cells": {}}))
    with pytest.raises(ValueError):
        dr_ab.load(str(tmp_path), prefixes=("gap_ab", "gap_abx"))
