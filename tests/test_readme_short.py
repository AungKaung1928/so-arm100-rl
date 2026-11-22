"""The short README repeats rows and numbers of docs/DETAILS.md, which the other README tests check."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHORT = open(os.path.join(ROOT, "README.md")).read()
DETAILS = open(os.path.join(ROOT, "docs", "DETAILS.md")).read()


def test_short_rows_are_copies():
    rows = [l for l in SHORT.splitlines() if l.startswith("| ") and not l.startswith("| method |")]
    assert len(rows) == 5
    for l in rows:
        assert l in DETAILS.splitlines(), l


def test_short_numbers_in_details():
    flat = " ".join(DETAILS.split())
    for n in ("0.120 → 0.056", "+0.227 ± 0.122", "all four seeds"):
        assert n in SHORT, n
    # the headline and the plot's alt text each carry the laggy lead: check both lines, not one substring
    import json
    lag = json.load(open(os.path.join(ROOT, "runs", "dr_ab.json")))["cells"]["laggy"]
    lead = "+{:.3f} ± {:.3f}".format(*lag["diff_mean_sd"])
    head = [l for l in SHORT.splitlines() if l.startswith("PPO for")]
    alt = [l for l in SHORT.splitlines() if l.startswith("<img")]
    assert len(head) == 1 and lead in head[0] and len(lag["diff"]) == 4 and min(lag["diff"]) > 0
    assert len(alt) == 1 and lead in alt[0] and "4 training seeds" in alt[0]
    assert "| laggy | 0.521 +- 0.120 | 0.748 +- 0.058 | +0.227 +- 0.122 |" in DETAILS
    assert "0.120 → 0.056" in flat and "all four seeds" in flat


def test_dr_ab_plot_rebuilds_from_runs(tmp_path):
    import subprocess
    import sys

    import pytest
    assert '<img src="docs/media/dr_ab.png"' in SHORT
    assert os.path.exists(os.path.join(ROOT, "docs", "media", "dr_ab.png"))
    pytest.importorskip("matplotlib")
    out = tmp_path / "dr_ab.png"
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "plot_dr_ab.py"),
                    os.path.join(ROOT, "runs", "dr_ab.json"), str(out)], check=True)
    assert out.stat().st_size > 10_000
