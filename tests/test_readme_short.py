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
    assert "| laggy | 0.521 +- 0.120 | 0.748 +- 0.058 | +0.227 +- 0.122 |" in DETAILS
    assert "0.120 → 0.056" in flat and "all four seeds" in flat
