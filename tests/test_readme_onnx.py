"""The README ONNX numbers (summary row and the export table) are what runs/onnx.json recorded."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = open(os.path.join(ROOT, "docs", "DETAILS.md")).read()
O = json.load(open(os.path.join(ROOT, "runs", "onnx.json")))
LAT = f"{O['latency']['p50_ms']:.3f} / {O['latency']['p99_ms']:.3f}"
DIFF = f"{O['max_abs_diff']:.1e}".replace("e-0", "e-")
SIZE = f"{O['bytes'] / 1000:.0f} kB / {O['params']:,} parameters"


def test_summary_row():
    assert O["verified"] and O["latency"]["n"] == 2000 and O["latency"]["threads"] == 1
    assert (f"| ONNX export (DR fine-tune) | 1-thread latency p50 / p99 | {LAT} ms | "
            f"n = 2000; max \\|torch − onnx\\| {DIFF}; {SIZE} |") in README


def test_export_table():
    for line in (f"| 1-thread latency p50 / p99, ms (n = 2000) | {LAT} |",
                 f"| max \\|torch − onnx\\| (1000 observations) | {DIFF} |",
                 f"| size / parameters | {SIZE.replace(' parameters', '')} |"):
        assert line in README, line
