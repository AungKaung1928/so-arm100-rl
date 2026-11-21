"""README push-gate bound must match runs/push_gate_bound.json (written by tools/push_gate_bound.py)."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = " ".join(open(os.path.join(ROOT, "README.md")).read().split())
B = json.load(open(os.path.join(ROOT, "runs", "push_gate_bound.json")))


def test_push_gate_sentence_matches_json():
    f = B["first_step_push_at_gate"]
    assert f"first reached 0.5 at {f['0.5']:,} steps and 0.15 at {f['0.15']:,}" in README
    assert f"lift from scratch reached 0.6 at {B['from_scratch_steps_to_lift']:,} in total" in README
    assert f"any push gate above {B['max_gate_promoting_before_scratch']} loses" in README
    assert f"at {B['max_gate_promoting_before_scratch']} it promotes after {f['0.1']:,} steps" in README
    # the bound itself: every gate above 0.1 first reached after from-scratch finished, or never
    assert all(s is None or s > B["from_scratch_steps_to_lift"] for g, s in f.items() if float(g) > 0.1)
    assert B["push_peak"] < 0.8      # the default gate was never reached
