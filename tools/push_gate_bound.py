"""Could a lower push gate have let the curriculum beat lift-from-scratch? A lower bound from the logs, no training.

Reads the seed-0 curriculum run (runs/lift_cur_nominal/log.jsonl, stuck on push) and the from-scratch run
(runs/lift_scr_nominal/summary.json). For each push gate g: the first step at which push's rolling success
(100-episode window, cleared on promotion, so only rows with >= 100 push episodes count) reached g. The
curriculum cannot reach lift >= 0.6 before it leaves push, so that step is a lower bound on its steps-to-lift
with gate g, for this seed and this run's trajectory (a different gate does not change the push trajectory
before promotion).

    python tools/push_gate_bound.py    # writes runs/push_gate_bound.json
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = [json.loads(line) for line in open(os.path.join(ROOT, "runs/lift_cur_nominal/log.jsonl"))]
scratch = json.load(open(os.path.join(ROOT, "runs/lift_scr_nominal/summary.json")))["reached_threshold_at"]["lift"]
promo_eps = max(r["episodes"] for r in rows if r["stage"] == "reach")
push = [r for r in rows if r["stage"] == "push" and r["episodes"] - promo_eps >= 100]
gates = [round(0.05 * i, 2) for i in range(2, 17)]
first = {g: next((r["step"] for r in push if r["rolling_success"] >= g), None) for g in gates}
out = {"from_scratch_steps_to_lift": scratch, "push_peak": max(r["rolling_success"] for r in push),
       "first_step_push_at_gate": {str(g): s for g, s in first.items()},
       "max_gate_promoting_before_scratch": max([g for g, s in first.items() if s is not None and s < scratch],
                                                default=None)}
with open(os.path.join(ROOT, "runs/push_gate_bound.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
