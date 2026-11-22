"""Training-log numbers quoted in docs/DETAILS.md, from runs/<run>/log.jsonl + summary.json -> runs/training_summary.json.

The logs and per-run summaries are git-ignored (large); this keeps the few numbers the docs quote, so a test can check them.
Chunked runs (--resume) keep one log; summary.json's env_steps_per_s_mean covers the last invocation only, so the
earlier chunk is the log prefix whose remaining tail has exactly that mean.

    python tools/training_summary.py            # writes runs/training_summary.json
"""
import json
import os
import sys

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"


def load(run):
    with open(os.path.join(RUNS, run, "log.jsonl")) as f:
        log = [json.loads(line) for line in f]
    path = os.path.join(RUNS, run, "summary.json")      # absent for a run stopped by hand
    if not os.path.exists(path):
        return log, None
    with open(path) as f:
        return log, json.load(f)


def chunks(log, last_mean):
    """Mean env-steps/s per invocation: split where the log tail's mean equals the last invocation's mean."""
    sp = [x["env_steps_per_s"] for x in log]
    tail = 0.0
    for k in range(len(sp) - 1, 0, -1):
        tail += sp[k]
        if abs(tail / (len(sp) - k) - last_mean) < 1e-6:
            return [sum(sp[:k]) / k, last_mean], log[k - 1]["step"]
    return [last_mean], None


out = {}
log, s = load("lift_scr_nominal")
means, split = chunks(log, s["env_steps_per_s_mean"])
out["lift_scr_nominal"] = {"steps": s["steps"], "chunk_env_steps_per_s": means, "chunk_split_step": split}
log, _ = load("lift_scr_dr_stalled")
out["lift_scr_dr_stalled"] = {"last_step": log[-1]["step"], "max_rolling_success": max(x["rolling_success"] or 0 for x in log)}
log, s = load("lift_ft_dr")
out["lift_ft_dr"] = {"steps": s["steps"], "env_steps_per_s_mean": s["env_steps_per_s_mean"]}
log, s = load("lift_cur_nominal")
out["lift_cur_nominal"] = {"steps": s["steps"], "final_stage": s["final_stage"], "transitions": s["transitions"],
                           "final_eval_success": s["final_eval_success"],
                           "stages_seen": sorted({x["stage"] for x in log})}
with open(os.path.join(RUNS, "training_summary.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
