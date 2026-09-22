"""train.py end to end on a tiny budget: 2 workers, 64-step rollouts,
2048 env steps, one evaluation, one checkpoint, a resume. Under 90 s."""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(args, timeout=120):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, os.path.join(ROOT, "train.py"), *args],
                          capture_output=True, text=True, timeout=timeout, env=env, cwd=ROOT)


def test_train_smoke_and_resume(tmp_path):
    out = os.path.join(tmp_path, "smoke")
    t0 = time.perf_counter()
    r = run(["--task", "reach", "--total-steps", "2048", "--num-envs", "2", "--num-steps", "64",
             "--update-epochs", "2", "--num-minibatches", "2", "--eval-every", "8",
             "--eval-episodes", "2", "--ckpt-every", "4", "--chunk-steps", "1024",
             "--out", out, "--quiet", "--force"])
    assert r.returncode == 0, r.stdout + r.stderr
    s = json.load(open(os.path.join(out, "summary.json")))
    assert s["steps"] == 1024 and s["updates"] == 8 and not s["finished"]
    r = run(["--resume", os.path.join(out, "ckpt.pt"), "--out", out, "--quiet", "--force"])
    assert r.returncode == 0, r.stdout + r.stderr
    s = json.load(open(os.path.join(out, "summary.json")))
    assert s["steps"] == 2048 and s["updates"] == 16 and s["finished"]
    rows = [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))]
    assert len(rows) == 16 and rows[-1]["step"] == 2048
    assert any("eval_success" in row for row in rows)
    assert all(row["env_steps_per_s"] > 0 for row in rows)
    assert os.path.exists(os.path.join(out, "policy.pt"))
    assert time.perf_counter() - t0 < 90
