"""Export the deterministic policy (normaliser + actor mean) as one ONNX graph,
verify it against PyTorch, and time it on one thread.

    python export_onnx.py --ckpt runs/lift_cur_nominal/policy.pt --out runs/policy.onnx

The normaliser is folded INTO the graph so the deployed artifact takes raw
observations: a policy whose preprocessing lives in a separate script is the
classic way a deployment silently diverges from its evaluation.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                       # noqa: E402
import onnxruntime as ort                # noqa: E402
import torch                             # noqa: E402
import torch.nn as nn                    # noqa: E402

from so_arm100_sim.evaluate import provenance     # noqa: E402

from so_arm100_rl.ppo import FrozenPolicy, PPO    # noqa: E402


class Deployable(nn.Module):
    """raw obs -> clipped mean action, normaliser constants baked in."""

    def __init__(self, agent, norm):
        super().__init__()
        self.mu = agent.mu
        self.register_buffer("mean", torch.as_tensor(norm.mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(norm.std, dtype=torch.float32))
        self.clip = float(norm.clip)
        self.count = float(norm.count)

    def forward(self, obs):
        if self.count >= 2:
            z = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
        else:
            z = torch.clamp(obs, -self.clip, self.clip)
        return torch.clamp(self.mu(z), -1.0, 1.0)


def export(agent, norm, path, obs_dim):
    m = Deployable(agent, norm).eval()
    x = torch.zeros(1, obs_dim)
    torch.onnx.export(m, (x,), path, opset_version=17, input_names=["obs"],
                      output_names=["action"], dynamo=False)
    return m, os.path.getsize(path)


def verify(m, path, obs_dim, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.normal(0, 2, (n, obs_dim)).astype(np.float32)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = m(torch.from_numpy(xs)).numpy()
    out = np.concatenate([s.run(None, {"obs": xs[i:i + 1]})[0] for i in range(n)])
    return float(np.abs(ref - out).max()), s


def latency(session, obs_dim, n=2000):
    x = np.zeros((1, obs_dim), np.float32)
    for _ in range(50):
        session.run(None, {"obs": x})
    ts = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        session.run(None, {"obs": x})
        ts[i] = (time.perf_counter() - t0) * 1e3
    return {"p50_ms": float(np.percentile(ts, 50)), "p99_ms": float(np.percentile(ts, 99)),
            "mean_ms": float(ts.mean()), "n": n, "threads": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/policy.onnx")
    ap.add_argument("--save", default="runs/onnx.json")
    a = ap.parse_args()
    ppo, _ = PPO.load(a.ckpt, restore_rng=False)
    pol = FrozenPolicy(ppo.agent, ppo.norm)
    m, nbytes = export(pol.agent, pol.norm, a.out, ppo.agent.obs_dim)
    diff, s = verify(m, a.out, ppo.agent.obs_dim)
    lat = latency(s, ppo.agent.obs_dim)
    ok = diff < 1e-5
    print(f"exported {a.out} ({nbytes / 1e3:.1f} kB)  max|torch-onnx| {diff:.2e}  "
          f"{'IDENTICAL' if ok else 'DIFFERENT -- export is wrong'}")
    print(f"1-thread latency  p50 {lat['p50_ms']:.3f} ms  p99 {lat['p99_ms']:.3f} ms  (n={lat['n']})")
    res = {"ckpt": a.ckpt, "onnx": a.out, "bytes": nbytes, "max_abs_diff": diff, "verified": ok,
           "latency": lat, "params": sum(p.numel() for p in pol.agent.mu.parameters()), **provenance()}
    os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
    json.dump(res, open(a.save, "w"), indent=2)
    print(f"wrote {a.save}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
