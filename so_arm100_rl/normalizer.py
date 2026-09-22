"""Running observation normaliser (Welford), and why it exists.

The 25-d state mixes radians, rad/s (already scaled by 0.1 in the bench),
metres and a jaw gap in metres. Group RMS spans more than an order of
magnitude, so an unnormalised first layer attends to whichever group is
numerically largest, not to whichever is informative. A running estimate is
used during rollouts and FROZEN at evaluation and export time, so the policy
that is measured is the policy that was trained.
"""
import numpy as np


class RunningNormalizer:
    def __init__(self, dim, clip=10.0, eps=1e-8):
        self.dim = int(dim)
        self.clip = float(clip)
        self.eps = float(eps)
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.count = 0.0
        self.frozen = False

    @property
    def var(self):
        return self.m2 / max(self.count - 1.0, 1.0)

    @property
    def std(self):
        return np.sqrt(self.var + self.eps)

    def update(self, batch):
        """Chan et al. parallel update with a batch of rows (N, dim)."""
        if self.frozen:
            return
        x = np.asarray(batch, dtype=np.float64).reshape(-1, self.dim)
        n = x.shape[0]
        if n == 0:
            return
        b_mean = x.mean(0)
        b_m2 = ((x - b_mean) ** 2).sum(0)
        tot = self.count + n
        delta = b_mean - self.mean
        self.mean = self.mean + delta * (n / tot)
        self.m2 = self.m2 + b_m2 + delta ** 2 * (self.count * n / tot)
        self.count = tot

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.count < 2:
            return np.clip(x, -self.clip, self.clip)
        z = (x - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def freeze(self):
        self.frozen = True
        return self

    def state_dict(self):
        return {"dim": self.dim, "clip": self.clip, "eps": self.eps, "mean": self.mean.copy(),
                "m2": self.m2.copy(), "count": float(self.count), "frozen": bool(self.frozen)}

    def load_state_dict(self, d):
        self.dim, self.clip, self.eps = int(d["dim"]), float(d["clip"]), float(d["eps"])
        self.mean = np.asarray(d["mean"], dtype=np.float64).copy()
        self.m2 = np.asarray(d["m2"], dtype=np.float64).copy()
        self.count = float(d["count"])
        self.frozen = bool(d.get("frozen", False))
        return self

    @classmethod
    def from_state_dict(cls, d):
        return cls(int(d["dim"])).load_state_dict(d)
