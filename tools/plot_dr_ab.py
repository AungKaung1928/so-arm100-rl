"""Plot the DR A/B (runs/dr_ab.json) as paired seed dots: docs/media/dr_ab.png.

Usage: python tools/plot_dr_ab.py [runs/dr_ab.json] [docs/media/dr_ab.png]
Needs matplotlib (plotting only, so not in requirements.txt).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "dr_ab.json")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "docs", "media", "dr_ab.png")
CELLS = ("nominal", "real_fit", "laggy")
ARMS = (("full", "A: --dr full", "#2a78d6"), ("centred", "B: --dr centred", "#eb6834"))
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#5f5e5a", "#e4e3df", "#fcfcfb"

d = json.load(open(SRC))
seeds = d["train_seeds"]
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=100, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for i, cell in enumerate(CELLS):
    c = d["cells"][cell]
    xa, xb = i - 0.15, i + 0.15
    for a, b in zip(c["full"], c["centred"]):          # one line per training seed
        ax.plot([xa, xb], [a, b], color=GRID, lw=2, zorder=1)
    for (arm, label, color), x in zip(ARMS, (xa, xb)):
        ax.scatter([x] * len(c[arm]), c[arm], s=64, color=color, edgecolor=SURFACE, linewidth=2,
                   zorder=3, label=label if i == 0 else None)
        m = c[f"{arm}_mean_sd"][0]
        ax.plot([x - 0.07, x + 0.07], [m, m], color=INK, lw=2, zorder=4)
    dm, ds = c["diff_mean_sd"]
    ax.text(i, 0.30, f"B − A {dm:+.3f} ± {ds:.3f}", ha="center", color=INK, fontsize=9)
ax.set_xticks(range(len(CELLS)), CELLS, color=INK)
ax.set_xlim(-0.6, len(CELLS) - 0.4)
ax.set_ylim(0.25, 1.03)
ax.set_ylabel("lift success", color=MUTED)
ax.tick_params(colors=MUTED, length=0)
ax.grid(axis="y", color=GRID, lw=1)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(f"DR A/B, {len(seeds)} training seeds; dots = seeds, bar = mean, lines pair a seed",
             color=INK, fontsize=10, loc="left")
ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, facecolor=SURFACE)
print("wrote", OUT)
