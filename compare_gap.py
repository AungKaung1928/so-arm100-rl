"""Merge two eval_gap.py results into the headline table:
nominal-trained vs DR-trained, per held-out physics, with the delta.

    python compare_gap.py runs/gap_nominal.json runs/gap_dr.json
"""
import json
import sys


def table(a, b, name_a="nominal-trained", name_b="DR-trained"):
    cells = [k for k in a["cells"] if k in b["cells"]]
    lines = [f"| physics | {name_a} | {name_b} | delta (DR - nominal) |", "|---|---|---|---|"]
    for k in cells:
        ra, rb = a["cells"][k], b["cells"][k]
        lines.append(f"| {k} | {ra['success_rate']:.3f} +- {ra['success_std']:.3f} "
                     f"| {rb['success_rate']:.3f} +- {rb['success_std']:.3f} "
                     f"| {rb['success_rate'] - ra['success_rate']:+.3f} |")
    base_a = a["cells"].get("nominal", {}).get("success_rate")
    base_b = b["cells"].get("nominal", {}).get("success_rate")
    if base_a is not None and base_b is not None:
        held = [k for k in cells if k not in ("nominal", "dr")]
        gap_a = sum(base_a - a["cells"][k]["success_rate"] for k in held) / max(len(held), 1)
        gap_b = sum(base_b - b["cells"][k]["success_rate"] for k in held) / max(len(held), 1)
        lines.append(f"| **mean drop vs own nominal** | {gap_a:.3f} | {gap_b:.3f} | {gap_b - gap_a:+.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    print(f"task {a['task']}, {a['episodes']} episodes x {a['seeds']} seeds per cell\n")
    print(table(a, b))
