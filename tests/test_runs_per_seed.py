"""Every eval cell in runs/*.json: success_rate and success_std equal the mean and sample sd of its own per_seed list."""
import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cells_match_their_per_seed_lists():
    n = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "*.json"))):
        cells = json.load(open(path)).get("cells")
        if not isinstance(cells, dict):
            continue
        for name, c in cells.items():
            if not isinstance(c, dict) or not c.get("per_seed"):
                continue
            n += 1
            assert abs(statistics.mean(c["per_seed"]) - c["success_rate"]) < 1e-9, (path, name)
            assert abs(statistics.stdev(c["per_seed"]) - c["success_std"]) < 1e-12, (path, name)
    assert n > 0
