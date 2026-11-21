"""Every per-cell row of the two README gap tables matches the saved eval JSONs."""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = open(os.path.join(ROOT, "README.md")).read()


def _cells(*names):
    out = {}
    for n in names:
        out.update(json.load(open(os.path.join(ROOT, "runs", n)))["cells"])
    return out


def _tables(header):
    """Every table under this header line (the README repeats one): [{cell: [col1, col2, delta]}]."""
    starts = [m.start() for m in re.finditer(re.escape(header), README)]
    assert starts, header
    return [_rows(README[i:].splitlines()[2:]) for i in starts]


def _rows(lines):
    """{cell: [col1, col2, delta]} until the table ends; cell = first word of the label."""
    rows = {}
    for line in lines:
        if not line.startswith("|"):
            break
        c = [x.strip().replace("**", "").replace("−", "-") for x in line.strip("|").split("|")]
        if not c[0].startswith("mean drop"):
            rows[c[0].split()[0]] = c[1:]
    return rows


def _check(rows, a, b):
    fmt = lambda x: f"{x['success_rate']:.3f} ± {x['success_std']:.3f}"
    assert rows and set(rows) <= set(a) & set(b), sorted(rows)
    for k, (ca, cb, d) in rows.items():
        assert ca == fmt(a[k]), (k, ca, fmt(a[k]))
        assert cb == fmt(b[k]), (k, cb, fmt(b[k]))
        assert d == f"{b[k]['success_rate'] - a[k]['success_rate']:+.3f}", (k, d)


def test_nominal_vs_dr_table():
    tables = _tables("| physics | nominal-trained | DR fine-tuned | delta |")
    assert len(tables) == 2  # results section + reproduce section
    for rows in tables:
        assert len(rows) == 8
        _check(rows, _cells("gap_nominal.json"), _cells("gap_dr.json"))


def test_dr_vs_dr_real_table():
    (rows,) = _tables("| physics | DR fine-tuned | `--dr real` fine-tuned | delta |")
    assert len(rows) == 9
    # the DR policy's real_fit cell was evaluated separately (runs/gap_realfit_dr.json)
    _check(rows, _cells("gap_dr.json", "gap_realfit_dr.json"), _cells("gap_real.json"))


def test_every_physics_table_is_checked():
    # 3 gap tables here + the DR A/B table (tests/test_readme_dr_ab.py)
    assert len(re.findall(r"^\| physics \|", README, re.M)) == 4
