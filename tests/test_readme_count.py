"""README quick start's 'N passed[, M skipped]' equals the number of tests pytest collects."""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_readme_test_count_matches_collection():
    readme = open(os.path.join(ROOT, "README.md")).read()
    (m,) = re.findall(r"python -m pytest -q tests +# (\d+) passed(?:, (\d+) skipped)?", readme)
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=", "-p", "no:warnings", "tests"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    collected = sum("::" in line for line in out.splitlines())
    assert collected == int(m[0]) + int(m[1] or 0), (collected, m)
