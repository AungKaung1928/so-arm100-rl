#!/usr/bin/env bash
# Reproduce the README's claims from a clean checkout.
#
# Tier 1 is the test suite: GAE by hand, the normaliser, the curriculum
# state machine, a bit-exact resume, a 2048-step smoke of train.py and the
# ONNX round trip. It runs in well under a minute and needs no trained
# policy. Tiers 2-4 print the commands that produce every TODO(measure) in
# the README; they take hours and need the machine to themselves.
set -u
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! "$PY" -c 'import torch, mujoco, so_arm100_sim' 2>/dev/null; then
  echo "torch / mujoco / so_arm100_sim are not importable with '$PY'. From the repo root:" >&2
  echo "    python3 -m venv .venv && . .venv/bin/activate" >&2
  echo "    pip install torch --index-url https://download.pytorch.org/whl/cpu" >&2
  echo "    pip install -r requirements.txt && pip install -e ." >&2
  exit 1
fi
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

hr() { printf '\n=== %s ===\n' "$1"; }

hr "1/4  tests"
"$PY" -m pytest tests || exit 1

hr "2/4  training -- curriculum vs from scratch, nominal vs DR"
cat <<'MSG'
Each run is 8 processes for hours. Chunk it (--chunk-steps) and resume;
never leave it unattended. The box check refuses to start on a loaded machine.

    nice -n 10 python train.py --task lift --curriculum    --dr none --out runs/lift_cur_nominal --seed 0
    nice -n 10 python train.py --task lift --curriculum    --dr full --out runs/lift_cur_dr      --seed 0
    nice -n 10 python train.py --task lift --no-curriculum --dr none --out runs/lift_scr_nominal --seed 0

Repeat with --seed 1 and 2 for the steps-to-threshold table.
MSG

hr "3/4  the held-out gap table"
cat <<'MSG'
    python eval_gap.py --ckpt runs/lift_cur_nominal/policy.pt --task lift --tag nominal
    python eval_gap.py --ckpt runs/lift_cur_dr/policy.pt      --task lift --tag dr
    python compare_gap.py runs/gap_nominal.json runs/gap_dr.json
MSG

hr "4/4  ONNX export and one-thread latency"
echo "    python export_onnx.py --ckpt runs/lift_cur_dr/policy.pt --out runs/policy.onnx"
for f in runs/gap_nominal.json runs/gap_dr.json; do
  [ -f "$f" ] && "$PY" - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"\n{sys.argv[1]}: task {d['task']}, {d['episodes']} x {d['seeds']}")
for k, r in d["cells"].items():
    print(f"  {k:9s} {r['success_rate']:.3f} +- {r['success_std']:.3f}")
PY
done
exit 0
