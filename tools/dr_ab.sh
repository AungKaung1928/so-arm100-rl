#!/usr/bin/env bash
# DR A/B: does centring the DR on the real-fitted servo beat the default ranges?
#
#   arm A  --dr full     default DRConfig (kp x0.6-1.4, frictionloss x0.5-2.0)
#   arm B  --dr centred  DRConfig.real_centred() (kp x0.65-0.77, frictionloss x2.2-2.8)
#
# Same recipe for both arms: fine-tune runs/lift_scr_nominal/policy.pt for STEPS
# env steps, training seeds SEEDS, then eval_gap.py (100 episodes x 5 eval seeds)
# on CELLS. Every step is skipped when its output exists and a stopped training
# run resumes from its checkpoint, so rerunning this script continues where it stopped.
#
#   tools/dr_ab.sh          # ~2.5 h on 4 cores at the defaults; run it under your CPU limiter
#   python dr_ab.py         # table + runs/dr_ab.json
#   SEEDS=9 STEPS=200000 EPISODES=10 tools/dr_ab.sh    # 4-minute plumbing check
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-3000000}"
CELLS="${CELLS:-nominal real_fit laggy}"
EPISODES="${EPISODES:-100}"
export MUJOCO_GL="${MUJOCO_GL:-disable}"

quiet() {   # train.py refuses to start above load 4; wait for our own last job to drain
  for _ in $(seq 20); do
    awk '{exit !($1 < 3.5)}' /proc/loadavg && return 0
    sleep 60
  done
  echo "box still busy after 20 min" >&2; return 1
}

for s in $SEEDS; do
  for arm in full centred; do
    out=runs/ab_${arm}_s$s
    if ! grep -q '"finished": true' "$out/summary.json" 2>/dev/null; then
      quiet || exit 1
      if [ -f "$out/ckpt.pt" ]; then
        "$PY" train.py --resume "$out/ckpt.pt" --quiet || exit $?
      else
        "$PY" train.py --task lift --no-curriculum --dr "$arm" --seed "$s" \
          --target-kl 0.02 --no-terminate-on-success --total-steps "$STEPS" \
          --init-from runs/lift_scr_nominal/policy.pt --out "$out" --quiet || exit $?
      fi
      grep -q '"finished": true' "$out/summary.json" || exit 1
    fi
    if [ ! -f "runs/gap_ab_${arm}_s$s.json" ]; then
      # shellcheck disable=SC2086
      "$PY" eval_gap.py --ckpt "$out/policy.pt" --task lift --physics $CELLS \
        --episodes "$EPISODES" --tag "ab_${arm}_s$s" --jobs 3 || exit $?
    fi
  done
done
echo "dr_ab: all runs present"
