# so-arm100-rl

PPO on the [so-arm100-sim](https://github.com/AungKaung1928/so-arm100-sim)
tabletop tasks, with a reach → push → lift curriculum, with and without
domain randomisation, evaluated on physics the policy never trained on.
CPU only: 8 forked MuJoCo processes, one thread each.

The question this repository answers is not "can PPO learn to lift a cube in
simulation" -- it can -- but two measured ones:

1. **Does training under randomised physics buy success on physics outside
   the randomisation range?** The bench defines seven held-out cells
   (`heavy`, `slippery`, `weak`, `laggy`, `noisy`, `small`, and `nominal`
   as the reference), each outside the DR range in at least one factor. A
   nominal-trained and a DR-trained policy are evaluated on all of them
   under the same 100-episode × 5-seed protocol, and the table below is the
   result.
2. **Does the curriculum buy anything?** Steps-to-threshold on `lift` with
   the reach → push → lift curriculum versus `lift` from scratch, same
   budget, same seeds.

**Status: code complete, tests green, nothing trained yet.** Every number
marked `TODO(measure)` comes from the named command and its JSON, never from
a keyboard.

## What is in the box

```
so_arm100_rl/
  ppo.py           PPO: Gaussian actor, separate critic, GAE with termination AND truncation,
                   running normaliser, checkpoints, throughput monitor
  curriculum.py    stage promotion on rolling success; steps-to-threshold bookkeeping
  normalizer.py    Welford running mean/var, frozen for evaluation and export
  boxcheck.py      refuse to start on a busy machine
train.py           the training entry point; chunked, resumable
eval_gap.py        bench protocol on every held-out physics cell -> runs/gap_<tag>.json
compare_gap.py     nominal-trained vs DR-trained, the headline table
export_onnx.py     normaliser + actor mean as one ONNX graph, verified, timed on one thread
tests/             16 checks incl. a hand-computed GAE case and a bit-exact resume
```

## The algorithm, and the two places it differs from a textbook PPO

The loop is adapted from the author's `ppo-from-scratch` (clipped surrogate,
GAE, orthogonal init, separate trunks). Two things the cart-pole never
forced:

- **Two kinds of episode end.** These tasks *terminate* on success (there is
  no future to bootstrap) and *truncate* on the step limit (there is, and it
  is the value of the state the episode ended in, which the bench's vector
  env carries as `terminal_obs`). Treating a truncation as a termination
  zeroes the bootstrap and corrupts the value target for roughly
  1/(1−γ) ≈ 100 steps before every step limit. `tests/test_gae.py` has a
  four-step case with one of each, worked by hand.
- **A running observation normaliser.** The 25-d state mixes radians,
  metres and a scaled velocity; group RMS spans more than an order of
  magnitude. The normaliser is updated from rollouts and **frozen** at
  evaluation and export, so the policy measured is the policy trained.

Actions are clipped to [−1, 1] at the env boundary while the log-probability
uses the unclipped sample. That is biased and common; the saturation
fraction is logged and reported so its size is visible.

## Curriculum

Stages `reach → push → lift` (`pick_place` with `--task pick_place`).
Promotion when the rolling success over the last 100 finished episodes
reaches 0.8 (reach, push) or 0.6 (lift). Weights and normaliser are kept
across the promotion, the learning-rate anneal restarts, the vector env's
task is swapped. `--no-curriculum` trains the final task from scratch with
the same budget.

What transfers between stages is the joint-space skill of putting the end
effector at a point; what does not is anything about contact. Whether that
is worth the steps spent on reach and push is the measurement:

| run | steps to lift ≥ 0.6 (rolling 100) | final eval success | seeds |
|---|---|---|---|
| curriculum | TODO(measure) | TODO(measure) | 3 |
| from scratch | TODO(measure) | TODO(measure) | 3 |

```
nice -n 10 python train.py --task lift --curriculum    --out runs/lift_cur_s0 --seed 0
nice -n 10 python train.py --task lift --no-curriculum --out runs/lift_scr_s0 --seed 0
```
`runs/<tag>/summary.json → reached_threshold_at`.

## Domain randomisation and the held-out gap

`--dr full` samples the bench's `DRConfig` at every reset: cube mass
0.5–2×, sliding friction 0.5–1.5×, servo gain 0.6–1.4×, joint damping and
frictionloss 0.5–2×, cube half-extent 10–15 mm, 0–2 steps of action latency,
observation noise. `--dr none` trains on the vendored model as is.

Both policies are then evaluated on the bench's `EVAL_PHYSICS`, every cell of
which sits **outside** that range in at least one factor, plus a `dr` cell
(in-range sampling, the DR policy's home turf). The headline table:

```
python eval_gap.py --ckpt runs/lift_cur_nominal/policy.pt --task lift --tag nominal
python eval_gap.py --ckpt runs/lift_cur_dr/policy.pt      --task lift --tag dr
python compare_gap.py runs/gap_nominal.json runs/gap_dr.json
```

| physics | nominal-trained | DR-trained | delta |
|---|---|---|---|
| nominal | TODO(measure) | TODO(measure) | |
| heavy (mass 3×) | TODO(measure) | TODO(measure) | |
| slippery (friction 0.3×) | TODO(measure) | TODO(measure) | |
| weak (kp 0.45×) | TODO(measure) | TODO(measure) | |
| laggy (3 steps) | TODO(measure) | TODO(measure) | |
| noisy (3× obs noise) | TODO(measure) | TODO(measure) | |
| small (9 mm) | TODO(measure) | TODO(measure) | |
| dr (in-range) | TODO(measure) | TODO(measure) | |
| **mean drop vs own nominal** | TODO(measure) | TODO(measure) | |

Each cell is 100 episodes × 5 seeds; ± is the sample standard deviation over
seeds. The scripted expert's own numbers on the same cells are in the bench
repository and are the ceiling to read these against.

What the table will and will not show: a smaller mean drop for the DR policy
is evidence that randomisation over *these* factors transfers to shifts in
*these* factors. It says nothing about a real SO-ARM100, whose unmodelled
effects (backlash, cable friction, camera latency) are not in either column.

## Budget

One env step is 25 physics steps plus observation and reward. The bench's
throughput sweep gives env-steps/s at 8 processes; PPO adds the forward
passes and the update on top.

```
budget_hours = total_steps / (env_steps_per_s_training × 3600)
```

| quantity | value |
|---|---|
| training env-steps/s, 8 workers (`train.py` log, mean over updates) | TODO(measure) |
| steps to lift ≥ 0.6 with curriculum | TODO(measure) |
| wall-clock for one 20 M-step run | TODO(measure) |

Runs are chunked (`--chunk-steps`, `--resume`) because the machine is shared
and is never left unattended; the checkpoint carries model, optimizer,
normaliser, RNG and curriculum state, and `tests/test_checkpoint.py` asserts
two updates after a reload equal two without one, bit for bit. The
throughput monitor warns when the rolling env-steps/s drops more than 20 %
below the first five updates; the box cannot read its own temperature, and
CPU contention looks identical, so the warning says to check both.

## ONNX export

```
python export_onnx.py --ckpt runs/lift_cur_dr/policy.pt --out runs/policy.onnx
```

The normaliser is folded into the graph so the artifact takes raw
observations; the export is verified against PyTorch to 1e-5 on 1000 random
observations before any latency is quoted.

| | value |
|---|---|
| 1-thread latency p50 / p99, ms | TODO(measure) |
| max |torch − onnx| | TODO(measure) |

## Reproducing

```
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt          # includes the bench from its tag
pip install -e .
./verify.sh                              # tests + the measurement commands
```

`docker build -t so-arm100-rl . && docker run --rm so-arm100-rl` runs the
tests in a clean image (no GL; nothing here renders).

## Limits, stated

- State-based: the policy sees the cube's true position. Image-based control
  is the imitation and language repositories' problem, not this one's.
- The DR ranges and the held-out cells are the bench's; a different choice
  gives a different table.
- One arm, one cube, four tasks. Nothing about clutter, occlusion or
  multi-object reasoning.
- No claim about hardware. See the gap section.

## Licence

MIT.
