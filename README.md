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

**Status: measured on 2026-09-24, one training seed.** Nominal from
scratch reaches lift ≥ 0.6 at 1.0 M steps; DR from scratch stalled (rolling
success ≤ 0.09 through 4.9 M) and was stopped; a DR fine-tune of the nominal policy halves the mean
held-out drop (0.120 → 0.056), almost all of it on action latency. The
curriculum never reached lift: it cleared reach at 54 k steps, then
plateaued on push (rolling success at most 0.60 against the 0.8 gate) for
the rest of the 20 M. Every number
comes from the named command and its JSON, never from a keyboard.

**Walkthrough:** https://aungkaung1928.github.io/projects/so-arm100.html — the bench and the three policy projects built on it, explained end to end.

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
tests/             21 checks incl. a hand-computed GAE case and a bit-exact resume
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
  four-step case with one of each, worked by hand. Terminating on success
  turned out to be exploitable on `lift`, so training there runs with
  `--no-terminate-on-success`; see [Three training failures](#three-training-failures-kept-as-evidence).
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

| run | steps to lift ≥ 0.6 (rolling 100) | final eval success (20 episodes) | seeds run |
|---|---|---|---|
| curriculum | not reached in 20 M (stuck on push) | 0.45 on push; lift never started | 1 (seed 0) |
| from scratch | 0.996 M | 1.00 | 1 (seed 0) |

One seed, so these are one sample each and carry no spread; the plan's three
seeds did not fit the CPU budget. From scratch, `lift` crossed 0.6 inside the
first million steps once `--no-terminate-on-success` removed the hover
exploit, which leaves the curriculum little room to win on this task.

It did not win; it did not finish. Reach promoted at 54 k steps, then push
plateaued: rolling success peaked at 0.60 (at 17.2 M) against its 0.8 gate,
so 99.7% of the budget went to push and lift was never trained. What
failed is the gate, not the transfer: on this bench, push at 0.8 is a harder
bar than lift at 0.6 is from scratch, so a curriculum ordered by intuition
blocked the task it was meant to help. One seed. The next test is a lower
push gate or a step cap per stage, not a claim that curricula cannot help.

```
F="--dr none --seed 0 --target-kl 0.02 --no-terminate-on-success"
nice -n 10 python train.py --task lift --curriculum    $F --out runs/lift_cur_nominal
nice -n 10 python train.py --task lift --no-curriculum $F --out runs/lift_scr_nominal
```
`runs/<tag>/summary.json → reached_threshold_at`.

## Three training failures, kept as evidence

All three runs were stopped, not tuned into silence, and their logs and
checkpoints are kept (`runs/*_aborted/`, `runs/lift_scr_dr_stalled/`,
gitignored with the other runs).

**1. Step size, without a KL stop.** The first curriculum run used plain
clipped PPO, 8 epochs per update. By 4.46 M steps it was still on `push`
with rolling success 0.02, clip fraction 0.77, approximate KL 1.44 and the
policy σ collapsed from 0.50 to 0.054: each update moved the policy far past
the clip region, and the collapsed σ stopped exploration. This is the same
failure as chunk 1 of the author's biped run. Fix: `--target-kl 0.02`, which
ends the epoch loop once the last epoch's mean approximate KL exceeds the
target (`tests/test_target_kl.py`). The check is per epoch, so one update can
overshoot to about 0.04 before it stops; that is logged (`kl`, `ep`) and
visible.

**2. Reward gaming: hovering under the success line.** With the KL stop,
`lift` from scratch learned to grasp, and then its rolling success *fell*
from 0.24 to 0.10 while its return kept rising, to ~240. A probe
(`tools/probe_hover.py`, 20 episodes, mean action, no termination) showed
why: the cube was grasped for ~138 of 150 steps and held at 4.7–4.9 cm, just
below the 5.0 cm success height; 3 of 20 episodes ever crossed it. The
per-step shaping reward while grasped is up to +2.25, and success
*terminated* the episode, so crossing the line traded a future worth roughly
2.25 / (1 − γ) for a +1 bonus. The policy found the better deal. Fix:
`--no-terminate-on-success` for training only (above the line now pays
3.25/step against 2.21 below; `tests/test_terminate_flag.py`). Every
evaluation, in training and in `eval_gap.py`, still ends the episode at the
first success and counts it, so the metric the tables report did not
change -- only the incentive did. The run was stopped at 5.7 M steps.

**3. Domain randomisation from scratch stalled.** The same
recipe that crossed 0.6 on nominal physics at 1.0 M steps was run with
`--dr full`. Rolling success peaked at 0.09 at 2.7 M steps, and by 4.89 M
it had been exactly 0 for the last 700 updates, its return sat at 15–30 over the last 1,000 updates (the nominal run was at
150 by 1 M), σ had fallen from 0.50 to 0.36, and KL and clip fraction were
normal: not an unstable update but a policy that stopped improving while
mass, friction, gain and size moved every episode. No probe was run on it,
so where in the motion it fails is not measured.
One seed, so this says the recipe fails on seed 0, not that DR from scratch
cannot work with more steps or a curriculum. It was stopped
(`runs/lift_scr_dr_stalled/`) and replaced by the standard fix: start from
the nominal policy and fine-tune under DR (`--init-from`, fresh optimizer,
step count and LR schedule, 10 M steps; `tests/test_init_from.py`).

In all three runs the headline training metrics (rolling success, return) either
looked fine or lied. Clip fraction, KL, σ and a direct probe of the
behaviour showed the failures.

## Domain randomisation and the held-out gap

`--dr full` samples the bench's `DRConfig` at every reset: cube mass
0.5–2×, sliding friction 0.5–1.5×, servo gain 0.6–1.4×, joint damping and
frictionloss 0.5–2×, cube half-extent 10–15 mm, 0–2 steps of action latency,
observation noise. `--dr none` trains on the vendored model as is.

The DR policy is therefore the nominal policy fine-tuned for 10 M more
steps under `--dr full` (see failure 3), so the comparison is "same policy,
with and without a DR fine-tune", and the DR column has seen 30 M steps to
the nominal column's 20 M.

Both policies are then evaluated on the bench's `EVAL_PHYSICS`, every cell of
which sits **outside** that range in at least one factor, plus a `dr` cell
(in-range sampling, the DR policy's home turf). The headline table:

```
python eval_gap.py --ckpt runs/lift_scr_nominal/policy.pt --task lift --tag nominal --jobs 8
python train.py --task lift --no-curriculum --dr full --seed 0 --target-kl 0.02 \
    --no-terminate-on-success --total-steps 10000000 \
    --init-from runs/lift_scr_nominal/policy.pt --out runs/lift_ft_dr
python eval_gap.py --ckpt runs/lift_ft_dr/policy.pt       --task lift --tag dr      --jobs 8
python compare_gap.py runs/gap_nominal.json runs/gap_dr.json
```

| physics | nominal-trained | DR fine-tuned | delta |
|---|---|---|---|
| nominal | 0.998 ± 0.004 | 0.998 ± 0.004 | +0.000 |
| heavy (mass 3×) | 0.998 ± 0.004 | 0.988 ± 0.008 | −0.010 |
| slippery (friction 0.3×) | 0.998 ± 0.004 | 0.998 ± 0.004 | +0.000 |
| weak (kp 0.45×) | 0.982 ± 0.013 | 1.000 ± 0.000 | +0.018 |
| laggy (3 steps) | 0.458 ± 0.052 | 0.760 ± 0.030 | **+0.302** |
| noisy (3× obs noise) | 0.956 ± 0.017 | 0.948 ± 0.024 | −0.008 |
| small (9 mm) | 0.878 ± 0.051 | 0.960 ± 0.020 | +0.082 |
| dr (in-range) | 0.888 ± 0.038 | 0.958 ± 0.013 | +0.070 |
| **mean drop vs own nominal** (6 held-out cells) | 0.120 | 0.056 | −0.064 |

Each cell is 100 episodes × 5 seeds; ± is the sample standard deviation over
seeds. The mean drop is over the six held-out cells (the `dr` cell is
excluded), with a gain counted as a negative drop. Output of `compare_gap.py`. The scripted expert's own numbers on the same cells are in the bench
repository and are the ceiling to read these against.

The DR fine-tune halves the mean drop, and almost all of it is one cell:
three steps of action latency, one step past the DR range, costs the
nominal policy more than half its success (0.998 → 0.458) and the DR policy
a quarter (0.998 → 0.760). Mass and friction shifts cost neither policy
anything; this `lift` policy was never limited by them. The small cube and
the in-range DR cell gain 0.07–0.08. Two things weaken the claim: one
training seed per column (the ± is over evaluation seeds only, so it says
nothing about how a second training run would land), and the DR column
has 10 M more steps of training, so part of the gain may be training
length rather than randomisation. A 30 M-step nominal run is the control
that separates the two, and it has not been run.

What the table does and does not show: a smaller mean drop for the DR policy
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
| training env-steps/s, 8 workers (`train.py` log, mean over updates) | 3,043 (chunk 1), 2,425 (chunk 2) |
| steps to lift ≥ 0.6 with curriculum | not reached in 20 M (stage stuck at push) |
| wall-clock for one 20 M-step run | 2 h 10 min (59 + 70 min) |

Runs are chunked (`--chunk-steps`, `--resume`) because the machine is shared
and is never left unattended; the checkpoint carries model, optimizer,
normaliser, RNG and curriculum state, and `tests/test_checkpoint.py` asserts
two updates after a reload equal two without one, bit for bit. The
throughput monitor warns when the rolling env-steps/s drops more than 20 %
below the first five updates; the box cannot read its own temperature, and
CPU contention looks identical, so the warning says to check both.

The chunk-2 rate is lower because the Windows host was running other desktop
programs at the same time (host CPU 24–29 % busy, clock at 190–196 % of base,
so boosting, not throttled; nothing else ran inside WSL). The warning fired
and is recorded in `summary.json` as `throughput_warning: true`. Treat 3,043
as the clean figure and 2,425 as what a shared machine gives. Action
saturation, the fraction of sampled actions clipped at ±1, ended at 58 % for
the nominal run and 78 % for the DR fine-tune: the log-probability bias noted
above is not small on this task. The DR fine-tune ran at 2,546 env-steps/s
mean, 10 M steps in 68 min, on the same shared host.

## ONNX export

```
python export_onnx.py --ckpt runs/lift_ft_dr/policy.pt --out runs/policy.onnx
```

The normaliser is folded into the graph so the artifact takes raw
observations; the export is verified against PyTorch to 1e-5 on 1000 random
observations before any latency is quoted. The exported policy is the DR
fine-tune. At 8 µs a step the network is not the bottleneck anywhere in
this stack: in training, 8 workers at ~3,040 env-steps/s spend ~2.6 ms of
one core per env step, roughly 300 times the forward pass. Output
`runs/onnx.json`.

| | value |
|---|---|
| 1-thread latency p50 / p99, ms (n = 2000) | 0.008 / 0.015 |
| max \|torch − onnx\| (1000 observations) | 2.1e-6 |
| size / parameters | 298 kB / 73,990 |

## Reproducing

```
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt          # includes the bench from its tag
pip install -e .
./verify.sh                              # tests + the measurement commands
```

`docker build -t so-arm100-rl . && docker run --rm so-arm100-rl` runs the
tests in a clean image (no GL; nothing here renders). Image built on 2026-09-23 and its default command passed inside it (16 tests passed), image size 1.84 GB.

## Limits, stated

- State-based: the policy sees the cube's true position. Image-based control
  is the imitation and language repositories' problem, not this one's.
- The DR ranges and the held-out cells are the bench's; a different choice
  gives a different table.
- One arm, one cube, four tasks. Nothing about clutter, occlusion or
  multi-object reasoning.
- One training seed per configuration. Every ± in this README is over
  evaluation seeds of one trained policy, not over training runs.
- No claim about hardware. See the gap section.

## Licence

MIT.
