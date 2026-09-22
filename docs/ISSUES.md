# Issues to open on GitHub

One block per issue: title, then body. The known gaps at the first release.

---

**`pick_place` is not in the default curriculum**

The curriculum stops at `lift`. `--task pick_place` adds a fourth stage with
a 0.6 threshold, untested beyond the state machine. Decide whether the
headline tables include it once `lift` numbers exist.

---

**Tanh-squashed policy not tried**

Actions are clipped at the env boundary and the log-probability is taken on
the unclipped sample, which is biased. The saturation fraction is logged.
If it is high (> 20 %) in the real runs, implement the tanh squash with the
change-of-variables correction and compare steps-to-threshold.

---

**Asymmetric actor-critic (privileged critic, proprio actor)**

The policy currently sees the privileged 25-d state. A follow-up that moves
the actor to the 15-d proprio observation plus an image, while the critic
keeps the privileged state, would make the RL policy deployable on the same
inputs the imitation policies use. Not started.

---

**Curriculum promotion uses a fixed threshold**

Promotion at rolling success 0.8 / 0.8 / 0.6 is a choice, not a
measurement. A sweep over the `lift` threshold (0.4–0.8) against final
success would show whether the value is sensitive to it.

---

**Bit-exact resume is tested on synthetic batches only**

`tests/test_checkpoint.py` proves the optimizer, normaliser and RNG restore
exactly. A resume mid-run also recreates the vector env from its seed, so
the env states differ from an uninterrupted run; the training-level test
checks step counts and completion, not equality. Documented, not fixed.

---

**Throughput monitor cannot tell power limiting from contention**

The > 20 % drop warning fires for both. Logging the 1-minute load average
alongside each update's env-steps/s would let the log itself say which.
