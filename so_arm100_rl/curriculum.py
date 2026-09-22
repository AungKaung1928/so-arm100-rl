"""Task curriculum: reach -> push -> lift (-> pick_place).

Promotion is decided on a rolling window of finished episodes, not on the
return, because success is the number the bench reports and the return
scales differ between tasks. On promotion the policy weights and the
observation normaliser are kept -- the joint-space skill of moving the end
effector to a point is the thing that transfers -- and the learning-rate
schedule restarts. The trainer swaps the vector env's task; this class only
decides when.

`--no-curriculum` trains the final task from scratch with the same budget so
the curriculum's value is a measured steps-to-threshold, not an assumption.
"""
from collections import deque

STAGES = ("reach", "push", "lift", "pick_place")
DEFAULT_THRESHOLDS = {"reach": 0.8, "push": 0.8, "lift": 0.6, "pick_place": 0.6}


class Curriculum:
    def __init__(self, final_task="lift", color="red", window=100, thresholds=None,
                 enabled=True, min_episodes=None):
        if final_task not in STAGES:
            raise KeyError(f"unknown task {final_task!r}")
        self.color = color
        self.window = int(window)
        self.min_episodes = int(min_episodes if min_episodes is not None else window)
        self.thresholds = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        last = STAGES.index(final_task)
        self.stages = list(STAGES[:last + 1]) if enabled else [final_task]
        self.idx = 0
        self.successes = deque(maxlen=self.window)
        self.transitions = []          # (step, from_stage, to_stage, rolling_success)
        self.reached_threshold_at = {}  # stage -> global step when the threshold was first met

    @property
    def stage(self):
        return self.stages[self.idx]

    @property
    def task(self):
        return f"{self.stage}:{self.color}"

    @property
    def is_last(self):
        return self.idx == len(self.stages) - 1

    @property
    def rolling_success(self):
        return sum(self.successes) / len(self.successes) if self.successes else 0.0

    def record(self, success, step):
        """Call once per finished episode. Returns True if a promotion is due."""
        self.successes.append(1.0 if success else 0.0)
        if (len(self.successes) >= self.min_episodes
                and self.rolling_success >= self.thresholds[self.stage]
                and self.stage not in self.reached_threshold_at):
            self.reached_threshold_at[self.stage] = int(step)
        return self.should_promote()

    def should_promote(self):
        return (not self.is_last and len(self.successes) >= self.min_episodes
                and self.rolling_success >= self.thresholds[self.stage])

    def promote(self, step):
        if self.is_last:
            return None
        frm = self.stage
        self.transitions.append((int(step), frm, self.stages[self.idx + 1], self.rolling_success))
        self.idx += 1
        self.successes.clear()
        return self.task

    def state_dict(self):
        return {"idx": self.idx, "successes": list(self.successes),
                "transitions": list(self.transitions), "stages": list(self.stages),
                "reached_threshold_at": dict(self.reached_threshold_at)}

    def load_state_dict(self, d):
        self.idx = int(d["idx"])
        self.successes = deque(d["successes"], maxlen=self.window)
        self.transitions = [tuple(t) for t in d["transitions"]]
        self.reached_threshold_at = dict(d.get("reached_threshold_at", {}))
        return self
