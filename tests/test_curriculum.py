from so_arm100_rl.curriculum import Curriculum


def test_promotes_on_rolling_success_only_after_window_is_full():
    c = Curriculum("lift", window=10)
    assert c.task == "reach:red" and c.stages == ["reach", "push", "lift"]
    for i in range(9):
        assert not c.record(True, step=i)          # window not full yet
    assert c.record(True, step=9)                  # 10/10 >= 0.8
    assert c.promote(step=9) == "push:red"
    assert c.transitions == [(9, "reach", "push", 1.0)]
    assert c.reached_threshold_at == {"reach": 9}
    assert len(c.successes) == 0                   # window cleared


def test_threshold_per_stage_and_last_stage_never_promotes():
    c = Curriculum("lift", window=10)
    c.idx = 2
    for i in range(30):
        c.record(True, step=i)
    assert not c.should_promote() and c.promote(step=30) is None
    assert c.is_last and c.reached_threshold_at["lift"] == 9


def test_stream_below_threshold_never_promotes():
    c = Curriculum("push", window=10)
    stream = [True, False] * 20          # 0.5 < 0.8
    for i, s in enumerate(stream):
        assert not c.record(s, step=i)
    assert c.stage == "reach"


def test_disabled_curriculum_is_single_stage():
    c = Curriculum("lift", enabled=False)
    assert c.stages == ["lift"] and c.is_last


def test_state_dict_round_trip():
    c = Curriculum("pick_place", window=5, color="blue")
    for i in range(5):
        c.record(True, step=i)
    c.promote(step=5)
    c.record(False, step=6)
    d = Curriculum("pick_place", window=5, color="blue").load_state_dict(c.state_dict())
    assert d.task == "push:blue" and list(d.successes) == [0.0] and d.transitions == c.transitions


def test_custom_thresholds():
    c = Curriculum("lift", window=4, thresholds={"reach": 0.5})
    for i, s in enumerate([True, False, True, False]):
        c.record(s, step=i)
    assert c.should_promote()
