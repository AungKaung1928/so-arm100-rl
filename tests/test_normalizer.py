import numpy as np

from so_arm100_rl.normalizer import RunningNormalizer


def test_matches_numpy_over_batches():
    rng = np.random.default_rng(0)
    data = rng.normal([1.0, -3.0, 10.0], [0.1, 2.0, 5.0], (1000, 3))
    n = RunningNormalizer(3)
    for chunk in np.array_split(data, 7):
        n.update(chunk)
    assert np.allclose(n.mean, data.mean(0), atol=1e-9)
    assert np.allclose(n.var, data.var(0, ddof=1), atol=1e-9)
    z = n.normalize(data)
    assert np.allclose(z.mean(0), 0.0, atol=1e-3) and np.allclose(z.std(0), 1.0, atol=1e-2)


def test_clip_and_cold_start():
    n = RunningNormalizer(2, clip=3.0)
    assert np.array_equal(n.normalize(np.array([100.0, -100.0])), [3.0, -3.0])
    n.update(np.zeros((1, 2)))
    assert n.count == 1 and np.array_equal(n.normalize(np.array([1.0, 1.0])), [1.0, 1.0])


def test_state_dict_round_trip_and_freeze():
    n = RunningNormalizer(2)
    n.update(np.random.default_rng(1).normal(size=(50, 2)))
    m = RunningNormalizer.from_state_dict(n.state_dict())
    x = np.array([[0.3, -0.2]], np.float32)
    assert np.array_equal(n.normalize(x), m.normalize(x))
    m.freeze()
    before = m.mean.copy()
    m.update(np.ones((10, 2)) * 100)
    assert np.array_equal(m.mean, before)
