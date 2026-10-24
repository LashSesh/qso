import itertools
import math

import numpy as np
import pytest

import metatron_qso as q


def naive_permanent(a):
    n = a.shape[0]
    return sum(np.prod([a[i, p[i]] for i in range(n)]) for p in itertools.permutations(range(n)))


def test_permanent_matches_definition():
    rng = np.random.default_rng(1)
    for n in range(1, 7):
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        assert q.permanent(a) == pytest.approx(naive_permanent(a), rel=1e-10)
    assert q.permanent(np.ones((6, 6))) == pytest.approx(720.0)
    assert q.permanent(np.zeros((0, 0))) == 1.0


def test_haar_unitary_and_beam_splitter():
    u = q.haar_unitary(8, seed=3)
    assert u.shape == (8, 8)
    assert np.allclose(u.conj().T @ u, np.eye(8), atol=1e-12)
    assert np.array_equal(u, q.haar_unitary(8, seed=3))
    bs = q.beam_splitter(math.pi / 4)
    assert np.allclose(bs, np.array([[1, -1], [1, 1]]) / math.sqrt(2))


def test_hong_ou_mandel():
    s = q.BosonSampler(q.beam_splitter(math.pi / 4), [1, 1])
    assert s.n_modes == 2 and s.n_photons == 2 and s.input == [1, 1]
    assert s.probability([1, 1]) == pytest.approx(0.0, abs=1e-30)
    assert s.probability_distinguishable([1, 1]) == pytest.approx(0.5)
    samples = s.sample(200, seed=1)
    assert samples.shape == (200, 2)
    assert set(map(tuple, samples)) <= {(2, 0), (0, 2)}


def test_distributions_and_sampling():
    u = q.haar_unitary(5, seed=4)
    s = q.BosonSampler(u, [1, 1, 0, 1, 0])
    patterns, probs = s.distribution()
    assert patterns.shape == (35, 5) and probs.shape == (35,)
    assert probs.sum() == pytest.approx(1.0)
    assert (patterns.sum(axis=1) == 3).all()
    _, probs_d = s.distribution(distinguishable=True)
    assert probs_d.sum() == pytest.approx(1.0)
    # Individual probabilities agree with the tabulated distribution.
    assert s.probability(list(patterns[7])) == pytest.approx(probs[7])
    # Samples: reproducible, conserve photons, follow the distribution.
    a = s.sample(20_000, seed=5)
    assert np.array_equal(a, s.sample(20_000, seed=5))
    assert (a.sum(axis=1) == 3).all()
    index = {tuple(p): k for k, p in enumerate(patterns)}
    freq = np.bincount([index[tuple(t)] for t in a], minlength=len(probs)) / len(a)
    assert 0.5 * np.abs(freq - probs).sum() < 0.03
    d = s.sample(5_000, seed=6, distinguishable=True)
    assert (d.sum(axis=1) == 3).all()


def test_full_bunching_ratio_is_n_factorial():
    s = q.BosonSampler(q.haar_unitary(6, seed=7), [1, 1, 1, 0, 0, 0])
    bunched = [0, 0, 3, 0, 0, 0]
    ratio = s.probability(bunched) / s.probability_distinguishable(bunched)
    assert ratio == pytest.approx(6.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: q.BosonSampler(np.eye(3) * 2, [1, 0, 0]),
        lambda: q.BosonSampler(np.eye(3), [0, 0, 0]),
        lambda: q.BosonSampler(np.eye(3), [1, 0]),
        lambda: q.BosonSampler(np.eye(3), [2, 0, 0]).sample(1),
        lambda: q.BosonSampler(np.eye(3), [1, 1, 0]).probability([1, 0, 0]),
        lambda: q.permanent(np.ones((2, 3))),
        lambda: q.permanent(np.full((2, 2), np.nan)),
        lambda: q.haar_unitary(0),
        # Overflowing photon totals and oversized outputs are rejected
        # before anything is allocated.
        lambda: q.BosonSampler(np.eye(3), [2**64 - 1, 2, 0]),
        lambda: q.BosonSampler(np.eye(3), [1, 1, 0]).probability([2**64 - 1, 3, 0]),
        lambda: q.BosonSampler(np.eye(3), [1, 1, 0]).sample(2**26 // 3 + 1),
        lambda: q.BosonSampler(np.eye(3), [1, 1, 0]).sample(2**64 - 1, distinguishable=True),
        lambda: q.BosonSampler(np.eye(1024), [1, 1] + [0] * 1022).distribution(),
    ],
)
def test_invalid_input_raises_value_error(call):
    with pytest.raises(ValueError):
        call()
