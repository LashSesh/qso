"""Gaussian boson sampling, hafnians and imperfect photons."""

import math

import numpy as np
import pytest

import metatron_qso as q


def test_hafnians():
    ones = np.ones((8, 8))
    assert q.hafnian(ones) == pytest.approx(105)
    assert q.hafnian(ones, loop=True) == pytest.approx(764)
    a = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=complex)
    assert q.hafnian(a) == 0
    b = np.array([[1.0, 2.0], [2.0, 3.0]])
    # Repeating each row twice: the 4 × 4 matrix of the expansion.
    expanded = b[np.ix_([0, 0, 1, 1], [0, 0, 1, 1])]
    assert q.hafnian_repeated(b, [2, 2]) == pytest.approx(q.hafnian(expanded))
    mu = np.array([0.5, -1.0])
    with_loops = expanded.astype(complex)
    np.fill_diagonal(with_loops, mu[[0, 0, 1, 1]])
    assert q.hafnian_repeated(b, [2, 2], mu) == pytest.approx(q.hafnian(with_loops, loop=True))
    assert q.torontonian(np.zeros((4, 4))) == pytest.approx(0)
    with pytest.raises(ValueError):
        q.hafnian(np.array([[0, 1], [2, 0]]))


def test_squeezed_state_statistics():
    r = 0.7
    s = q.GaussianState(1).squeeze(0, r)
    assert s.mean_photon_numbers()[0] == pytest.approx(math.sinh(r) ** 2)
    for k in range(5):
        want = math.tanh(r) ** (2 * k) * math.factorial(2 * k) / (4**k * math.factorial(k) ** 2 * math.cosh(r))
        assert s.probability([2 * k]) == pytest.approx(want, abs=1e-14)
        assert s.probability([2 * k + 1]) == pytest.approx(0, abs=1e-15)
    t = q.GaussianState.thermal([1.0])
    assert t.click_probability([False]) == pytest.approx(0.5)


def test_gbs_on_an_interferometer():
    u = q.haar_unitary(3, seed=2)
    s = q.GaussianState(3)
    for i in range(3):
        s.squeeze(i, 0.4, 0.3 * i)
    s.interferometer(u).loss(1, 0.8)
    total = sum(
        s.probability([a, b, c]) for a in range(13) for b in range(13 - a) for c in range(13 - a - b)
    )
    assert total == pytest.approx(1, abs=1e-5)  # 3·10⁻⁶ lies beyond 12 photons
    clicks = sum(s.click_probability([bool(m >> i & 1) for i in range(3)]) for m in range(8))
    assert clicks == pytest.approx(1)
    samples = s.sample(200, seed=1)
    assert samples.shape == (200, 3)
    assert (s.sample(5, seed=4) == s.sample(5, seed=4)).all()
    c = s.sample_clicks(100, seed=1)
    assert set(np.unique(c)) <= {0, 1}
    r = s.reduced([0, 2])
    assert r.n_modes == 2 and r.cov.shape == (4, 4)
    copy = s.copy().rotate(0, 0.3)
    assert not np.allclose(copy.cov, s.cov)
    with pytest.raises(ValueError):
        q.GaussianState.from_parts(0.5 * np.eye(2), np.zeros(2))
    with pytest.raises(ValueError):
        q.GaussianState(1).displace(0, 0.3).click_probability([True])


def test_imperfect_photons():
    bs = q.BosonSampler(q.beam_splitter(math.pi / 4), [1, 1])
    for x in [0.0, 0.5, 1.0]:
        s = q.ImperfectBosonSampler(bs, indistinguishability=x)
        assert s.probability([1, 1]) == pytest.approx((1 - x) / 2)
    states = np.array([[1, 0], [1, 1]], dtype=complex)
    s = q.ImperfectBosonSampler(bs, internal_states=states)
    assert s.gram[0, 1] == pytest.approx(1 / math.sqrt(2))
    assert s.probability([1, 1]) == pytest.approx(0.25)
    lossy = q.ImperfectBosonSampler(bs, transmission=[0.5, 0.5], efficiency=[0.9, 0.9])
    patterns, probabilities = lossy.distribution()
    assert probabilities.sum() == pytest.approx(1)
    assert patterns.shape[1] == 2
    assert lossy.sample(10, seed=1).shape == (10, 2)
    with pytest.raises(ValueError):
        q.ImperfectBosonSampler(bs, indistinguishability=0.5, gram=np.eye(2))
