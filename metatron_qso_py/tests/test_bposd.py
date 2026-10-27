"""BP+OSD decoding, CSS codes and their memory experiments."""

import itertools

import numpy as np
import pytest

import metatron_qso as q


def check_matrix(code: q.CssCode) -> np.ndarray:
    h = np.zeros((len(code.z_checks), code.n), dtype=np.uint8)
    for i, c in enumerate(code.z_checks):
        h[i, c] = 1
    return h


def test_code_parameters() -> None:
    for d, n in [(3, 7), (5, 19), (7, 37)]:
        c = q.CssCode.color_code(d)
        assert (c.n, c.k) == (n, 1)
        assert c.x_checks == c.z_checks
    for n, k, d in [(72, 12, 6), (90, 8, 10), (144, 12, 12)]:
        c, distance = q.CssCode.bivariate_bicycle_code(n)
        assert (c.n, c.k, distance) == (n, k, d)
    gross = q.CssCode.bivariate_bicycle(12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])
    assert (gross.n, gross.k) == (144, 12)
    # Logicals pair up and commute with the other type's checks.
    hx = np.zeros((len(gross.x_checks), gross.n), dtype=np.uint8)
    for i, c in enumerate(gross.x_checks):
        hx[i, c] = 1
    zl = np.zeros((gross.k, gross.n), dtype=np.uint8)
    xl = np.zeros((gross.k, gross.n), dtype=np.uint8)
    for i, (x, z) in enumerate(zip(gross.x_logicals, gross.z_logicals)):
        xl[i, x] = 1
        zl[i, z] = 1
    assert not (hx @ zl.T % 2).any()
    assert np.array_equal(xl @ zl.T % 2, np.eye(gross.k, dtype=np.uint8))
    steane = q.CssCode(7, [[3, 4, 5, 6], [1, 2, 5, 6], [0, 2, 4, 6]], [[3, 4, 5, 6], [1, 2, 5, 6], [0, 2, 4, 6]])
    assert steane.k == 1
    assert "CssCode(n=7" in repr(steane)


def test_bposd_corrects_low_weight_errors() -> None:
    code = q.CssCode.color_code(5)
    h = check_matrix(code)
    p = 0.01
    dem = q.DetectorErrorModel(h.shape[0], [(p, list(np.flatnonzero(h[:, j])), 0) for j in range(code.n)])
    decoder = q.BpOsdDecoder(dem)
    for w in (1, 2):
        for qubits in itertools.combinations(range(code.n), w):
            e = np.zeros(code.n, dtype=np.uint8)
            e[list(qubits)] = 1
            s = h @ e % 2
            out = decoder.decode_detailed(list(np.flatnonzero(s)))
            x = np.zeros(code.n, dtype=np.uint8)
            x[out.errors] = 1
            assert np.array_equal(h @ x % 2, s)
            # The correction equals the error up to a stabilizer.
            residual = (x + e) % 2
            assert not any(sum(residual[z]) % 2 for z in code.z_logicals)


def test_exhaustive_osd_is_minimum_weight() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        m, n = 5, 10
        h = (rng.random((m, n)) < 0.4).astype(np.uint8)
        probs = 0.01 + 0.3 * rng.random(n)
        dem = q.DetectorErrorModel(m, [(float(probs[j]), list(np.flatnonzero(h[:, j])), 0) for j in range(n)])
        decoder = q.BpOsdDecoder(dem, max_iterations=0, osd="exhaustive", osd_order=20)
        llr = np.log((1 - probs) / probs)
        for bits in range(1 << m):
            s = np.array([bits >> i & 1 for i in range(m)], dtype=np.uint8)
            best = None
            for subset in range(1 << n):
                x = np.array([subset >> j & 1 for j in range(n)], dtype=np.uint8)
                if np.array_equal(h @ x % 2, s):
                    w = float(llr @ x)
                    best = w if best is None else min(best, w)
            if best is None:
                with pytest.raises(ValueError):
                    decoder.decode(list(np.flatnonzero(s)))
            else:
                assert decoder.decode_detailed(list(np.flatnonzero(s))).weight == pytest.approx(best, abs=1e-9)


def test_css_memory_and_logical_error_rate() -> None:
    code = q.CssCode.color_code(5)
    for basis in ("z", "x"):
        c = q.css_code_memory(code, 5, 0.002, basis=basis)
        assert c.n_detectors == 6 * len(code.z_checks)
        assert c.n_observables == 1
        det, obs = c.without_noise().sample(100)
        assert not det.any() and not obs.any()
    c = q.css_code_memory(code, 5, 0.002)
    dem = c.detector_error_model()
    decoder = q.BpOsdDecoder(dem)
    det, obs = c.sample(500, seed=2)
    predicted = decoder.decode_batch(det)
    actual = obs.astype(np.uint64) @ (1 << np.arange(obs.shape[1], dtype=np.uint64))
    failures = int(np.sum(predicted != actual))
    assert failures < 25
    rate = q.logical_error_rate(c, 500, seed=2, decoder="bposd")
    assert rate.errors < 25
    # Union-find needs a matching graph: colour-code errors flip three
    # detectors.
    with pytest.raises(ValueError):
        q.logical_error_rate(c, 10, decoder="union_find")


def test_invalid_input_raises() -> None:
    with pytest.raises(ValueError):
        q.CssCode.color_code(4)
    with pytest.raises(ValueError):
        q.CssCode.bivariate_bicycle_code(100)
    with pytest.raises(ValueError):
        q.CssCode(2, [[0]], [[0, 1]])
    with pytest.raises(ValueError):
        q.DetectorErrorModel(2, [(0.1, [2], 0)])
    with pytest.raises(ValueError):
        q.DetectorErrorModel(2, [(1.5, [0], 0)])
    with pytest.raises(ValueError):
        q.DetectorErrorModel(2, [(0.1, [0, 0], 0)])
    with pytest.raises(ValueError):
        q.DetectorErrorModel(2, [(0.1, [0], 2)], n_observables=1)
    dem = q.DetectorErrorModel(2, [(0.1, [0, 1], 1)])
    with pytest.raises(ValueError):
        q.BpOsdDecoder(dem, method="sum")
    with pytest.raises(ValueError):
        q.BpOsdDecoder(dem, osd="osd2")
    with pytest.raises(ValueError):
        q.BpOsdDecoder(dem, scaling=2.0)
    d = q.BpOsdDecoder(dem)
    assert d.decode([0, 1]) == 1
    with pytest.raises(ValueError):
        d.decode([0])
    with pytest.raises(ValueError):
        d.decode([5])
    with pytest.raises(ValueError):
        q.css_code_memory(q.CssCode.color_code(3), 0)
    with pytest.raises(ValueError):
        q.logical_error_rate(q.css_code_memory(q.CssCode.color_code(3), 1), 10, decoder="bp")
