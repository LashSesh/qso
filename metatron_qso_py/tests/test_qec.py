"""Quantum error correction: detector circuits, sampling, error models and
the union-find decoder (the same checks as metatron-qso-rs/src/qec)."""

import math

import numpy as np
import pytest

import metatron_qso as q


def bit_flip_repetition(d: int, p: float) -> q.DetectorCircuit:
    """One round of a distance-d repetition code with bit flips on the data."""
    c = q.DetectorCircuit(2 * d - 1)
    c.x_error(p, list(range(d)))
    for i in range(d - 1):
        c.cx(i, d + i).cx(i + 1, d + i)
    m = [c.measure(d + i) for i in range(d - 1)]
    for r in m:
        c.detector([r])
    data = [c.measure(i) for i in range(d)]
    for i in range(d - 1):
        c.detector([m[i], data[i], data[i + 1]])
    c.observable(0, [data[0]])
    return c


def test_a_hand_built_circuit() -> None:
    c = bit_flip_repetition(3, 0.1)
    assert (c.n_qubits, c.n_measurements, c.n_detectors, c.n_observables) == (5, 5, 4, 1)
    det, obs = c.sample(20_000, seed=1)
    assert det.shape == (20_000, 4) and det.dtype == np.bool_
    assert obs.shape == (20_000, 1)
    # Detectors of the final round never fire: the data bits agree with the
    # parities they were measured with.
    assert not det[:, 2:].any()
    # Observable 0 flips with the X error on data qubit 0.
    assert obs.mean() == pytest.approx(0.1, abs=0.01)
    dem = c.detector_error_model()
    assert dem.n_detectors == 4 and len(dem) == 3
    assert sorted(e[1] for e in dem.errors) == [[0], [0, 1], [1]]
    assert all(e[0] == pytest.approx(0.1, abs=1e-15) for e in dem.errors)


def test_decoding_a_repetition_code_is_majority_vote() -> None:
    d, p = 5, 0.15
    c = bit_flip_repetition(d, p)
    decoder = q.UnionFindDecoder(c.detector_error_model(decompose=True))
    det, obs = c.sample(100_000, seed=2)
    predicted = decoder.decode_batch(det)
    assert predicted.dtype == np.uint64
    assert [decoder.decode(list(np.flatnonzero(row))) for row in det[:50]] == list(
        predicted[:50]
    )
    failures = np.mean(predicted != obs[:, 0])
    majority = sum(math.comb(d, k) * p**k * (1 - p) ** (d - k) for k in range(3, d + 1))
    assert failures == pytest.approx(majority, abs=4 * math.sqrt(majority / 100_000))
    r = q.logical_error_rate(c, 100_000, seed=2)
    assert r.errors == int(np.sum(predicted != obs[:, 0]))
    assert r.rate == pytest.approx(failures)


def test_matching_decodes_at_least_as_well_as_union_find() -> None:
    c = q.surface_code_memory(5, 5, 0.006)
    dem = c.detector_error_model(decompose=True)
    matching = q.MatchingDecoder(dem)
    det, obs = c.sample(5000, seed=3)
    predicted = matching.decode_batch(det)
    assert [matching.decode(list(np.flatnonzero(row))) for row in det[:50]] == list(predicted[:50])
    # Every single fault is corrected.
    for _, detectors, flips in dem.errors:
        assert matching.decode(detectors) == flips
    uf = q.logical_error_rate(c, 5000, seed=3)
    mw = q.logical_error_rate(c, 5000, seed=3, decoder="matching")
    assert mw.errors == int(np.sum(predicted != obs[:, 0]))
    assert mw.errors < uf.errors
    assert uf == q.logical_error_rate(c, 5000, seed=3, decoder="union_find")
    with pytest.raises(ValueError, match="unknown decoder"):
        q.logical_error_rate(c, 10, decoder="blossom")  # type: ignore[arg-type]
    assert repr(matching).startswith("MatchingDecoder(detectors=")


def test_stim_circuits_import() -> None:
    c = q.surface_code_memory(3, 3, 0.002)
    back = q.DetectorCircuit.from_stim(c.to_stim())
    assert back.to_stim() == c.to_stim()
    assert back.n_detectors == c.n_detectors
    text = "R 0 1\nREPEAT 2 {\n  X_ERROR(0.1) 0\n  CX 0 1\n  MR 1\n  DETECTOR rec[-1]\n}\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]"
    r = q.DetectorCircuit.from_stim(text)
    assert (r.n_qubits, r.n_measurements, r.n_detectors, r.n_observables) == (2, 3, 2, 1)
    with pytest.raises(ValueError, match="MPP is not supported"):
        q.DetectorCircuit.from_stim("MPP X0*X1")


def test_surface_code_below_threshold() -> None:
    noise = q.CircuitNoise.uniform(0.001)
    rates = [q.logical_error_rate(q.surface_code_memory(d, d, noise), 50_000, seed=7) for d in (3, 5)]
    assert rates[1].rate + 3 * rates[1].standard_error < rates[0].rate - 3 * rates[0].standard_error
    # A float is the same uniform noise.
    a = q.surface_code_memory(3, 2, 0.002)
    b = q.surface_code_memory(3, 2, q.CircuitNoise.uniform(0.002))
    assert a.to_stim() == b.to_stim()
    assert q.surface_code_memory(5, 5).n_detectors == 72


def test_stim_export_and_appended_circuits() -> None:
    c = q.DetectorCircuit(2).append(q.Circuit(2).h(0).cx(0, 1))
    m0, m1 = c.measure(0, flip=0.01), c.measure(1)
    c.depolarize2(0.02, [0, 1]).detector([m0, m1])
    c.observable(0, [m1])
    assert c.to_stim() == (
        "H 0\nCX 0 1\nM(0.01) 0\nM 1\nDEPOLARIZE2(0.02) 0 1\n"
        "DETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    )
    assert "DEPOLARIZE2" not in c.without_noise().to_stim()
    noise = q.CircuitNoise(measurement=0.01, before_round=0.02)
    assert noise == q.CircuitNoise(before_round=0.02, measurement=0.01)
    assert (noise.after_gate2, noise.measurement) == (0.0, 0.01)


def test_invalid_input_raises() -> None:
    with pytest.raises(ValueError):
        q.DetectorCircuit(0)
    c = q.DetectorCircuit(2)
    with pytest.raises(ValueError):
        c.measure(2)
    with pytest.raises(ValueError):
        c.depolarize1(0.9, [0])
    with pytest.raises(ValueError):
        c.detector([0])
    with pytest.raises(ValueError, match="Clifford"):
        c.append(q.Circuit(1).t(0))
    with pytest.raises(ValueError):
        q.surface_code_memory(4, 1)
    r = q.DetectorCircuit(1).h(0)
    r.detector([r.measure(0)])
    with pytest.raises(ValueError, match="deterministic"):
        r.sample(10)
    decoder = q.UnionFindDecoder(bit_flip_repetition(3, 0.1).detector_error_model(True))
    with pytest.raises(ValueError, match="columns"):
        decoder.decode_batch(np.zeros((2, 3), dtype=bool))
