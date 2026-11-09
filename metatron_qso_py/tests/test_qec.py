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
    # The sparse decoder (used by decoder="matching") finds corrections of
    # the same weight.
    sparse = q.SparseMatchingDecoder(dem)
    for row in det[:500]:
        fired = [int(d) for d in np.flatnonzero(row)]
        assert sparse.decode_with_weight(fired)[1] == matching.decode_with_weight(fired)[1]
    sparse_predicted = sparse.decode_batch(det)
    uf = q.logical_error_rate(c, 5000, seed=3)
    mw = q.logical_error_rate(c, 5000, seed=3, decoder="matching")
    assert mw.errors == int(np.sum(sparse_predicted != obs[:, 0]))
    assert mw.errors < uf.errors
    assert uf == q.logical_error_rate(c, 5000, seed=3, decoder="union_find")
    with pytest.raises(ValueError, match="unknown decoder"):
        q.logical_error_rate(c, 10, decoder="blossom")  # type: ignore[arg-type]
    assert repr(matching).startswith("MatchingDecoder(detectors=")
    assert repr(sparse).startswith("SparseMatchingDecoder(detectors=")


def test_x_basis_memory() -> None:
    noise = q.CircuitNoise.uniform(0.003)
    x = q.surface_code_memory(5, 5, noise, basis="x")
    z = q.surface_code_memory(5, 5, noise)
    assert x.n_detectors == z.n_detectors and x.to_stim() != z.to_stim()
    rx = q.logical_error_rate(x, 20_000, seed=1, decoder="matching")
    rz = q.logical_error_rate(z, 20_000, seed=1, decoder="matching")
    assert 0.5 * rz.rate < rx.rate < 2 * rz.rate
    with pytest.raises(ValueError, match="unknown basis"):
        q.surface_code_memory(3, 3, basis="y")  # type: ignore[arg-type]


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


def test_splitting_reaches_rates_far_below_sampling() -> None:
    # The repetition code of distance 3 over two rounds (15 errors) from 3 %
    # down to 10⁻⁵, against the enumeration of all 2¹⁵ configurations; the
    # leading term counts the failing pairs, and errors of the callable
    # reach Python.
    def at(p: float) -> q.DetectorCircuit:
        return q.repetition_code_memory(3, 2, p)

    exact = q.enumerated_failure_rate(at(1e-5), 15)
    assert exact["remainder"] == 0.0 and exact["configurations"] == 2**15
    est = q.splitting_failure_rate(at, 0.03, 1e-5, start_shots=20000, chains=8, steps=2000, burn_in=500, seed=3)
    z = (math.log(est["rate"]) - math.log(exact["rate"])) / est["log_standard_error"]
    assert abs(z) < 4.0, (est["rate"], exact["rate"])
    assert est["rates"][-1]["p"] == 1e-5 and est["start"].errors > 0
    lead = q.leading_failures(at(1e-5))
    assert lead["distance"] == 3 and lead["weight"] == 2
    assert lead["configurations"] == exact["failing"][2]

    def broken(p: float) -> q.DetectorCircuit:
        raise RuntimeError("no circuit")

    with pytest.raises(ValueError, match="no circuit"):
        q.splitting_failure_rate(broken, 0.03, 1e-3)
    with pytest.raises(ValueError, match="decoder"):
        q.leading_failures(at(1e-3), decoder="nearest")  # type: ignore[arg-type]


# Erasures with heralds (1.66).


def erased_then_measured(channel: str) -> q.DetectorCircuit:
    """Case A of the Rust tests (or F): a heralded channel on qubit 0, then
    M 0; detector 0 reads the herald, detector 1 the measurement."""
    c = q.DetectorCircuit(1)
    if channel == "erase":
        records = c.heralded_erase(0.1, [0])
    else:
        records = c.heralded_pauli_channel_1(0.3, 0.3, 0.3, 0.1, [0])
    assert records == [0]
    m = c.measure(0)
    assert m == 1
    assert c.detector(records) == 0
    assert c.detector([m]) == 1
    return c


def test_heralded_builders_and_stim_format() -> None:
    c = q.DetectorCircuit(3)
    assert c.heralded_erase(0.2, [0, 2, 0]) == [0, 1, 2]
    assert c.heralded_pauli_channel_1(0.01, 0.02, 0.03, 0.04, [1]) == [3]
    assert c.heralded_erase(0.1, []) == []
    assert c.measure(1) == 4
    assert c.n_measurements == 5 and len(c) == 3
    assert "operations=3" in repr(c)
    text = c.to_stim()
    assert "HERALDED_ERASE(0.2) 0 2 0" in text
    assert "HERALDED_PAULI_CHANNEL_1(0.01, 0.02, 0.03, 0.04) 1" in text
    back = q.DetectorCircuit.from_stim(text)
    assert back.to_stim() == text and back.n_measurements == 5
    rep = q.RepeatedCircuit.from_stim("REPEAT 3 {\n" + text + "}\n")
    assert rep.n_measurements == 15
    assert rep.flatten().n_measurements == 15
    # Kept at probability 0, as Stim's MPAD 0.
    quiet = c.without_noise()
    assert quiet.n_measurements == 5 and "HERALDED_ERASE(0.0) 0 2 0" in quiet.to_stim()
    with pytest.raises(ValueError):
        c.heralded_erase(1.5, [0])
    with pytest.raises(ValueError):
        c.heralded_erase(0.1, [3])
    with pytest.raises(ValueError, match="add up"):
        c.heralded_pauli_channel_1(0.25, 0.25, 0.25, 0.2500002, [0])
    with pytest.raises(ValueError, match="qubit targets"):
        q.DetectorCircuit.from_stim("HERALDED_ERASE(0.1) rec[-1]")


def test_heralded_models_follow_stims_rule() -> None:
    # Stim 1.16, detector_error_model(approximate_disjoint_errors=True).
    def model(c: q.DetectorCircuit) -> dict[tuple[int, ...], float]:
        return {tuple(e[1]): e[0] for e in c.detector_error_model().errors}

    a = erased_then_measured("erase")
    got = model(a)
    assert got.keys() == {(0,), (0, 1)}
    assert got[(0,)] == pytest.approx(0.05, rel=1e-15)
    assert got[(0, 1)] == pytest.approx(0.05, rel=1e-15)
    f = model(erased_then_measured("pauli"))
    assert f[(0,)] == pytest.approx(0.4, rel=1e-15)
    assert f[(0, 1)] == pytest.approx(0.6, rel=1e-15)

    h = a.heralded_error_model()
    assert (h.n_detectors, h.n_observables) == (2, 0)
    assert h.herald_detectors == [0] and len(h.base) == 0
    assert h.herald_probability(0) == 0.1
    assert h.outcomes(0) == [(0.5, [1], 0)]
    assert h.conditioned([0]).errors == [(0.5, [1], 0)]
    assert h.conditioned([0, 0]).errors == [] and h.conditioned([]).errors == []
    with pytest.raises(ValueError, match="not a herald detector"):
        h.conditioned([1])
    with pytest.raises(ValueError):
        h.herald_probability(1)
    with pytest.raises(ValueError):
        h.outcomes(1)
    hf = erased_then_measured("pauli").heralded_error_model(decompose=False)
    assert hf.herald_probability(0) == pytest.approx(1.0, abs=1e-15)
    [(p, dets, obs)] = hf.outcomes(0)
    assert p == pytest.approx(0.6, abs=1e-15) and dets == [1] and obs == 0
    # The repeated form folds to the same model.
    rep = q.RepeatedCircuit.from_stim(
        "REPEAT 6 {\nHERALDED_ERASE(0.1) 0\nM 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n}\n"
    )
    folded, flat = rep.heralded_error_model(), rep.flatten().heralded_error_model()
    assert folded.herald_detectors == flat.herald_detectors == [0, 2, 4, 6, 8, 10]
    assert folded.base.errors == flat.base.errors
    assert [folded.outcomes(k) for k in range(6)] == [flat.outcomes(k) for k in range(6)]
    assert "heralds=6" in repr(folded)


def test_heralded_model_from_its_parts() -> None:
    base = q.DetectorErrorModel(4, [(0.01, [2, 3], 0)], n_observables=1)
    m = q.HeraldedErrorModel(
        base, [0, 1], [0.1, 0.2], [[(0.5, [3, 2], 1)], [(0.5, [3], 0), (0.5, [3], 0)]]
    )
    assert (m.n_detectors, m.n_observables) == (4, 1)
    assert m.herald_detectors == [0, 1]
    assert m.herald_probability(1) == 0.2
    # Sorted detectors; equal parts merged (½ absorbs).
    assert m.outcomes(0) == [(0.5, [2, 3], 1)]
    assert m.outcomes(1) == [(0.5, [3], 0)]
    assert m.base.errors == [(0.01, [2, 3], 0)]
    assert m.conditioned([0, 1]).errors == [
        (0.01, [2, 3], 0),
        (0.5, [2, 3], 1),
        (0.5, [3], 0),
    ]
    # The base error and herald 0's part of the same detectors merge.
    same = q.HeraldedErrorModel(base, [0], [0.1], [[(0.3, [2, 3], 0)]])
    [(p, dets, obs)] = same.conditioned([0]).errors
    assert p == pytest.approx(0.01 * 0.7 + 0.3 * 0.99, rel=1e-15)
    assert (dets, obs) == ([2, 3], 0)
    bad: list[tuple[list[int], list[float], list[list[tuple[float, list[int], int]]]]] = [
        ([1, 0], [0.1, 0.1], [[], []]),  # not ascending
        ([0, 0], [0.1, 0.1], [[], []]),  # repeated
        ([0], [0.1, 0.1], [[]]),  # lengths differ
        ([0], [1.5], [[]]),  # not a probability
        ([4], [0.1], [[]]),  # no such detector
        ([2], [0.1], [[]]),  # the base flips it
    ]
    for detectors, probabilities, outcomes in bad:
        with pytest.raises(ValueError):
            q.HeraldedErrorModel(base, detectors, probabilities, outcomes)
    with pytest.raises(ValueError, match="herald detector"):
        q.HeraldedErrorModel(base, [0, 1], [0.1, 0.1], [[(0.5, [1], 0)], []])


def surface_capacity_d3() -> q.DetectorCircuit:
    """The rotated surface code of distance 3 at code capacity: every data
    qubit erased with probability ½ (herald detectors 0..8), then read out;
    detectors 9..12 are the Z checks, observable 0 is Z_L."""
    c = q.DetectorCircuit(9)
    for k in range(9):
        c.detector(c.heralded_erase(0.5, [k]))
    m = [c.measure(k) for k in range(9)]
    for check in Z_CHECKS:
        c.detector([m[k] for k in check])
    c.observable(0, [m[k] for k in Z_LOGICAL])
    return c


# Data qubit 3r + c of the 3 × 3 grid; X checks {0,1,3,4}, {4,5,7,8},
# {3,6}, {2,5}.
Z_CHECKS = [[0, 1], [1, 2, 4, 5], [3, 4, 6, 7], [7, 8]]
Z_LOGICAL = [0, 3, 6]


@pytest.mark.parametrize("decoder", ["union_find", "matching", "bposd"])
def test_surface_code_at_half_erasure_fails_exactly_a_quarter(decoder: str) -> None:
    # Every erased set E (probability 2⁻⁹) and every X pattern on it
    # (2^−|E| each): the decoder fails on exactly a quarter, that is
    # Σ_E Σ_x fail · 2^(9−|E|) = 2¹⁸ / 4.
    model = surface_capacity_d3().heralded_error_model()
    assert model.herald_detectors == list(range(9)) and len(model.base) == 0
    dec = q.ConditionedDecoder(model, decoder)  # type: ignore[arg-type]
    total = 0
    for erased in range(512):
        support = [k for k in range(9) if erased >> k & 1]
        for pattern in range(1 << len(support)):
            x = {support[i] for i in range(len(support)) if pattern >> i & 1}
            fired = [9 + j for j, check in enumerate(Z_CHECKS) if len(x & set(check)) % 2]
            flip = len(x & set(Z_LOGICAL)) % 2
            if dec.decode(fired, support) != flip:
                total += 1 << (9 - len(support))
    assert total == 65536


def test_conditioned_decoder_on_an_erasure_memory() -> None:
    noise = q.ErasureNoise(0.01, erasure_share=0.9)
    c = q.surface_code_memory(3, 3, noise)
    det, obs = c.sample(2000, seed=5)
    for name, decompose in [("union_find", True), ("matching", True), ("bposd", False)]:
        model = c.heralded_error_model(decompose=decompose)
        assert len(model.herald_detectors) > 0
        dec = q.ConditionedDecoder(model, name)  # type: ignore[arg-type]
        assert dec.n_detectors == c.n_detectors
        assert dec.herald_detectors == model.herald_detectors
        assert len(dec.slots()) >= len(model.base)
        heralds = set(dec.herald_detectors)
        for row in det[:200]:
            fired = [int(d) for d in np.flatnonzero(row)]
            ordinary = [d for d in fired if d not in heralds]
            erased = [d for d in fired if d in heralds]
            assert dec.decode(fired) == dec.decode(ordinary, erased)
        predicted = dec.decode_batch(det)
        assert predicted.dtype == np.uint64
        rate = q.logical_error_rate(c, 2000, seed=5, decoder=name)  # type: ignore[arg-type]
        assert rate.errors == int(np.sum(predicted != obs[:, 0]))
    with pytest.raises(ValueError, match="heralds"):
        q.logical_error_rate(c, 10, decoder="correlated_matching")
    with pytest.raises(ValueError, match="heralds"):
        q.logical_error_rate(c, 10, decoder="window_matching", window=2)
    with pytest.raises(ValueError, match="heralds"):
        q.logical_error_rate(c, 10, decoder="parallel_window_bposd", commit=1)


def test_conditioned_decoder_detail_and_errors() -> None:
    model = surface_capacity_d3().heralded_error_model()
    matching = q.ConditionedDecoder(model, "matching")
    # Erasing qubit 1 turns its edge on at ½: weight 0.
    assert matching.decode_with_weight([9, 10], [1]) == (0, 0.0)
    assert matching.decode_with_weight([9, 10, 1]) == (0, 0.0)
    assert matching.decode_with_weight([], [1]) == (0, 0.0)
    bposd = q.ConditionedDecoder(model, "bposd", osd="osd0")
    out = bposd.decode_detailed([9, 10, 1])
    assert out.observables == 0
    assert [bposd.slots()[e][1] for e in out.errors] == [[9, 10]]
    uf = q.ConditionedDecoder(model)
    assert "union_find" in repr(uf)
    with pytest.raises(ValueError, match="herald detector"):
        uf.decode([0], [])
    with pytest.raises(ValueError, match="not a herald detector"):
        uf.decode([9], [9])
    with pytest.raises(ValueError):
        uf.decode([13])
    with pytest.raises(ValueError):
        uf.decode([9], [13])
    with pytest.raises(ValueError, match="matching"):
        uf.decode_with_weight([9])
    with pytest.raises(ValueError, match="BP"):
        matching.decode_detailed([9])
    with pytest.raises(ValueError, match="bposd"):
        q.ConditionedDecoder(model, "union_find", osd="osd0")
    with pytest.raises(ValueError, match="bposd"):
        q.ConditionedDecoder(model, "matching", max_iterations=10)
    with pytest.raises(ValueError, match="unknown decoder"):
        q.ConditionedDecoder(model, "nearest")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown osd"):
        q.ConditionedDecoder(model, "bposd", osd="none")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="columns"):
        uf.decode_batch(np.zeros((2, 3), dtype=bool))
    # A part that flips three detectors is not graph-like.
    wide = q.DetectorCircuit.from_stim(
        "HERALDED_ERASE(0.1) 0\nCX 0 1 0 2\nM 0 1 2\n"
        "DETECTOR rec[-4]\nDETECTOR rec[-3]\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n"
    )
    w = wide.heralded_error_model()
    assert w.outcomes(0) == [(0.5, [1, 2, 3], 0)]
    with pytest.raises(ValueError, match="flips 3 detectors"):
        q.ConditionedDecoder(w, "matching")
    assert q.ConditionedDecoder(w, "bposd").decode([1, 2, 3], [0]) == 0


def test_parts_above_one_half_are_applied_first() -> None:
    # HERALDED_PAULI_CHANNEL_1(0, 0.5, 0, 0): a fired herald is a certain
    # X (part 1.0), applied before decoding, so no decoder ever fails, also
    # when every qubit is heralded and the syndrome is empty.
    d = 5
    qs = " ".join(map(str, range(d)))
    text = (
        f"R {qs}\nHERALDED_PAULI_CHANNEL_1(0, 0.5, 0, 0) {qs}\n"
        + "".join(f"DETECTOR rec[-{d - k}]\n" for k in range(d))
        + f"M {qs}\n"
        + "".join(f"DETECTOR rec[-{d - k}] rec[-{d - k - 1}]\n" for k in range(d - 1))
        + f"OBSERVABLE_INCLUDE(0) rec[-{d}]\n"
    )
    c = q.DetectorCircuit.from_stim(text)
    model = c.heralded_error_model()
    assert model.outcomes(0) == [(1.0, [d], 1)]
    det, obs = c.sample(5000, seed=3)
    for name, decompose in [("union_find", True), ("matching", True), ("bposd", False)]:
        dec = q.ConditionedDecoder(c.heralded_error_model(decompose=decompose), name)  # type: ignore[arg-type]
        assert int(np.sum(dec.decode_batch(det) != obs[:, 0])) == 0
        assert dec.decode([], list(range(d))) == 1
    # A Y-basis readout: X and Z have the same effect, one part at
    # (p_x + p_z)/t = 1, not the two merged as independent.
    y = q.DetectorCircuit.from_stim(
        "RY 0\nHERALDED_PAULI_CHANNEL_1(0, 0.1, 0, 0.1) 0\nDETECTOR rec[-1]\nMY 0\nDETECTOR rec[-1]\n"
    )
    assert y.heralded_error_model().outcomes(0) == [(1.0, [1], 0)]


def test_erasure_noise_in_memory_experiments() -> None:
    n = q.ErasureNoise(0.01, 0.5)
    assert n == q.ErasureNoise(q.CircuitNoise.uniform(0.01), erasure_share=0.5)
    assert n != q.ErasureNoise(0.01, 0.4)
    assert n.noise == q.CircuitNoise.uniform(0.01) and n.erasure_share == 0.5
    assert "erasure_share=0.5" in repr(n)
    zero = q.ErasureNoise(0.01, 0.0)
    code = q.CssCode.color_code(3)
    pairs = [
        (q.repetition_code_memory(3, 3, zero), q.repetition_code_memory(3, 3, 0.01)),
        (q.surface_code_memory(3, 3, zero, "x"), q.surface_code_memory(3, 3, 0.01, "x")),
        (q.css_code_memory(code, 3, zero), q.css_code_memory(code, 3, 0.01)),
    ]
    for a, b in pairs:
        assert a.to_stim() == b.to_stim()
    rpairs = [
        (q.repetition_code_memory_repeated(3, 3, zero), q.repetition_code_memory_repeated(3, 3, 0.01)),
        (q.surface_code_memory_repeated(3, 3, zero), q.surface_code_memory_repeated(3, 3, 0.01)),
        (q.css_code_memory_repeated(code, 3, zero), q.css_code_memory_repeated(code, 3, 0.01)),
    ]
    for ra, rb in rpairs:
        assert ra.to_stim() == rb.to_stim()
    # Repetition d = 3, r = 3 at share ½: 53 herald detectors and 8 others.
    c = q.repetition_code_memory(3, 3, n)
    assert (c.n_detectors, c.n_measurements) == (61, 62)
    assert len(set(c.detector_layers())) == 4
    assert len(c.heralded_error_model().herald_detectors) == 53
    rep = q.repetition_code_memory_repeated(3, 3, n)
    assert rep.flatten().to_stim() == c.to_stim()
    assert rep.heralded_error_model().herald_detectors == c.heralded_error_model().herald_detectors
    s = q.surface_code_memory(3, 3, n)
    assert (s.n_detectors, s.n_measurements) == (285, 302)
    assert q.surface_code_memory_repeated(3, 3, n).flatten().to_stim() == s.to_stim()
    assert q.css_code_memory_repeated(code, 2, n).flatten().to_stim() == q.css_code_memory(code, 2, n).to_stim()
    # Share 1: no unheralded noise is left.
    full = q.surface_code_memory(3, 2, q.ErasureNoise(0.01, 1.0)).to_stim()
    assert "DEPOLARIZE" not in full and "X_ERROR" not in full and "M(" not in full
    for share in [-0.1, 1.5, math.nan]:
        with pytest.raises(ValueError, match="share"):
            q.ErasureNoise(0.01, share)
    with pytest.raises(TypeError):
        q.ErasureNoise(n, 0.5)  # type: ignore[arg-type]
