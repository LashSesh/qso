"""Circuits with repeated blocks and parallel windows (1.61)."""

import numpy as np
import pytest

import metatron_qso as q


def _models_equal(a: q.DetectorErrorModel, b: q.DetectorErrorModel) -> None:
    assert a.n_detectors == b.n_detectors
    ea, eb = a.errors, b.errors
    assert len(ea) == len(eb)
    for (pa, da, oa), (pb, db, ob) in zip(ea, eb):
        assert (da, oa) == (db, ob)
        assert pa == pytest.approx(pb, rel=1e-12)


def test_repeated_memory_experiments_expand_to_the_flat_ones() -> None:
    r = q.surface_code_memory_repeated(3, 30, 1e-3)
    flat = q.surface_code_memory(3, 30, 1e-3)
    assert r.n_detectors == flat.n_detectors
    assert r.n_operations == len(flat)
    assert r.n_stored_operations < len(flat)
    assert r.flatten().to_stim() == flat.to_stim()
    assert r.detector_layers() == flat.detector_layers()
    for decompose in (False, True):
        _models_equal(r.detector_error_model(decompose), flat.detector_error_model(decompose))
    det, obs = r.sample(200, seed=5)
    fdet, fobs = flat.sample(200, seed=5)
    assert np.array_equal(det, fdet) and np.array_equal(obs, fobs)
    rep = q.repetition_code_memory_repeated(5, 12, 2e-3)
    assert rep.flatten().to_stim() == q.repetition_code_memory(5, 12, 2e-3).to_stim()
    code, _ = q.CssCode.bivariate_bicycle_code(72)
    bb = q.css_code_memory_repeated(code, 3, 2e-3, basis="x", schedule="bicycle")
    assert bb.flatten().to_stim() == q.css_code_memory(code, 3, 2e-3, basis="x", schedule="bicycle").to_stim()


def test_stim_blocks_are_kept() -> None:
    text = "R 0 1 2\nREPEAT 50 {\n    X_ERROR(0.01) 0 1\n    CX 0 2 1 2\n    MR(0.002) 2\n    DETECTOR rec[-1]\n}\nM 0 1\nDETECTOR rec[-1] rec[-2] rec[-3]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
    r = q.RepeatedCircuit.from_stim(text)
    assert "REPEAT 50 {" in r.to_stim()
    assert q.RepeatedCircuit.from_stim(r.to_stim()).to_stim() == r.to_stim()
    assert r.flatten().to_stim() == q.DetectorCircuit.from_stim(text).to_stim()
    assert q.RepeatedCircuit.from_circuit(r.flatten()).n_detectors == r.n_detectors
    with pytest.raises(ValueError):
        q.RepeatedCircuit.from_stim("REPEAT 0 {\nM 0\n}\n")
    assert "RepeatedCircuit(" in repr(r)


def test_a_hundred_thousand_rounds() -> None:
    r = q.surface_code_memory_repeated(3, 100_000, 1e-3)
    m = r.detector_error_model(decompose=True)
    assert m.n_detectors == 4 * 100_001
    # A million rounds no longer fit a flat circuit.
    with pytest.raises(ValueError):
        q.surface_code_memory_repeated(3, 1_000_000, 1e-3).flatten()


def test_parallel_windows() -> None:
    c = q.surface_code_memory(3, 30, 6e-3)
    m = c.detector_error_model(decompose=True)
    layers = c.detector_layers()
    w = q.WindowDecoder.parallel(m, layers, commit=3, buffer=2)
    assert "parallel" in repr(w)
    whole = q.WindowDecoder.parallel(m, layers, commit=40)
    assert whole.n_windows == 1
    det, _ = c.sample(100, seed=2)
    for row in det:
        fired = [int(d) for d in np.flatnonzero(row)]
        batch = w.decode(fired)
        for layer in range(w.n_layers):
            w.push_layer([d for d in fired if layers[d] == layer])
        assert w.finish() == batch
    r = q.surface_code_memory_repeated(3, 30, 6e-3)
    a = q.logical_error_rate(c, 2000, seed=4, decoder="parallel_window_matching", commit=3, buffer=2)
    b = q.logical_error_rate(r, 2000, seed=4, decoder="parallel_window_matching", commit=3, buffer=2)
    assert a == b and a.errors > 0
    with pytest.raises(ValueError):
        q.logical_error_rate(c, 10, decoder="parallel_window_matching", window=4)
    with pytest.raises(ValueError):
        q.logical_error_rate(c, 10, decoder="window_matching", window=4, buffer=2)
    with pytest.raises(ValueError):
        q.logical_error_rate(c, 10, decoder="parallel_window_matching")
