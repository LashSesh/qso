"""Circuit distance, CNOT schedules and correlated (two-pass) matching."""

from typing import Literal, Optional

import numpy as np
import pytest

import metatron_qso as q


def test_circuit_distance_of_memory_experiments() -> None:
    for d in [3, 5]:
        c = q.surface_code_memory(d, d, 0.001)
        whole = q.circuit_distance(c.detector_error_model())
        assert (whole.lower_bound, whole.upper_bound, whole.distance) == (d, d, d)
        graph = q.graphlike_distance(c.detector_error_model(decompose=True))
        assert graph.distance == d
        # The witness flips observables and no detector.
        model = c.detector_error_model()
        fired = np.zeros(model.n_detectors, dtype=bool)
        obs = 0
        assert whole.witness is not None
        for k in whole.witness:
            _, dets, o = model.errors[k]
            fired[dets] ^= True
            obs ^= o
        assert not fired.any() and obs == whole.observables != 0
        assert "distance=" in repr(whole)


def test_schedules_of_css_codes() -> None:
    code, _ = q.CssCode.bivariate_bicycle_code(72)
    distances: dict[Optional[str], Optional[int]] = {}
    schedules: list[Optional[Literal["bicycle", "greedy"]]] = ["bicycle", "greedy", None]
    for schedule in schedules:
        c = q.css_code_memory(code, 3, 0.001, schedule=schedule)
        distances[schedule] = q.circuit_distance(c.detector_error_model()).distance
    assert distances == {"bicycle": 6, "greedy": 4, None: 4}
    colour = q.CssCode.color_code(5)
    c = q.css_code_memory(colour, 2, 0.001, schedule=[5, 4, 3, 2, 1, 0])
    assert q.circuit_distance(c.detector_error_model()).distance == 3
    with pytest.raises(ValueError):
        q.css_code_memory(colour, 2, 0.001, schedule="bicycle")
    with pytest.raises(ValueError):
        q.css_code_memory(code, 2, 0.001, schedule=[0, 1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        q.css_code_memory(colour, 2, 0.001, schedule=[0, 1, 2])
    with pytest.raises(ValueError):
        q.css_code_memory(colour, 2, 0.001, schedule=[0, 0, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        q.css_code_memory(colour, 2, 0.001, schedule="fastest")  # type: ignore[arg-type]


def test_limits_and_bounds() -> None:
    c = q.css_code_memory(q.CssCode.color_code(5), 2, 0.001)
    model = c.detector_error_model()
    loose = q.circuit_distance(model, max_weight=1, bposd_trials=1)
    assert loose.lower_bound == 2 and loose.upper_bound is not None
    assert loose.distance is None or loose.distance == 2
    none = q.circuit_distance(q.DetectorErrorModel(2, [(0.1, [0, 1], 0)]))
    assert (none.witness, none.upper_bound, none.distance) == (None, None, None)
    assert "inf" in repr(none)


def test_correlated_error_model() -> None:
    c = q.surface_code_memory(3, 3, 0.004)
    model = c.correlated_error_model()
    assert model.n_detectors == c.n_detectors and len(model) == len(model.errors)
    flat = model.to_dem()
    reference = c.detector_error_model(decompose=True)
    assert sorted((d, o) for _, d, o in flat.errors) == sorted((d, o) for _, d, o in reference.errors)
    assert all(1 <= len(d) <= 2 for _, parts in model.errors for d, _ in parts)
    assert "CorrelatedErrorModel" in repr(model)
    # By hand.
    hand = q.CorrelatedErrorModel(4, [(0.1, [([0, 1], 0), ([2, 3], 0)]), (0.02, [([1, 2], 1)])])
    assert hand.n_observables == 1
    with pytest.raises(ValueError):
        q.CorrelatedErrorModel(4, [(0.1, [([0, 1, 2], 0)])])
    with pytest.raises(ValueError):
        q.CorrelatedErrorModel(2, [(0.1, [([0, 5], 0)])])
    with pytest.raises(ValueError):
        q.CorrelatedErrorModel(2, [(1.5, [([0], 0)])])
    with pytest.raises(ValueError):
        q.CorrelatedErrorModel(2, [(0.1, [([0], 4)])], n_observables=1)


def test_correlated_matching_decoder() -> None:
    c = q.surface_code_memory(3, 3, 0.004)
    model = c.correlated_error_model()
    decoder = q.CorrelatedMatchingDecoder(model)
    assert decoder.n_detectors == c.n_detectors
    det, obs = c.sample(2000, seed=3)
    batch = decoder.decode_batch(det)
    for s in range(50):
        assert decoder.decode(list(np.flatnonzero(det[s]))) == batch[s]
    truth = (obs[:, 0]).astype(np.uint64)
    assert np.mean(batch != truth) < 0.1
    assert "CorrelatedMatchingDecoder" in repr(decoder)
    r = q.logical_error_rate(c, 2000, seed=1, decoder="correlated_matching")
    assert r.shots == 2000
    with pytest.raises(ValueError):
        q.CorrelatedMatchingDecoder(q.CorrelatedErrorModel(2, [(0.7, [([0, 1], 0)])]))
    with pytest.raises(ValueError):
        q.logical_error_rate(c, 10, decoder="two_pass")  # type: ignore[arg-type]
