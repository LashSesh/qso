"""Decoding in sliding windows (1.57)."""

import numpy as np
import pytest

import metatron_qso as q


def test_windows_over_everything_equal_the_whole_decoding() -> None:
    c = q.surface_code_memory(3, 4, 5e-3)
    model = c.detector_error_model(decompose=True)
    layers = c.detector_layers()
    assert layers == sorted(layers) and layers[-1] == 4
    whole = q.SparseMatchingDecoder(model)
    w = q.WindowDecoder(model, layers, window=6)
    assert w.n_windows == 1
    detectors, _ = c.sample(200, seed=3)
    assert np.array_equal(w.decode_batch(detectors), whole.decode_batch(detectors))


def test_streaming_equals_the_batch() -> None:
    c = q.surface_code_memory(3, 10, 6e-3)
    model = c.detector_error_model(decompose=True)
    layers = c.detector_layers()
    w = q.WindowDecoder(model, layers, window=4, commit=2)
    assert w.n_windows > 3 and w.n_decoders <= w.n_windows
    detectors, _ = c.sample(50, seed=1)
    batch = w.decode_batch(detectors)
    for row, expected in zip(detectors, batch):
        fired = np.flatnonzero(row)
        for layer in range(w.n_layers):
            w.push_layer([int(d) for d in fired if layers[d] == layer])
        assert w.finish() == expected
    with pytest.raises(ValueError):
        w.push_layer([10**6])
    with pytest.raises(ValueError):
        w.finish()


def test_logical_error_rate_in_windows() -> None:
    c = q.surface_code_memory(3, 12, 5e-3)
    whole = q.logical_error_rate(c, 2000, seed=4, decoder="matching")
    windowed = q.logical_error_rate(c, 2000, seed=4, decoder="window_matching", window=6)
    assert abs(windowed.errors - whole.errors) <= 3 * whole.errors**0.5 + 3
    with pytest.raises(ValueError):
        q.logical_error_rate(c, 10, decoder="window_matching")
    with pytest.raises(ValueError):
        q.logical_error_rate(c, 10, decoder="matching", window=4)
    with pytest.raises(ValueError):
        q.WindowDecoder(c.detector_error_model(decompose=True), c.detector_layers(), 2, decoder="mwpm")  # type: ignore[arg-type]
