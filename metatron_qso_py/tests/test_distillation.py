import numpy as np
import pytest

import metatron_qso as q


def noisy_ghz():
    prep = q.Circuit(3).h(0).cx(0, 1).cx(1, 2).ry(1, 0.3)
    rho = q.DensityMatrix.simulate(prep, noise=q.NoiseModel.depolarizing(0.01, 0.04))
    return prep, rho


def test_distillation_suppresses_noise():
    prep, rho = noisy_ghz()
    o = q.PauliSum("Z0 Z1 + X0 X1 X2")
    ideal = o.expectation(q.simulate(prep))
    errors = [abs(q.distilled_expectation(rho, o, k) - ideal) for k in (1, 2, 3)]
    assert errors[1] < 0.2 * errors[0] and errors[2] < errors[1]
    assert abs(q.distilled_expectation(rho, o, 1) - rho.expectation(o)) < 1e-12
    assert errors[2] >= abs(q.dominant_expectation(rho, o) - ideal) - 1e-12


def test_circuits_estimate_the_distilled_value():
    prep, rho = noisy_ghz()
    o = q.PauliSum("Z0 Z1 + 0.5 X0 X1 X2 - 0.3 Y2 + 0.1")
    vd = q.DistillationCircuits(prep, o, copies=2)
    assert vd.copies == 2 and len(vd) == 4
    assert all(c.n_qubits == 6 for c in vd.circuits)
    exact = q.distilled_expectation(rho, o, 2)
    assert abs(vd.evaluate_on(rho) - exact) < 1e-11
    # Noiseless circuits on a pure preparation give the plain value.
    probs = [q.simulate(c).probabilities() for c in vd.circuits]
    pure = o.expectation(q.simulate(prep))
    assert abs(vd.evaluate(probs) - pure) < 1e-11
    # Shots.
    counts = [q.simulate(c).sample(20_000, seed=s) for s, c in enumerate(vd.circuits)]
    e = vd.estimate(counts)
    assert abs(e.value - pure) < 5 * e.std_error + 1e-12
    with pytest.raises(ValueError):
        q.DistillationCircuits(prep, o, copies=4)
    with pytest.raises(ValueError):
        vd.evaluate(probs[:2])
    with pytest.raises(ValueError):
        q.dominant_expectation(q.DensityMatrix.maximally_mixed(2), q.PauliSum("Z0"))
