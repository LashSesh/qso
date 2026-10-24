import numpy as np
import pytest


def kron_all(*ops):
    """Kronecker product with the first factor most significant."""
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


def assert_same_up_to_phase(a, b, tol=1e-10):
    """Assert |<a|b>| = 1 for normalised vectors a and b."""
    overlap = abs(np.vdot(a, b))
    assert overlap == pytest.approx(1.0, abs=tol)


I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0 + 0j, -1.0])
