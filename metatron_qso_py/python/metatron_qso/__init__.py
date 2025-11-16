"""
Metatron Quantum State Operator - Python SDK

High-performance quantum computing framework powered by Rust.

This package provides Python bindings to the Metatron QSO quantum computing
framework, enabling quantum walks, variational quantum algorithms (VQE, QAOA),
and more on the sacred geometry of the Metatron Cube.

Example:
    >>> import metatron_qso
    >>> graph = metatron_qso.MetatronGraph()
    >>> result = metatron_qso.run_quantum_walk(graph, [0], t_max=5.0, dt=0.1)
    >>> print(result['final_state'])
"""

# Import from the internal Rust module
from ._metatron_qso_internal import (
    MetatronGraph,
    run_quantum_walk,
    solve_maxcut_qaoa,
    run_vqe,
    __version__,
    __doc__ as _internal_doc,
)

__all__ = [
    "MetatronGraph",
    "run_quantum_walk",
    "solve_maxcut_qaoa",
    "run_vqe",
    "__version__",
]
