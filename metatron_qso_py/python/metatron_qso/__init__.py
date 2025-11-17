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
    # High-level toolkits
    quantum_walk_centrality,
    quantum_walk_anomaly_score,
    quantum_walk_connectivity,
    solve_maxcut_qaoa_advanced,
    __version__,
)

# Import auto-tuning integration (optional SCS support)
from .auto_tuning import (
    run_quantum_walk_with_tuning,
    solve_maxcut_qaoa_with_tuning,
    run_vqe_with_tuning,
    SCS_AVAILABLE,
)

__all__ = [
    "MetatronGraph",
    # Core functions
    "run_quantum_walk",
    "solve_maxcut_qaoa",
    "run_vqe",
    # Quantum Walk Toolkit
    "quantum_walk_centrality",
    "quantum_walk_anomaly_score",
    "quantum_walk_connectivity",
    # QAOA Optimizer
    "solve_maxcut_qaoa_advanced",
    # Auto-Tuning Integration (SCS)
    "run_quantum_walk_with_tuning",
    "solve_maxcut_qaoa_with_tuning",
    "run_vqe_with_tuning",
    "SCS_AVAILABLE",
    "__version__",
]
