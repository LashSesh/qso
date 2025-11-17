"""
Auto-tuning integration for Metatron QSO with SCS.

Provides wrapper functions that integrate SCS auto-tuning with
Quantum Walk and QAOA algorithms.
"""

from typing import Dict, Any, Optional, Tuple, List
import sys
from pathlib import Path

# Import core functions from Rust bindings
from ._metatron_qso_internal import (
    run_quantum_walk,
    solve_maxcut_qaoa,
    run_vqe,
)

# Try to import SCS (optional dependency)
try:
    # Add scs directory to path if needed
    scs_path = Path(__file__).parent.parent.parent.parent / "scs"
    if scs_path.exists() and str(scs_path.parent) not in sys.path:
        sys.path.insert(0, str(scs_path.parent))

    from scs import (
        AutoTuner,
        Configuration,
        write_benchmark,
        NewConfigProposal,
    )

    SCS_AVAILABLE = True
except ImportError:
    SCS_AVAILABLE = False
    AutoTuner = None
    Configuration = None
    write_benchmark = None
    NewConfigProposal = None


def run_quantum_walk_with_tuning(
    graph: Any,
    source_nodes: List[int],
    t_max: float = 5.0,
    dt: float = 0.1,
    auto_tune: bool = False,
    benchmark_dir: str = "benchmarks",
) -> Tuple[Dict[str, Any], Optional[Any]]:
    """
    Run quantum walk with optional SCS auto-tuning.

    Args:
        graph: MetatronGraph instance
        source_nodes: List of source node indices
        t_max: Maximum evolution time
        dt: Time step
        auto_tune: Enable SCS auto-tuning
        benchmark_dir: Directory for benchmark files

    Returns:
        Tuple of (result_dict, config_proposal or None)

    Example:
        >>> graph = MetatronGraph()
        >>> result, proposal = run_quantum_walk_with_tuning(
        ...     graph, [0], t_max=5.0, dt=0.1, auto_tune=True
        ... )
        >>> if proposal:
        ...     print(f"SCS suggests: {proposal.config.to_dict()}")
    """
    if auto_tune and not SCS_AVAILABLE:
        print("Warning: SCS not available. Running without auto-tuning.")
        auto_tune = False

    # Run quantum walk
    result = run_quantum_walk(graph, source_nodes, t_max, dt)

    # If auto-tuning is disabled, return early
    if not auto_tune:
        return result, None

    # Compute performance metrics from result
    metrics = _compute_qw_metrics(result, t_max, dt)

    # Create configuration
    config = {
        "algorithm": "QuantumWalk",
        "walk_type": "CTQW",
        "t_max": t_max,
        "dt": dt,
        "source_nodes": source_nodes,
    }

    # Create auto-tuner and ingest benchmark
    tuner = AutoTuner(benchmark_dir=benchmark_dir, enabled=True)
    tuner.initialize()

    # Write benchmark
    tuner.ingest_benchmark(
        system="quantum_walk",
        config=config,
        metrics=metrics,
        raw_results=result,
    )

    # Get config proposal
    proposal = tuner.propose_new_config()

    return result, proposal


def solve_maxcut_qaoa_with_tuning(
    graph: Any,
    depth: int = 3,
    max_iters: int = 100,
    auto_calibrate: bool = False,
    benchmark_dir: str = "benchmarks",
    optimizer: str = "COBYLA",
) -> Tuple[Dict[str, Any], Optional[Any]]:
    """
    Solve MaxCut with QAOA and optional SCS auto-calibration.

    Args:
        graph: MetatronGraph instance
        depth: QAOA depth (p parameter)
        max_iters: Maximum optimization iterations
        auto_calibrate: Enable SCS auto-calibration
        benchmark_dir: Directory for benchmark files
        optimizer: Optimizer name ("COBYLA", "Adam", etc.)

    Returns:
        Tuple of (result_dict, config_proposal or None)

    Example:
        >>> graph = MetatronGraph()
        >>> result, proposal = solve_maxcut_qaoa_with_tuning(
        ...     graph, depth=3, max_iters=100, auto_calibrate=True
        ... )
        >>> if proposal and proposal.por_accepted:
        ...     new_depth = proposal.config.ansatz_depth
        ...     # Run again with new depth
    """
    if auto_calibrate and not SCS_AVAILABLE:
        print("Warning: SCS not available. Running without auto-calibration.")
        auto_calibrate = False

    # Run QAOA
    result = solve_maxcut_qaoa(graph, depth, max_iters)

    # If auto-calibration is disabled, return early
    if not auto_calibrate:
        return result, None

    # Compute performance metrics
    metrics = _compute_qaoa_metrics(result)

    # Create configuration
    config = {
        "algorithm": "QAOA",
        "problem": "maxcut",
        "depth": depth,
        "optimizer": optimizer,
        "max_iterations": max_iters,
    }

    # Create auto-tuner
    tuner = AutoTuner(benchmark_dir=benchmark_dir, enabled=True)
    tuner.initialize()

    # Write benchmark
    tuner.ingest_benchmark(
        system="qaoa_maxcut",
        config=config,
        metrics=metrics,
        raw_results=result,
    )

    # Get config proposal
    proposal = tuner.propose_new_config()

    return result, proposal


def run_vqe_with_tuning(
    graph: Any,
    ansatz_type: str = "Metatron",
    depth: int = 2,
    max_iters: int = 150,
    optimizer: str = "Adam",
    auto_tune: bool = False,
    benchmark_dir: str = "benchmarks",
) -> Tuple[Dict[str, Any], Optional[Any]]:
    """
    Run VQE with optional SCS auto-tuning.

    Args:
        graph: MetatronGraph instance
        ansatz_type: Ansatz type ("Metatron", "EfficientSU2", "HardwareEfficient")
        depth: Ansatz depth
        max_iters: Maximum optimization iterations
        optimizer: Optimizer name
        auto_tune: Enable SCS auto-tuning
        benchmark_dir: Directory for benchmark files

    Returns:
        Tuple of (result_dict, config_proposal or None)
    """
    if auto_tune and not SCS_AVAILABLE:
        print("Warning: SCS not available. Running without auto-tuning.")
        auto_tune = False

    # Run VQE
    result = run_vqe(graph, depth, max_iters, ansatz_type)

    # If auto-tuning is disabled, return early
    if not auto_tune:
        return result, None

    # Compute performance metrics
    metrics = _compute_vqe_metrics(result)

    # Create configuration
    config = {
        "algorithm": "VQE",
        "ansatz_type": ansatz_type,
        "ansatz_depth": depth,
        "optimizer": optimizer,
        "max_iterations": max_iters,
    }

    # Create auto-tuner
    tuner = AutoTuner(benchmark_dir=benchmark_dir, enabled=True)
    tuner.initialize()

    # Write benchmark
    tuner.ingest_benchmark(
        system="vqe",
        config=config,
        metrics=metrics,
        raw_results=result,
    )

    # Get config proposal
    proposal = tuner.propose_new_config()

    return result, proposal


def _compute_qw_metrics(
    result: Dict[str, Any], t_max: float, dt: float
) -> Dict[str, float]:
    """
    Compute performance metrics (ψ, ρ, ω) from quantum walk result.

    Args:
        result: Quantum walk result dictionary
        t_max: Evolution time
        dt: Time step

    Returns:
        Metrics dictionary {psi, rho, omega}
    """
    # Quality (ψ): Spreading quality based on entropy
    final_state = result.get("final_state", [])
    if final_state:
        import math

        # Compute entropy
        entropy = -sum(p * math.log(p) if p > 1e-10 else 0 for p in final_state)
        max_entropy = math.log(len(final_state))
        psi = min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.5
    else:
        psi = 0.5

    # Stability (ρ): Assume high stability for quantum walk
    rho = 0.90

    # Efficiency (ω): Based on computation speed
    wallclock = result.get("meta", {}).get("wallclock_time_ms", 1000)
    expected_time = 200  # ms for reference
    omega = min(1.0, expected_time / max(wallclock, 1))

    return {"psi": psi, "rho": rho, "omega": omega}


def _compute_qaoa_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute performance metrics (ψ, ρ, ω) from QAOA result.

    Args:
        result: QAOA result dictionary

    Returns:
        Metrics dictionary {psi, rho, omega}
    """
    # Quality (ψ): Approximation ratio
    psi = result.get("approximation_ratio", 0.8)

    # Stability (ρ): Assume good stability if converged
    meta = result.get("meta", {})
    iterations = meta.get("iterations", 100)
    max_iters = meta.get("max_iterations", 100)

    if iterations < max_iters:
        rho = 0.85  # Converged early → stable
    else:
        rho = 0.70  # Used all iterations → less stable

    # Efficiency (ω): Based on iteration count
    omega = 1.0 - min(1.0, iterations / max_iters)
    omega = max(0.3, omega)  # Floor at 0.3

    return {"psi": psi, "rho": rho, "omega": omega}


def _compute_vqe_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute performance metrics (ψ, ρ, ω) from VQE result.

    Args:
        result: VQE result dictionary

    Returns:
        Metrics dictionary {psi, rho, omega}
    """
    # Quality (ψ): Based on quality score
    psi = result.get("quality_score", 0.8)

    # Stability (ρ): Based on convergence
    meta = result.get("meta", {})
    converged = meta.get("converged", False)
    rho = 0.85 if converged else 0.65

    # Efficiency (ω): Based on iteration count
    iterations = meta.get("iterations", 100)
    max_iters = meta.get("max_iterations", 150)
    omega = 1.0 - min(1.0, iterations / max_iters)
    omega = max(0.3, omega)

    return {"psi": psi, "rho": rho, "omega": omega}


# Export convenience functions
__all__ = [
    "run_quantum_walk_with_tuning",
    "solve_maxcut_qaoa_with_tuning",
    "run_vqe_with_tuning",
    "SCS_AVAILABLE",
]
