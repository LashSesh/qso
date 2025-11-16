"""
Performance triplet Φ(c) = (ψ, ρ, ω) computation from benchmark results.

Given a benchmark outcome vector u(c) for configuration c, computes:
- ψ(c): semantic quality (how well it performs)
- ρ(c): stability/path invariance (robustness)
- ω(c): phase readiness/efficiency (resource usage)

All components are normalized to [0, 1].
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json
import numpy as np
from pathlib import Path


@dataclass
class PerformanceTriplet:
    """
    The performance triplet Φ(c) = (ψ, ρ, ω) for a configuration.
    """

    psi: float  # ψ: semantic quality [0, 1]
    rho: float  # ρ: stability/path invariance [0, 1]
    omega: float  # ω: phase readiness/efficiency [0, 1]

    def __post_init__(self):
        """Validate that all components are in [0, 1]."""
        for name, val in [('ψ', self.psi), ('ρ', self.rho), ('ω', self.omega)]:
            if not 0 <= val <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {val}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {'psi': self.psi, 'rho': self.rho, 'omega': self.omega}

    def norm(self) -> float:
        """Compute Euclidean norm of the triplet."""
        return np.sqrt(self.psi**2 + self.rho**2 + self.omega**2)

    def harmonic_mean(self) -> float:
        """Compute harmonic mean of components (overall quality indicator)."""
        if self.psi == 0 or self.rho == 0 or self.omega == 0:
            return 0.0
        return 3.0 / (1.0/self.psi + 1.0/self.rho + 1.0/self.omega)

    def geometric_mean(self) -> float:
        """Compute geometric mean of components."""
        return (self.psi * self.rho * self.omega) ** (1.0/3.0)


class BenchmarkLoader:
    """
    Loads benchmark JSON files and extracts metrics.
    """

    def __init__(self, benchmark_dir: str = "metatron-qso-rs/ci"):
        """Initialize with path to benchmark directory."""
        self.benchmark_dir = Path(benchmark_dir)

    def load_vqe_benchmark(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load VQE benchmark results."""
        if path is None:
            path = self.benchmark_dir / "vqe_baseline.json"
        with open(path, 'r') as f:
            return json.load(f)

    def load_qaoa_benchmark(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load QAOA benchmark results."""
        if path is None:
            path = self.benchmark_dir / "qaoa_baseline.json"
        with open(path, 'r') as f:
            return json.load(f)

    def load_quantum_walk_benchmark(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load Quantum Walk benchmark results."""
        if path is None:
            path = self.benchmark_dir / "quantum_walk_baseline.json"
        with open(path, 'r') as f:
            return json.load(f)

    def load_advanced_algorithms_benchmark(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load advanced algorithms (Grover, Boson, QML) benchmark results."""
        if path is None:
            path = self.benchmark_dir / "advanced_algorithms_baseline.json"
        with open(path, 'r') as f:
            return json.load(f)

    def load_vqc_benchmark(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load VQC/QML benchmark results."""
        if path is None:
            path = self.benchmark_dir / "vqc_baseline.json"
        with open(path, 'r') as f:
            return json.load(f)

    def load_cross_system_benchmark(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load cross-system comparison benchmark."""
        if path is None:
            path = self.benchmark_dir / "cross_system_baseline.json"
        with open(path, 'r') as f:
            return json.load(f)

    def load_integration_benchmark(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Load integration benchmark results."""
        if path is None:
            path = self.benchmark_dir / "integration_baseline.json"
        with open(path, 'r') as f:
            return json.load(f)

    def load_all_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Load all available benchmark results."""
        benchmarks = {}
        loaders = {
            'vqe': self.load_vqe_benchmark,
            'qaoa': self.load_qaoa_benchmark,
            'quantum_walk': self.load_quantum_walk_benchmark,
            'advanced': self.load_advanced_algorithms_benchmark,
            'vqc': self.load_vqc_benchmark,
            'cross_system': self.load_cross_system_benchmark,
            'integration': self.load_integration_benchmark,
        }

        for name, loader in loaders.items():
            try:
                benchmarks[name] = loader()
            except FileNotFoundError:
                pass  # Skip missing benchmarks

        return benchmarks


def compute_vqe_quality(data: Dict[str, Any]) -> float:
    """
    Compute quality score ψ for VQE benchmarks.

    Uses the quality_score from the best result, or computes from
    approximation error.
    """
    if 'quality_metrics' in data:
        # Use aggregate metrics if available
        qm = data['quality_metrics']
        if 'best_ground_energy' in qm and 'avg_ground_energy' in qm:
            # Use convergence rate as proxy
            return qm.get('convergence_rate', 0.8)

    if 'results' in data and len(data['results']) > 0:
        # Use best quality score from results
        quality_scores = [r.get('quality_score', 0.0) for r in data['results']]
        return max(quality_scores)

    return 0.5  # Default


def compute_vqe_stability(data: Dict[str, Any]) -> float:
    """
    Compute stability ρ for VQE benchmarks.

    Based on variance of quality scores across different configurations.
    """
    if 'results' in data and len(data['results']) > 1:
        quality_scores = [r.get('quality_score', 0.0) for r in data['results']]
        variance = np.var(quality_scores)
        # Low variance → high stability
        # Map variance [0, 0.1] → stability [1, 0]
        stability = max(0.0, 1.0 - variance * 10.0)
        return min(1.0, stability)

    return 0.5  # Default for single result


def compute_vqe_efficiency(data: Dict[str, Any]) -> float:
    """
    Compute efficiency ω for VQE benchmarks.

    Based on evaluations per second and execution time.
    """
    if 'performance_metrics' in data:
        pm = data['performance_metrics']
        if 'evaluations_per_second' in pm:
            eps = pm['evaluations_per_second']
            # Normalize: 10000 eps = 1.0, 1000 eps = 0.1
            efficiency = min(1.0, eps / 10000.0)
            return efficiency

    return 0.5  # Default


def compute_qaoa_quality(data: Dict[str, Any]) -> float:
    """Compute quality score ψ for QAOA benchmarks."""
    if 'quality_metrics' in data:
        qm = data['quality_metrics']
        return qm.get('avg_approximation_ratio', 0.8)

    # Compute from individual problems
    ratios = []
    for key, val in data.items():
        if isinstance(val, dict) and 'approximation_ratio' in val:
            ratios.append(val['approximation_ratio'])

    return np.mean(ratios) if ratios else 0.5


def compute_qaoa_stability(data: Dict[str, Any]) -> float:
    """Compute stability ρ for QAOA benchmarks."""
    if 'quality_metrics' in data:
        qm = data['quality_metrics']
        variance = qm.get('ratio_variance', 0.0)
        # Low variance → high stability
        stability = max(0.0, 1.0 - variance * 10.0)
        return min(1.0, stability)

    return 0.8  # Default


def compute_qaoa_efficiency(data: Dict[str, Any]) -> float:
    """Compute efficiency ω for QAOA benchmarks."""
    if 'performance_metrics' in data:
        pm = data['performance_metrics']
        if 'evaluations_per_second' in pm:
            eps = pm['evaluations_per_second']
            efficiency = min(1.0, eps / 10000.0)
            return efficiency

    return 0.5  # Default


def compute_cross_system_quality(data: Dict[str, Any]) -> float:
    """Compute quality from cross-system comparison."""
    if 'metatron_qso' in data:
        return data['metatron_qso'].get('overall_score', 0.5)
    return 0.5


def compute_cross_system_stability(data: Dict[str, Any]) -> float:
    """Compute stability from cross-system comparison."""
    # High overall score indicates stable performance
    if 'metatron_qso' in data:
        mqso = data['metatron_qso']
        vqe_quality = mqso.get('vqe_performance', {}).get('quality_score', 0.0)
        qaoa_quality = mqso.get('qaoa_performance', {}).get('quality_score', 0.0)
        # Stability is consistency across different algorithms
        variance = ((vqe_quality + qaoa_quality) / 2.0 - min(vqe_quality, qaoa_quality))
        return max(0.0, 1.0 - variance)
    return 0.5


def compute_cross_system_efficiency(data: Dict[str, Any]) -> float:
    """Compute efficiency from cross-system comparison."""
    if 'metatron_qso' in data:
        mqso = data['metatron_qso']
        # Average of speed scores
        vqe_speed = mqso.get('vqe_performance', {}).get('speed_score', 0.0)
        qaoa_speed = mqso.get('qaoa_performance', {}).get('speed_score', 0.0)
        return (vqe_speed + qaoa_speed) / 2.0
    return 0.5


def compute_performance_triplet(
    benchmarks: Dict[str, Dict[str, Any]],
    algorithm_weights: Optional[Dict[str, float]] = None
) -> PerformanceTriplet:
    """
    Compute the performance triplet Φ(c) = (ψ, ρ, ω) from benchmark results.

    Args:
        benchmarks: Dictionary of benchmark results keyed by algorithm name
        algorithm_weights: Optional weights for each algorithm (defaults to equal)

    Returns:
        PerformanceTriplet with computed (ψ, ρ, ω)
    """
    if algorithm_weights is None:
        algorithm_weights = {
            'vqe': 1.0,
            'qaoa': 1.0,
            'quantum_walk': 0.5,
            'advanced': 0.5,
            'vqc': 0.5,
            'cross_system': 1.5,
            'integration': 0.5,
        }

    psi_values = []
    rho_values = []
    omega_values = []
    weights = []

    # Process each benchmark
    for name, data in benchmarks.items():
        weight = algorithm_weights.get(name, 0.5)
        if weight <= 0:
            continue

        if name == 'vqe':
            psi_values.append(compute_vqe_quality(data))
            rho_values.append(compute_vqe_stability(data))
            omega_values.append(compute_vqe_efficiency(data))
            weights.append(weight)
        elif name == 'qaoa':
            psi_values.append(compute_qaoa_quality(data))
            rho_values.append(compute_qaoa_stability(data))
            omega_values.append(compute_qaoa_efficiency(data))
            weights.append(weight)
        elif name == 'cross_system':
            psi_values.append(compute_cross_system_quality(data))
            rho_values.append(compute_cross_system_stability(data))
            omega_values.append(compute_cross_system_efficiency(data))
            weights.append(weight)
        # Add more algorithm handlers as needed

    # Compute weighted averages
    if not psi_values:
        return PerformanceTriplet(psi=0.5, rho=0.5, omega=0.5)

    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    psi = float(np.average(psi_values, weights=weights))
    rho = float(np.average(rho_values, weights=weights))
    omega = float(np.average(omega_values, weights=weights))

    # Ensure all values are in [0, 1]
    psi = max(0.0, min(1.0, psi))
    rho = max(0.0, min(1.0, rho))
    omega = max(0.0, min(1.0, omega))

    return PerformanceTriplet(psi=psi, rho=rho, omega=omega)
