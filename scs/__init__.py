"""
Seraphic Calibration Shell (SCS) for Q⊗DASH (Metatron VM)

A meta-model for fixpoint-directed quantum-hybrid optimization that wraps
the Q⊗DASH core with field-theoretic feedback and contraction dynamics.

The SCS enforces monotonic convergence towards fixpoint attractors in
configuration space through:
- Performance triplet Φ(c) = (ψ, ρ, ω) tracking
- Seraphic feedback and Mandorla-like calibration field M(t)
- Double-kick operator T = Φ_V ∘ Φ_U with local contraction
- Proof-of-Resonance (PoR) acceptance criterion
- CRI-style resonance impulses for regime switching
"""

__version__ = "0.1.0"
__author__ = "Sebastian Klemm"

from .config import Configuration, ConfigurationSpace
from .performance import PerformanceTriplet, compute_performance_triplet, BenchmarkLoader
from .field import MandorlaField, SeraphicFeedback
from .operators import DoubleKickOperator, UpdateKick, StabilizationKick
from .por import ProofOfResonance, PoRCriteria
from .cri import ResonanceImpulse, ResonanceImpulseConfig, GlobalCalibrationState
from .calibrator import SeraphicCalibrator, CalibratorConfig, CalibrationHistory
from .benchmark import (
    BenchmarkRecord,
    BenchmarkValidationError,
    load_benchmark,
    load_benchmark_batch,
    load_benchmarks,
    write_benchmark,
    validate_benchmark,
    generate_config_id,
    filter_benchmarks,
    aggregate_benchmarks,
)
from .core import (
    AutoTuner,
    NewConfigProposal,
    create_auto_tuner,
    quick_tune,
    load_and_compute_performance,
)

__all__ = [
    # Config
    "Configuration",
    "ConfigurationSpace",
    # Performance
    "PerformanceTriplet",
    "compute_performance_triplet",
    "BenchmarkLoader",
    # Field
    "MandorlaField",
    "SeraphicFeedback",
    # Operators
    "DoubleKickOperator",
    "UpdateKick",
    "StabilizationKick",
    # PoR
    "ProofOfResonance",
    "PoRCriteria",
    # CRI
    "ResonanceImpulse",
    "ResonanceImpulseConfig",
    "GlobalCalibrationState",
    # Calibrator
    "SeraphicCalibrator",
    "CalibratorConfig",
    "CalibrationHistory",
    # Benchmark
    "BenchmarkRecord",
    "BenchmarkValidationError",
    "load_benchmark",
    "load_benchmark_batch",
    "load_benchmarks",
    "write_benchmark",
    "validate_benchmark",
    "generate_config_id",
    "filter_benchmarks",
    "aggregate_benchmarks",
    # Auto-Tuner API
    "AutoTuner",
    "NewConfigProposal",
    "create_auto_tuner",
    "quick_tune",
    "load_and_compute_performance",
]
