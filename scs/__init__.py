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

__all__ = [
    "Configuration",
    "ConfigurationSpace",
    "PerformanceTriplet",
    "compute_performance_triplet",
    "BenchmarkLoader",
    "MandorlaField",
    "SeraphicFeedback",
    "DoubleKickOperator",
    "UpdateKick",
    "StabilizationKick",
    "ProofOfResonance",
    "PoRCriteria",
    "ResonanceImpulse",
    "ResonanceImpulseConfig",
    "GlobalCalibrationState",
    "SeraphicCalibrator",
    "CalibratorConfig",
    "CalibrationHistory",
]
