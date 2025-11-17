"""
Mandorla-like calibration field M(t) and seraphic feedback encoder.

The field M(t) ∈ R^m accumulates traces of benchmark outcomes and guides
the evolution of configurations through resonance patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import json

from .performance import PerformanceTriplet


@dataclass
class MandorlaField:
    """
    The Mandorla-like calibration field M(t) ∈ R^m.

    This field accumulates feedback from benchmark results and creates
    a resonance landscape that influences configuration updates.
    """

    dimension: int = 16  # Dimension m of the field
    field_state: np.ndarray = field(default_factory=lambda: np.zeros(16))

    # Update coefficients (α, βᵢ, γ from Eq. 1)
    alpha: float = 0.95  # Memory decay factor
    gamma: float = 0.5  # Injection weight
    beta_weights: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])

    # Resonance submodule contributions G_i(t)
    submodule_states: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        """Initialize field state if needed."""
        if len(self.field_state) != self.dimension:
            self.field_state = np.zeros(self.dimension)
        if not self.submodule_states:
            self.submodule_states = [
                np.zeros(self.dimension) for _ in self.beta_weights
            ]

    def update(self, injection: np.ndarray) -> None:
        """
        Update the field according to Eq. (1):

        M(t+1) = Norm(α M(t) + Σᵢ βᵢ Gᵢ(t) + γ Iₜ)

        Args:
            injection: Feedback vector Iₜ from seraphic encoder
        """
        if len(injection) != self.dimension:
            raise ValueError(
                f"Injection dimension {len(injection)} != field dimension {self.dimension}"
            )

        # Combine: α M(t) + Σᵢ βᵢ Gᵢ(t) + γ Iₜ
        new_state = self.alpha * self.field_state

        # Add submodule contributions
        for beta, G_i in zip(self.beta_weights, self.submodule_states):
            new_state += beta * G_i

        # Add injection
        new_state += self.gamma * injection

        # Normalize to bounded domain (L2 norm = 1)
        norm = np.linalg.norm(new_state)
        if norm > 0:
            self.field_state = new_state / norm
        else:
            self.field_state = new_state

    def update_submodules(
        self, performance: PerformanceTriplet, algorithm: str
    ) -> None:
        """
        Update resonant submodule states Gᵢ(t) based on algorithm performance.

        Different algorithms contribute to different submodules.
        """
        # Map algorithms to submodule indices
        algorithm_map = {
            "VQE": 0,
            "QAOA": 1,
            "QuantumWalk": 2,
            "Grover": 0,
            "Boson": 1,
            "VQC": 2,
        }

        idx = algorithm_map.get(algorithm, 0) % len(self.submodule_states)

        # Encode performance into submodule
        # Spread the triplet across the field dimension
        encoded = np.zeros(self.dimension)
        encoded[0] = performance.psi
        encoded[1] = performance.rho
        encoded[2] = performance.omega
        encoded[3] = performance.harmonic_mean()
        encoded[4] = performance.geometric_mean()

        # Add some harmonic structure
        for i in range(5, self.dimension):
            encoded[i] = np.sin(2 * np.pi * i / self.dimension) * performance.psi

        # Decay and update submodule
        self.submodule_states[idx] = 0.8 * self.submodule_states[idx] + 0.2 * encoded

    def resonance_with(self, injection: np.ndarray) -> float:
        """
        Compute resonance (correlation) between field state and injection.

        Returns:
            Correlation in [-1, 1], where positive indicates resonance
        """
        if len(injection) != self.dimension:
            raise ValueError("Injection dimension mismatch")

        # Normalize both vectors
        field_norm = self.field_state / (np.linalg.norm(self.field_state) + 1e-10)
        injection_norm = injection / (np.linalg.norm(injection) + 1e-10)

        # Compute dot product (cosine similarity)
        resonance = float(np.dot(field_norm, injection_norm))
        return resonance

    def to_dict(self) -> Dict[str, Any]:
        """Serialize field state to dictionary."""
        return {
            "dimension": self.dimension,
            "field_state": self.field_state.tolist(),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "beta_weights": self.beta_weights,
            "submodule_states": [s.tolist() for s in self.submodule_states],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MandorlaField":
        """Deserialize field from dictionary."""
        field = cls(dimension=data["dimension"])
        field.field_state = np.array(data["field_state"])
        field.alpha = data["alpha"]
        field.gamma = data["gamma"]
        field.beta_weights = data["beta_weights"]
        field.submodule_states = [np.array(s) for s in data["submodule_states"]]
        return field

    def save(self, path: str) -> None:
        """Save field state to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MandorlaField":
        """Load field state from file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


class SeraphicFeedback:
    """
    Seraphic feedback encoder g_SFM: R^n → R^m.

    Maps benchmark outcome vectors u(c) to field injection vectors Iₜ.
    """

    def __init__(self, field_dimension: int = 16):
        """Initialize seraphic feedback encoder."""
        self.field_dimension = field_dimension

    def encode(
        self,
        performance: PerformanceTriplet,
        benchmarks: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Encode performance triplet into field injection vector.

        Args:
            performance: The performance triplet (ψ, ρ, ω)
            benchmarks: Optional raw benchmark data for additional features

        Returns:
            Injection vector Iₜ ∈ R^m
        """
        injection = np.zeros(self.field_dimension)

        # Primary encoding: performance triplet
        injection[0] = performance.psi
        injection[1] = performance.rho
        injection[2] = performance.omega

        # Derived features
        injection[3] = performance.harmonic_mean()
        injection[4] = performance.geometric_mean()
        injection[5] = performance.norm()

        # Quality-stability product (desired resonance)
        injection[6] = performance.psi * performance.rho

        # Quality-efficiency product
        injection[7] = performance.psi * performance.omega

        # All three combined
        injection[8] = performance.psi * performance.rho * performance.omega

        # Add harmonic structure to create resonance patterns
        for i in range(9, self.field_dimension):
            phase = 2 * np.pi * i / self.field_dimension
            injection[i] = (
                0.4 * np.sin(phase) * performance.psi
                + 0.3 * np.cos(phase) * performance.rho
                + 0.3 * np.sin(2 * phase) * performance.omega
            )

        return injection

    def encode_from_benchmarks(
        self, benchmarks: Dict[str, Dict[str, Any]]
    ) -> np.ndarray:
        """
        Encode benchmark results directly into injection vector.

        Args:
            benchmarks: Dictionary of benchmark results

        Returns:
            Injection vector Iₜ ∈ R^m
        """
        from .performance import compute_performance_triplet

        # Compute performance triplet
        performance = compute_performance_triplet(benchmarks)

        # Encode into field vector
        return self.encode(performance, benchmarks)
