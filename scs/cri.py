"""
CRI-style resonance impulse for controlled regime switching.

When the system stagnates in a locally optimal but globally suboptimal
attractor, the CRI mechanism triggers a controlled phase transition to
a different contraction region.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

from .config import Configuration, ConfigurationSpace
from .performance import PerformanceTriplet
from .field import MandorlaField


@dataclass
class GlobalCalibrationState:
    """
    Tracks global calibration functional J(t) over time.

    J(t) = ψ_avg(t) · ρ_avg(t) · ω_avg(t)
    """

    history: List[float] = field(default_factory=list)
    window_size: int = 5  # Number of steps to track for stagnation detection

    def update(self, performance: PerformanceTriplet) -> float:
        """
        Update global functional with new performance.

        Returns:
            Current J(t) value
        """
        # For single configuration, J(t) = ψ · ρ · ω
        j_t = performance.psi * performance.rho * performance.omega
        self.history.append(j_t)

        # Keep only recent window
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size * 2 :]

        return j_t

    def current_value(self) -> float:
        """Get current J(t) value."""
        return self.history[-1] if self.history else 0.0

    def is_stagnating(self, threshold: float = 0.01) -> bool:
        """
        Check if J(t) has stagnated over the window.

        Returns True if variance of recent J(t) values is below threshold
        and the trend is not improving.
        """
        if len(self.history) < self.window_size:
            return False

        recent = self.history[-self.window_size :]
        variance = np.var(recent)

        # Low variance indicates stagnation
        if variance > threshold:
            return False

        # Also check if trend is improving
        if len(self.history) >= self.window_size * 2:
            older = self.history[-self.window_size * 2 : -self.window_size]
            recent_mean = np.mean(recent)
            older_mean = np.mean(older)

            # Improving trend → not stagnating
            if recent_mean > older_mean + threshold:
                return False

        return True

    def is_degrading(self, threshold: float = 0.05) -> bool:
        """
        Check if J(t) is actively degrading.

        Returns True if recent trend shows decline.
        """
        if len(self.history) < self.window_size * 2:
            return False

        recent = self.history[-self.window_size :]
        older = self.history[-self.window_size * 2 : -self.window_size]

        recent_mean = np.mean(recent)
        older_mean = np.mean(older)

        # Degrading if recent < older by threshold
        return recent_mean < older_mean - threshold


@dataclass
class ResonanceImpulseConfig:
    """
    Configuration for CRI-style resonance impulse triggers.
    """

    # Minimum steps before triggering impulse
    min_steps: int = 10

    # Stagnation detection threshold
    stagnation_threshold: float = 0.01

    # Degradation detection threshold
    degradation_threshold: float = 0.05

    # Minimum field resonance to trigger alternative regime
    min_field_resonance: float = 0.3


class ResonanceImpulse:
    """
    Implements CRI-style resonance impulse for regime switching.

    When J(t) stagnates/degrades and the field M(t) resonates with an
    alternative configuration manifold, triggers a controlled phase
    transition.
    """

    def __init__(self, config: Optional[ResonanceImpulseConfig] = None):
        """
        Initialize resonance impulse mechanism.

        Args:
            config: CRI trigger configuration
        """
        self.config = config or ResonanceImpulseConfig()
        self.global_state = GlobalCalibrationState()
        self.steps_since_last_impulse = 0

    def update(self, performance: PerformanceTriplet) -> float:
        """
        Update global calibration state.

        Args:
            performance: Current performance triplet

        Returns:
            Current J(t) value
        """
        self.steps_since_last_impulse += 1
        return self.global_state.update(performance)

    def should_trigger(self, field: MandorlaField) -> bool:
        """
        Determine if resonance impulse should be triggered.

        Args:
            field: Current Mandorla field state

        Returns:
            True if impulse should be triggered
        """
        # Must have minimum number of steps
        if self.steps_since_last_impulse < self.config.min_steps:
            return False

        # Check for stagnation or degradation
        is_stuck = self.global_state.is_stagnating(
            self.config.stagnation_threshold
        ) or self.global_state.is_degrading(self.config.degradation_threshold)

        if not is_stuck:
            return False

        # Check if field resonates with alternative regime
        # For this, we look at field energy distribution
        field_energy = np.linalg.norm(field.field_state)
        if field_energy < self.config.min_field_resonance:
            return False

        return True

    def apply_impulse(
        self,
        current_config: Configuration,
        config_space: ConfigurationSpace,
        field: MandorlaField,
    ) -> Configuration:
        """
        Apply resonance impulse to transition to new regime.

        Args:
            current_config: Current configuration
            config_space: Configuration space
            field: Mandorla field for resonance guidance

        Returns:
            New configuration in different contraction region
        """
        self.steps_since_last_impulse = 0  # Reset counter

        # Determine which regime to switch to based on field resonance
        new_config = self._select_alternative_regime(
            current_config, config_space, field
        )

        return new_config

    def _select_alternative_regime(
        self,
        current: Configuration,
        config_space: ConfigurationSpace,
        field: MandorlaField,
    ) -> Configuration:
        """
        Select alternative configuration regime based on field resonance.

        Strategy:
        1. If in VQE regime, try QAOA or vice versa
        2. If in one ansatz family, try another
        3. If using one optimizer, try another
        """
        new_config = current.copy()

        # Strategy 1: Switch algorithm family
        if current.algorithm == "VQE":
            new_config.algorithm = "QAOA"
            new_config.ansatz_depth = 3  # QAOA typically uses higher depth
        elif current.algorithm == "QAOA":
            new_config.algorithm = "VQE"
            new_config.ansatz_depth = 2  # VQE typically lower depth
        elif current.algorithm == "QuantumWalk":
            new_config.algorithm = "VQE"
        else:
            new_config.algorithm = "VQE"

        # Strategy 2: Switch ansatz type
        ansatz_alternatives = {
            "Metatron": "EfficientSU2",
            "EfficientSU2": "HardwareEfficient",
            "HardwareEfficient": "Metatron",
        }
        new_config.ansatz_type = ansatz_alternatives.get(
            current.ansatz_type, "Metatron"
        )

        # Strategy 3: Switch optimizer
        optimizer_alternatives = {
            "Adam": "LBFGS",
            "LBFGS": "GradientDescent",
            "GradientDescent": "Adam",
            "COBYLA": "Adam",
        }
        new_config.optimizer = optimizer_alternatives.get(current.optimizer, "Adam")

        # Adjust hyperparameters for new regime
        if new_config.optimizer == "Adam":
            new_config.learning_rate = 0.01
        elif new_config.optimizer == "LBFGS":
            new_config.learning_rate = 0.1  # LBFGS can use larger steps
        else:
            new_config.learning_rate = 0.005

        # Validate and return
        if config_space.is_valid(new_config):
            return new_config
        else:
            # Fallback to default
            return config_space.default_configuration()

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about CRI state.

        Returns:
            Dictionary with diagnostic data
        """
        return {
            "steps_since_impulse": self.steps_since_last_impulse,
            "current_j_t": self.global_state.current_value(),
            "is_stagnating": self.global_state.is_stagnating(),
            "is_degrading": self.global_state.is_degrading(),
            "history_length": len(self.global_state.history),
            "j_t_history": self.global_state.history[-10:],  # Last 10 values
        }
