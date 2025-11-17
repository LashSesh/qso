"""
Double-kick operator T = Φ_V ∘ Φ_U on configuration space.

Implements the field-theoretic contraction operator that moves configurations
towards fixpoint attractors through:
- Φ_U: update kick (improves quality ψ)
- Φ_V: stabilization kick (improves stability ρ and efficiency ω)
"""

from typing import Optional, Tuple

from .config import Configuration, ConfigurationSpace
from .performance import PerformanceTriplet
from .field import MandorlaField


class UpdateKick:
    """
    Φ_U: Update kick that improves semantic quality ψ.

    Φ_U(c) = c + η_U ∇_c ψ(c)

    Moves configuration along ascent direction of quality.
    """

    def __init__(self, step_size: float = 0.3):
        """
        Initialize update kick.

        Args:
            step_size: η_U, step size for quality improvement
        """
        self.step_size = step_size

    def apply(
        self,
        config: Configuration,
        current_performance: PerformanceTriplet,
        config_space: ConfigurationSpace,
        field: Optional[MandorlaField] = None,
    ) -> Configuration:
        """
        Apply update kick to improve quality.

        Args:
            config: Current configuration
            current_performance: Performance triplet Φ(c)
            config_space: Configuration space for neighbor generation
            field: Optional Mandorla field for resonance guidance

        Returns:
            Updated configuration c' = Φ_U(c)
        """
        # Generate candidate neighbors
        neighbors = config_space.generate_neighbors(config, num_neighbors=8)

        if not neighbors:
            return config

        # Select neighbor that would improve quality
        # In practice, this is a gradient estimation via finite differences
        best_neighbor = config
        best_score = current_performance.psi

        for neighbor in neighbors:
            # Heuristic score: prefer changes that might increase quality
            # For now, use a simple heuristic based on configuration parameters
            score = self._estimate_quality_improvement(
                config, neighbor, current_performance
            )

            if score > best_score:
                best_score = score
                best_neighbor = neighbor

        return best_neighbor

    def _estimate_quality_improvement(
        self,
        current: Configuration,
        candidate: Configuration,
        current_perf: PerformanceTriplet,
    ) -> float:
        """
        Estimate quality improvement heuristically.

        In a full implementation, this would run a partial benchmark.
        Here we use heuristics based on known good configurations.
        """
        score = current_perf.psi

        # Heuristics for quality improvement:
        # 1. Metatron ansatz with depth 1-3 is generally good
        if candidate.ansatz_type == "Metatron" and 1 <= candidate.ansatz_depth <= 3:
            score += 0.05

        # 2. Adam optimizer is generally reliable
        if candidate.optimizer == "Adam":
            score += 0.02

        # 3. Multiple random starts can improve quality
        if candidate.num_random_starts > current.num_random_starts:
            score += 0.01 * (candidate.num_random_starts - current.num_random_starts)

        # 4. Moderate learning rate is good
        if 0.005 <= candidate.learning_rate <= 0.02:
            score += 0.02

        return min(1.0, score)


class StabilizationKick:
    """
    Φ_V: Stabilization kick that improves stability ρ and efficiency ω.

    Φ_V(c) = c + η_V R(c)

    where R(c) points towards higher stability and efficiency.
    """

    def __init__(self, step_size: float = 0.2):
        """
        Initialize stabilization kick.

        Args:
            step_size: η_V, step size for stability/efficiency improvement
        """
        self.step_size = step_size

    def apply(
        self,
        config: Configuration,
        current_performance: PerformanceTriplet,
        config_space: ConfigurationSpace,
        field: Optional[MandorlaField] = None,
    ) -> Configuration:
        """
        Apply stabilization kick to improve stability and efficiency.

        Args:
            config: Current configuration
            current_performance: Performance triplet Φ(c)
            config_space: Configuration space
            field: Optional Mandorla field

        Returns:
            Stabilized configuration c' = Φ_V(c)
        """
        # Generate candidates
        neighbors = config_space.generate_neighbors(config, num_neighbors=8)

        if not neighbors:
            return config

        # Select neighbor that improves stability/efficiency without hurting quality
        best_neighbor = config
        best_score = current_performance.rho * 0.6 + current_performance.omega * 0.4

        for neighbor in neighbors:
            score = self._estimate_stability_improvement(
                config, neighbor, current_performance
            )

            if score > best_score:
                best_score = score
                best_neighbor = neighbor

        return best_neighbor

    def _estimate_stability_improvement(
        self,
        current: Configuration,
        candidate: Configuration,
        current_perf: PerformanceTriplet,
    ) -> float:
        """
        Estimate stability and efficiency improvement heuristically.
        """
        rho_score = current_perf.rho
        omega_score = current_perf.omega

        # Heuristics for stability:
        # 1. Multiple random starts increase stability
        if candidate.num_random_starts >= 3:
            rho_score += 0.05

        # 2. Lower depth can be more stable (less overparameterization)
        if candidate.ansatz_depth <= 2:
            rho_score += 0.03

        # Heuristics for efficiency:
        # 1. Lower depth is more efficient
        if candidate.ansatz_depth < current.ansatz_depth:
            omega_score += 0.05

        # 2. Fewer iterations can be more efficient if quality is maintained
        if candidate.max_iterations < current.max_iterations:
            omega_score += 0.02

        # Combined score (weighted)
        return min(1.0, 0.6 * rho_score + 0.4 * omega_score)


class DoubleKickOperator:
    """
    T = Φ_V ∘ Φ_U: Double-kick operator on configuration space.

    Applies update kick followed by stabilization kick to create
    locally contractive dynamics towards fixpoint attractors.
    """

    def __init__(self, update_step: float = 0.3, stabilization_step: float = 0.2):
        """
        Initialize double-kick operator.

        Args:
            update_step: Step size η_U for quality improvement
            stabilization_step: Step size η_V for stability improvement
        """
        self.update_kick = UpdateKick(step_size=update_step)
        self.stabilization_kick = StabilizationKick(step_size=stabilization_step)

    def apply(
        self,
        config: Configuration,
        current_performance: PerformanceTriplet,
        config_space: ConfigurationSpace,
        field: Optional[MandorlaField] = None,
    ) -> Configuration:
        """
        Apply T = Φ_V ∘ Φ_U to configuration.

        Args:
            config: Current configuration c
            current_performance: Performance triplet Φ(c)
            config_space: Configuration space
            field: Optional Mandorla field for resonance guidance

        Returns:
            New configuration c' = T(c) = Φ_V(Φ_U(c))
        """
        # First: update kick (improve quality)
        intermediate = self.update_kick.apply(
            config, current_performance, config_space, field
        )

        # Second: stabilization kick (improve stability and efficiency)
        result = self.stabilization_kick.apply(
            intermediate, current_performance, config_space, field
        )

        return result

    def iterate(
        self,
        config: Configuration,
        performance: PerformanceTriplet,
        config_space: ConfigurationSpace,
        field: Optional[MandorlaField] = None,
        num_iterations: int = 5,
    ) -> Tuple[Configuration, float]:
        """
        Iterate T multiple times to approach fixpoint.

        Args:
            config: Initial configuration
            performance: Initial performance
            config_space: Configuration space
            field: Optional Mandorla field
            num_iterations: Number of iterations

        Returns:
            Tuple of (final_config, convergence_rate)
        """
        current = config
        distances = []

        for i in range(num_iterations):
            next_config = self.apply(current, performance, config_space, field)

            # Measure distance moved
            dist = current.distance(next_config)
            distances.append(dist)

            # Check for fixpoint convergence
            if dist < 0.01:
                break

            current = next_config

        # Estimate convergence rate (Lipschitz constant approximation)
        if len(distances) >= 2:
            convergence_rate = distances[-1] / (distances[0] + 1e-10)
        else:
            convergence_rate = 0.0

        return current, convergence_rate

    def is_contractive(self, distances: list) -> bool:
        """
        Check if the operator is contractive based on distance sequence.

        Returns True if ||T(c_i) - T(c_{i-1})|| < ||c_i - c_{i-1}||
        """
        if len(distances) < 2:
            return False

        # Check if distances are decreasing (contraction)
        for i in range(1, len(distances)):
            if distances[i] >= distances[i - 1]:
                return False

        return True
