"""
Proof-of-Resonance (PoR) criterion for configuration acceptance.

Determines when a candidate configuration c' = T(c) should replace
the current configuration c.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .config import Configuration
from .performance import PerformanceTriplet
from .field import MandorlaField


@dataclass
class PoRCriteria:
    """
    Thresholds and tolerances for Proof-of-Resonance test.
    """

    # Minimum required quality improvement (or 0 for non-decrease)
    min_quality_delta: float = 0.0

    # Maximum allowed stability degradation
    stability_tolerance: float = 0.1

    # Minimum allowed efficiency (absolute)
    min_efficiency: float = 0.3

    # Minimum field resonance correlation
    min_field_resonance: float = 0.0


class ProofOfResonance:
    """
    Implements the Proof-of-Resonance (PoR) acceptance test.

    A candidate configuration c' passes PoR if:
    (i)   ψ(c') ≥ ψ(c)  (non-decrease of quality)
    (ii)  ρ(c') ≥ ρ(c) - tolerance  (stability consistency)
    (iii) ω(c') ≥ min_threshold  (efficiency consistency)
    (iv)  M(t) resonates positively with injection from c'
    """

    def __init__(self, criteria: Optional[PoRCriteria] = None):
        """
        Initialize PoR checker.

        Args:
            criteria: Acceptance criteria thresholds
        """
        self.criteria = criteria or PoRCriteria()

    def check(
        self,
        current_config: Configuration,
        current_performance: PerformanceTriplet,
        candidate_config: Configuration,
        candidate_performance: PerformanceTriplet,
        field: Optional[MandorlaField] = None,
        candidate_injection: Optional[np.ndarray] = None
    ) -> bool:
        """
        Check if candidate configuration passes PoR test.

        Args:
            current_config: Current configuration c
            current_performance: Φ(c)
            candidate_config: Candidate configuration c'
            candidate_performance: Φ(c')
            field: Mandorla field M(t)
            candidate_injection: Optional injection vector for c'

        Returns:
            True if candidate passes PoR, False otherwise
        """
        # (i) Non-decrease of quality
        if not self._check_quality(current_performance, candidate_performance):
            return False

        # (ii) Stability consistency
        if not self._check_stability(current_performance, candidate_performance):
            return False

        # (iii) Efficiency consistency
        if not self._check_efficiency(candidate_performance):
            return False

        # (iv) Field-level resonance
        if field is not None and candidate_injection is not None:
            if not self._check_field_resonance(field, candidate_injection):
                return False

        return True

    def _check_quality(
        self,
        current: PerformanceTriplet,
        candidate: PerformanceTriplet
    ) -> bool:
        """
        Check quality non-decrease: ψ(c') ≥ ψ(c) + δ
        """
        return candidate.psi >= current.psi + self.criteria.min_quality_delta

    def _check_stability(
        self,
        current: PerformanceTriplet,
        candidate: PerformanceTriplet
    ) -> bool:
        """
        Check stability consistency: ρ(c') ≥ ρ(c) - tolerance
        """
        return candidate.rho >= current.rho - self.criteria.stability_tolerance

    def _check_efficiency(self, candidate: PerformanceTriplet) -> bool:
        """
        Check efficiency consistency: ω(c') ≥ min_threshold
        """
        return candidate.omega >= self.criteria.min_efficiency

    def _check_field_resonance(
        self,
        field: MandorlaField,
        injection: np.ndarray
    ) -> bool:
        """
        Check field-level resonance: M(t) · I(c') > threshold
        """
        resonance = field.resonance_with(injection)
        return resonance >= self.criteria.min_field_resonance

    def detailed_check(
        self,
        current_config: Configuration,
        current_performance: PerformanceTriplet,
        candidate_config: Configuration,
        candidate_performance: PerformanceTriplet,
        field: Optional[MandorlaField] = None,
        candidate_injection: Optional[np.ndarray] = None
    ) -> dict:
        """
        Perform detailed PoR check and return results for each criterion.

        Returns:
            Dictionary with pass/fail for each criterion and overall result
        """
        results = {}

        # Quality check
        quality_pass = self._check_quality(current_performance, candidate_performance)
        results['quality'] = {
            'pass': quality_pass,
            'current': current_performance.psi,
            'candidate': candidate_performance.psi,
            'delta': candidate_performance.psi - current_performance.psi,
        }

        # Stability check
        stability_pass = self._check_stability(current_performance, candidate_performance)
        results['stability'] = {
            'pass': stability_pass,
            'current': current_performance.rho,
            'candidate': candidate_performance.rho,
            'delta': candidate_performance.rho - current_performance.rho,
            'tolerance': self.criteria.stability_tolerance,
        }

        # Efficiency check
        efficiency_pass = self._check_efficiency(candidate_performance)
        results['efficiency'] = {
            'pass': efficiency_pass,
            'candidate': candidate_performance.omega,
            'min_threshold': self.criteria.min_efficiency,
        }

        # Field resonance check
        if field is not None and candidate_injection is not None:
            resonance = field.resonance_with(candidate_injection)
            resonance_pass = self._check_field_resonance(field, candidate_injection)
            results['field_resonance'] = {
                'pass': resonance_pass,
                'resonance': resonance,
                'min_threshold': self.criteria.min_field_resonance,
            }
        else:
            results['field_resonance'] = {'pass': True, 'resonance': None}

        # Overall result
        results['overall'] = all([
            quality_pass,
            stability_pass,
            efficiency_pass,
            results['field_resonance']['pass']
        ])

        return results
