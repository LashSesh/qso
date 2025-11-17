"""
Seraphic Calibrator: Main orchestrator for the SCS meta-algorithm.

Implements the 5-step calibration loop:
1. Benchmark
2. Seraphic feedback
3. Double-kick update
4. Proof-of-Resonance
5. CRI-check
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json
import time

from .config import Configuration, ConfigurationSpace
from .performance import (
    PerformanceTriplet,
    BenchmarkLoader,
    compute_performance_triplet,
)
from .field import MandorlaField, SeraphicFeedback
from .operators import DoubleKickOperator
from .por import ProofOfResonance, PoRCriteria
from .cri import ResonanceImpulse, ResonanceImpulseConfig


@dataclass
class CalibrationHistory:
    """Records history of calibration steps."""

    steps: List[Dict[str, Any]] = field(default_factory=list)

    def add_step(
        self,
        step: int,
        config: Configuration,
        performance: PerformanceTriplet,
        j_t: float,
        por_result: bool,
        cri_triggered: bool,
    ) -> None:
        """Add a calibration step to history."""
        self.steps.append(
            {
                "step": step,
                "config": config.to_dict(),
                "performance": performance.to_dict(),
                "j_t": j_t,
                "por_accepted": por_result,
                "cri_triggered": cri_triggered,
                "timestamp": time.time(),
            }
        )

    def save(self, path: str) -> None:
        """Save history to JSON file."""
        with open(path, "w") as f:
            json.dump({"history": self.steps}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibrationHistory":
        """Load history from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        history = cls()
        history.steps = data["history"]
        return history


@dataclass
class CalibratorConfig:
    """Configuration for the Seraphic Calibrator."""

    # Paths
    benchmark_dir: str = "metatron-qso-rs/ci"
    state_file: str = "scs_state.json"
    history_file: str = "scs_history.json"

    # Operator parameters
    update_kick_step: float = 0.3
    stabilization_kick_step: float = 0.2

    # PoR criteria
    por_criteria: PoRCriteria = field(default_factory=PoRCriteria)

    # CRI configuration
    cri_config: ResonanceImpulseConfig = field(default_factory=ResonanceImpulseConfig)

    # Field dimension
    field_dimension: int = 16

    # Enable/disable SCS (opt-in)
    enabled: bool = True


class SeraphicCalibrator:
    """
    Main orchestrator for the Seraphic Calibration Shell.

    Implements the meta-algorithm for fixpoint-directed configuration evolution.
    """

    def __init__(self, config: Optional[CalibratorConfig] = None):
        """
        Initialize the Seraphic Calibrator.

        Args:
            config: Calibration configuration
        """
        self.config = config or CalibratorConfig()

        # Initialize components
        self.config_space = ConfigurationSpace()
        self.field = MandorlaField(dimension=self.config.field_dimension)
        self.feedback = SeraphicFeedback(field_dimension=self.config.field_dimension)
        self.double_kick = DoubleKickOperator(
            update_step=self.config.update_kick_step,
            stabilization_step=self.config.stabilization_kick_step,
        )
        self.por = ProofOfResonance(criteria=self.config.por_criteria)
        self.cri = ResonanceImpulse(config=self.config.cri_config)

        # State
        self.current_config: Optional[Configuration] = None
        self.current_performance: Optional[PerformanceTriplet] = None
        self.step_count = 0
        self.history = CalibrationHistory()

        # Benchmark loader
        self.benchmark_loader = BenchmarkLoader(self.config.benchmark_dir)

    def initialize(self, initial_config: Optional[Configuration] = None) -> None:
        """
        Initialize the calibrator with a starting configuration.

        Args:
            initial_config: Starting configuration (uses default if None)
        """
        if initial_config is None:
            initial_config = self.config_space.default_configuration()

        if not self.config_space.is_valid(initial_config):
            raise ValueError("Invalid initial configuration")

        self.current_config = initial_config
        self.config_space.set_current(initial_config)

        # Load initial benchmarks and compute performance
        benchmarks = self.benchmark_loader.load_all_benchmarks()
        self.current_performance = compute_performance_triplet(benchmarks)

        # Initialize field with first feedback
        injection = self.feedback.encode_from_benchmarks(benchmarks)
        self.field.update(injection)

        # Update CRI
        self.cri.update(self.current_performance)

    def calibration_step(self) -> Dict[str, Any]:
        """
        Execute one calibration step of the SCS meta-algorithm.

        Returns:
            Dictionary with step results and diagnostics
        """
        if not self.config.enabled:
            return {"enabled": False, "message": "SCS is disabled"}

        if self.current_config is None:
            raise RuntimeError("Calibrator not initialized. Call initialize() first.")

        self.step_count += 1
        step_result = {"step": self.step_count}

        # Step 1: Benchmark (load existing benchmarks)
        benchmarks = self.benchmark_loader.load_all_benchmarks()
        step_result["benchmarks_loaded"] = list(benchmarks.keys())

        # Step 2: Seraphic feedback
        injection = self.feedback.encode_from_benchmarks(benchmarks)
        self.field.update(injection)
        self.field.update_submodules(
            self.current_performance, self.current_config.algorithm
        )
        step_result["field_updated"] = True

        # Step 3: Double-kick update
        candidate_config = self.double_kick.apply(
            self.current_config, self.current_performance, self.config_space, self.field
        )
        step_result["candidate_generated"] = True
        step_result["candidate_config"] = candidate_config.to_dict()

        # Step 4: Proof-of-Resonance
        # In practice, we'd benchmark the candidate. For now, estimate performance.
        candidate_performance = self._estimate_candidate_performance(
            candidate_config, benchmarks
        )
        candidate_injection = self.feedback.encode(candidate_performance, benchmarks)

        por_result = self.por.check(
            self.current_config,
            self.current_performance,
            candidate_config,
            candidate_performance,
            self.field,
            candidate_injection,
        )

        por_detailed = self.por.detailed_check(
            self.current_config,
            self.current_performance,
            candidate_config,
            candidate_performance,
            self.field,
            candidate_injection,
        )

        step_result["por_result"] = por_result
        step_result["por_detailed"] = por_detailed

        # Accept or reject candidate
        if por_result:
            self.current_config = candidate_config
            self.current_performance = candidate_performance
            self.config_space.set_current(candidate_config)
            step_result["accepted"] = True
        else:
            step_result["accepted"] = False

        # Step 5: CRI-check
        j_t = self.cri.update(self.current_performance)
        step_result["j_t"] = j_t

        cri_triggered = False
        if self.cri.should_trigger(self.field):
            new_regime_config = self.cri.apply_impulse(
                self.current_config, self.config_space, self.field
            )
            self.current_config = new_regime_config
            self.config_space.set_current(new_regime_config)
            cri_triggered = True
            step_result["cri_triggered"] = True
            step_result["new_regime_config"] = new_regime_config.to_dict()
        else:
            step_result["cri_triggered"] = False

        # Record step in history
        self.history.add_step(
            self.step_count,
            self.current_config,
            self.current_performance,
            j_t,
            por_result,
            cri_triggered,
        )

        # Add diagnostics
        step_result["current_performance"] = self.current_performance.to_dict()
        step_result["cri_diagnostics"] = self.cri.get_diagnostics()

        return step_result

    def _estimate_candidate_performance(
        self, config: Configuration, benchmarks: Dict[str, Any]
    ) -> PerformanceTriplet:
        """
        Estimate performance of candidate configuration.

        In a full implementation, this would run actual benchmarks.
        Here we use heuristics to estimate.
        """
        # Start with current performance
        base_psi = self.current_performance.psi
        base_rho = self.current_performance.rho
        base_omega = self.current_performance.omega

        # Apply heuristic adjustments
        psi = base_psi
        rho = base_rho
        omega = base_omega

        # Quality heuristics
        if config.ansatz_type == "Metatron" and 1 <= config.ansatz_depth <= 3:
            psi = min(1.0, psi + 0.02)
        if config.optimizer == "Adam":
            psi = min(1.0, psi + 0.01)

        # Stability heuristics
        if config.num_random_starts >= 3:
            rho = min(1.0, rho + 0.03)

        # Efficiency heuristics
        if config.ansatz_depth <= 2:
            omega = min(1.0, omega + 0.02)

        return PerformanceTriplet(psi=psi, rho=rho, omega=omega)

    def save_state(self, path: Optional[str] = None) -> None:
        """
        Save current calibrator state to file.

        Args:
            path: Path to state file (uses config default if None)
        """
        if path is None:
            path = self.config.state_file

        state = {
            "step_count": self.step_count,
            "current_config": self.current_config.to_dict()
            if self.current_config
            else None,
            "current_performance": self.current_performance.to_dict()
            if self.current_performance
            else None,
            "field": self.field.to_dict(),
            "cri_diagnostics": self.cri.get_diagnostics(),
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: Optional[str] = None) -> None:
        """
        Load calibrator state from file.

        Args:
            path: Path to state file (uses config default if None)
        """
        if path is None:
            path = self.config.state_file

        with open(path, "r") as f:
            state = json.load(f)

        self.step_count = state["step_count"]
        if state["current_config"]:
            self.current_config = Configuration.from_dict(state["current_config"])
        if state["current_performance"]:
            perf_data = state["current_performance"]
            self.current_performance = PerformanceTriplet(
                psi=perf_data["psi"], rho=perf_data["rho"], omega=perf_data["omega"]
            )
        self.field = MandorlaField.from_dict(state["field"])

    def save_history(self, path: Optional[str] = None) -> None:
        """Save calibration history."""
        if path is None:
            path = self.config.history_file
        self.history.save(path)

    def get_best_configuration(self) -> Configuration:
        """
        Get the current best configuration.

        Returns:
            Current configuration (at fixpoint or approaching it)
        """
        if self.current_config is None:
            raise RuntimeError("Calibrator not initialized")
        return self.current_config

    def run_calibration(self, num_steps: int = 10) -> List[Dict[str, Any]]:
        """
        Run multiple calibration steps.

        Args:
            num_steps: Number of calibration steps to run

        Returns:
            List of step results
        """
        results = []
        for _ in range(num_steps):
            result = self.calibration_step()
            results.append(result)
        return results
