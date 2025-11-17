"""
SCS Auto-Tuner Core API

Simplified, high-level API for using SCS as a generic auto-tuner.
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass

from .config import Configuration, ConfigurationSpace
from .performance import PerformanceTriplet, compute_performance_triplet
from .field import MandorlaField
from .calibrator import SeraphicCalibrator, CalibratorConfig
from .benchmark import (
    load_benchmarks,
    write_benchmark,
)


@dataclass
class NewConfigProposal:
    """
    Proposal for a new configuration from SCS auto-tuner.
    """

    # Proposed configuration
    config: Configuration

    # Performance indicators
    current_performance: PerformanceTriplet
    estimated_performance: PerformanceTriplet

    # Decision indicators
    por_accepted: bool
    cri_triggered: bool
    delta_phi: float  # Change in performance norm

    # Detailed info
    step: int
    j_t: float  # Global calibration functional
    por_details: Optional[Dict[str, Any]] = None
    cri_diagnostics: Optional[Dict[str, Any]] = None


class AutoTuner:
    """
    Simplified auto-tuner API for SCS.

    Provides a high-level interface for:
    - Loading benchmarks
    - Proposing new configurations
    - Tracking calibration state
    """

    def __init__(
        self,
        benchmark_dir: str = "benchmarks",
        state_file: str = "scs_state.json",
        history_file: str = "scs_history.json",
        enabled: bool = True,
    ):
        """
        Initialize auto-tuner.

        Args:
            benchmark_dir: Directory containing benchmark JSON files
            state_file: Path to state persistence file
            history_file: Path to history file
            enabled: Whether SCS is enabled (opt-in)
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.state_file = state_file
        self.history_file = history_file
        self.enabled = enabled

        # Create calibrator config
        config = CalibratorConfig(
            benchmark_dir=str(self.benchmark_dir),
            state_file=state_file,
            history_file=history_file,
            enabled=enabled,
        )

        # Initialize calibrator
        self.calibrator = SeraphicCalibrator(config)
        self._initialized = False

    def initialize(
        self, initial_config: Optional[Configuration] = None
    ) -> Configuration:
        """
        Initialize the auto-tuner with a starting configuration.

        Args:
            initial_config: Starting configuration (uses default if None)

        Returns:
            The initial configuration
        """
        if not self.enabled:
            # Return default config without initializing
            return ConfigurationSpace().default_configuration()

        # Load existing state if available
        if Path(self.state_file).exists():
            self.calibrator.load_state()
            self._initialized = True
            return self.calibrator.current_config

        # Otherwise, initialize fresh
        self.calibrator.initialize(initial_config)
        self._initialized = True

        # Save initial state
        self.calibrator.save_state()

        return self.calibrator.current_config

    def ingest_benchmark(
        self,
        system: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        raw_results: Optional[Dict[str, Any]] = None,
        aux: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Ingest a new benchmark result.

        Args:
            system: System identifier (e.g., "vqe", "qaoa_maxcut")
            config: Configuration dictionary
            metrics: Performance triplet {psi, rho, omega}
            raw_results: Optional raw results
            aux: Optional auxiliary metadata

        Returns:
            Path to written benchmark file
        """
        # Ensure benchmark directory exists
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)

        # Write benchmark to file
        output_path = write_benchmark(
            system=system,
            config=config,
            metrics=metrics,
            raw_results=raw_results,
            aux=aux,
            output_path=None,  # Auto-generate
        )

        return output_path

    def propose_new_config(
        self,
        current_config: Optional[Configuration] = None,
        benchmarks_path: Optional[Union[str, Path]] = None,
    ) -> NewConfigProposal:
        """
        Propose a new configuration based on recent benchmarks.

        Args:
            current_config: Current configuration (uses calibrator's if None)
            benchmarks_path: Path to benchmark files (uses benchmark_dir if None)

        Returns:
            NewConfigProposal with proposed configuration and diagnostics

        Raises:
            RuntimeError: If auto-tuner not initialized
        """
        if not self.enabled:
            # Return no-op proposal
            default_config = ConfigurationSpace().default_configuration()
            default_perf = PerformanceTriplet(psi=0.5, rho=0.5, omega=0.5)
            return NewConfigProposal(
                config=default_config,
                current_performance=default_perf,
                estimated_performance=default_perf,
                por_accepted=False,
                cri_triggered=False,
                delta_phi=0.0,
                step=0,
                j_t=0.125,
            )

        if not self._initialized:
            raise RuntimeError("Auto-tuner not initialized. Call initialize() first.")

        # Run calibration step
        step_result = self.calibrator.calibration_step()

        # Extract proposal information
        current_perf = self.calibrator.current_performance
        estimated_perf_dict = step_result.get(
            "current_performance", current_perf.to_dict()
        )
        estimated_perf = PerformanceTriplet(
            psi=estimated_perf_dict["psi"],
            rho=estimated_perf_dict["rho"],
            omega=estimated_perf_dict["omega"],
        )

        # Compute delta
        delta_phi = estimated_perf.norm() - current_perf.norm()

        # Create proposal
        proposal = NewConfigProposal(
            config=self.calibrator.current_config.copy(),
            current_performance=current_perf,
            estimated_performance=estimated_perf,
            por_accepted=step_result.get("accepted", False),
            cri_triggered=step_result.get("cri_triggered", False),
            delta_phi=delta_phi,
            step=step_result["step"],
            j_t=step_result["j_t"],
            por_details=step_result.get("por_detailed"),
            cri_diagnostics=step_result.get("cri_diagnostics"),
        )

        # Save updated state
        self.calibrator.save_state()
        self.calibrator.save_history()

        return proposal

    def get_current_config(self) -> Configuration:
        """
        Get the current best configuration.

        Returns:
            Current configuration
        """
        if not self._initialized:
            return ConfigurationSpace().default_configuration()

        return self.calibrator.get_best_configuration()

    def get_current_performance(self) -> PerformanceTriplet:
        """
        Get the current performance triplet.

        Returns:
            Current performance Φ(c)
        """
        if not self._initialized:
            return PerformanceTriplet(psi=0.5, rho=0.5, omega=0.5)

        return self.calibrator.current_performance

    def get_field_state(self) -> MandorlaField:
        """
        Get the current Mandorla field state.

        Returns:
            MandorlaField instance
        """
        return self.calibrator.field

    def get_calibration_history(self) -> List[Dict[str, Any]]:
        """
        Get calibration history.

        Returns:
            List of historical step records
        """
        return self.calibrator.history.steps

    def run_auto_tuning(
        self,
        num_steps: int = 10,
        min_quality_threshold: float = 0.9,
    ) -> List[NewConfigProposal]:
        """
        Run multiple auto-tuning steps until convergence or max steps.

        Args:
            num_steps: Maximum number of tuning steps
            min_quality_threshold: Stop if quality exceeds this threshold

        Returns:
            List of configuration proposals
        """
        proposals = []

        for _ in range(num_steps):
            proposal = self.propose_new_config()
            proposals.append(proposal)

            # Check convergence
            if proposal.current_performance.psi >= min_quality_threshold:
                break

            # Check stagnation (no improvement in 5 steps)
            if len(proposals) >= 5:
                recent_psi = [p.current_performance.psi for p in proposals[-5:]]
                if max(recent_psi) - min(recent_psi) < 0.01:
                    break  # Stagnated

        return proposals

    def reset(self) -> None:
        """
        Reset auto-tuner state to default configuration.
        """
        # Remove state files
        if Path(self.state_file).exists():
            Path(self.state_file).unlink()

        # Re-initialize
        self._initialized = False
        config = CalibratorConfig(
            benchmark_dir=str(self.benchmark_dir),
            state_file=self.state_file,
            history_file=self.history_file,
            enabled=self.enabled,
        )
        self.calibrator = SeraphicCalibrator(config)


# Convenience functions


def create_auto_tuner(
    benchmark_dir: str = "benchmarks",
    enabled: bool = True,
) -> AutoTuner:
    """
    Create and initialize an AutoTuner instance.

    Args:
        benchmark_dir: Directory for benchmark files
        enabled: Whether to enable SCS (opt-in)

    Returns:
        AutoTuner instance
    """
    tuner = AutoTuner(benchmark_dir=benchmark_dir, enabled=enabled)
    tuner.initialize()
    return tuner


def quick_tune(
    system: str,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    benchmark_dir: str = "benchmarks",
) -> Configuration:
    """
    Quick auto-tuning: ingest benchmark and get new config proposal.

    Args:
        system: System identifier
        config: Configuration dict
        metrics: Performance triplet {psi, rho, omega}
        benchmark_dir: Benchmark directory

    Returns:
        Proposed new configuration
    """
    tuner = AutoTuner(benchmark_dir=benchmark_dir, enabled=True)
    tuner.initialize()

    # Ingest benchmark
    tuner.ingest_benchmark(system, config, metrics)

    # Get proposal
    proposal = tuner.propose_new_config()

    return proposal.config


def load_and_compute_performance(
    benchmark_path: Union[str, Path],
) -> PerformanceTriplet:
    """
    Load benchmarks and compute aggregate performance triplet.

    Args:
        benchmark_path: Path to benchmark file(s) or directory

    Returns:
        Aggregate performance triplet
    """
    # Load benchmarks
    records = load_benchmarks(benchmark_path)

    if not records:
        return PerformanceTriplet(psi=0.5, rho=0.5, omega=0.5)

    # Convert to dict format expected by compute_performance_triplet
    benchmarks = {}
    for record in records:
        system_key = record.system
        if system_key not in benchmarks:
            benchmarks[system_key] = {
                "metrics": record.metrics,
                "config": record.config,
                "results": [],
            }
        benchmarks[system_key]["results"].append(record.raw_results or {})

    # Compute aggregate performance
    return compute_performance_triplet(benchmarks)
