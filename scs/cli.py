#!/usr/bin/env python3
"""
CLI entrypoint for Seraphic Calibration Shell (SCS).

Provides command-line interface for running SCS calibration steps.
"""

import argparse
import sys
from pathlib import Path

from .calibrator import SeraphicCalibrator, CalibratorConfig
from .config import Configuration, ConfigurationSpace


def cmd_init(args) -> int:
    """Initialize SCS with default or specified configuration."""
    print("Initializing Seraphic Calibration Shell...")

    config = CalibratorConfig(
        benchmark_dir=args.benchmark_dir,
        state_file=args.state_file,
        history_file=args.history_file,
        enabled=True,
    )

    calibrator = SeraphicCalibrator(config)

    # Create initial configuration
    if args.config_file:
        initial_config = Configuration.from_file(args.config_file)
        print(f"Loaded initial configuration from {args.config_file}")
    else:
        initial_config = ConfigurationSpace().default_configuration()
        print("Using default initial configuration")

    calibrator.initialize(initial_config)

    # Save state
    calibrator.save_state()
    print(f"Saved initial state to {config.state_file}")

    # Print initial performance
    print("\nInitial Performance:")
    print(f"  ψ (quality):     {calibrator.current_performance.psi:.4f}")
    print(f"  ρ (stability):   {calibrator.current_performance.rho:.4f}")
    print(f"  ω (efficiency):  {calibrator.current_performance.omega:.4f}")

    # Save initial configuration
    best_config_path = args.output or "scs_best_config.json"
    calibrator.current_config.to_file(best_config_path)
    print(f"\nSaved configuration to {best_config_path}")

    return 0


def cmd_step(args) -> int:
    """Run one or more calibration steps."""
    print(f"Running {args.num_steps} calibration step(s)...")

    # Load configuration
    config = CalibratorConfig(
        benchmark_dir=args.benchmark_dir,
        state_file=args.state_file,
        history_file=args.history_file,
        enabled=True,
    )

    calibrator = SeraphicCalibrator(config)

    # Load existing state or initialize
    if Path(config.state_file).exists():
        calibrator.load_state()
        print(f"Loaded state from {config.state_file}")
    else:
        print("No existing state found, initializing...")
        calibrator.initialize()

    # Run calibration steps
    results = calibrator.run_calibration(args.num_steps)

    # Print results
    for i, result in enumerate(results, 1):
        print(f"\nStep {result['step']}:")
        print(f"  Accepted: {result.get('accepted', False)}")
        print(f"  J(t): {result['j_t']:.4f}")
        print(f"  CRI triggered: {result['cri_triggered']}")

        if result.get("accepted"):
            perf = result["current_performance"]
            print(
                f"  Performance: ψ={perf['psi']:.4f}, ρ={perf['rho']:.4f}, ω={perf['omega']:.4f}"
            )

    # Save state and history
    calibrator.save_state()
    calibrator.save_history()
    print(f"\nSaved state to {config.state_file}")
    print(f"Saved history to {config.history_file}")

    # Save best configuration
    best_config_path = args.output or "scs_best_config.json"
    best_config = calibrator.get_best_configuration()
    best_config.to_file(best_config_path)
    print(f"Saved best configuration to {best_config_path}")

    # Print final performance
    print("\nFinal Performance:")
    print(f"  ψ (quality):     {calibrator.current_performance.psi:.4f}")
    print(f"  ρ (stability):   {calibrator.current_performance.rho:.4f}")
    print(f"  ω (efficiency):  {calibrator.current_performance.omega:.4f}")
    print(f"  Harmonic mean:   {calibrator.current_performance.harmonic_mean():.4f}")

    return 0


def cmd_status(args) -> int:
    """Show current SCS status."""
    config = CalibratorConfig(
        benchmark_dir=args.benchmark_dir,
        state_file=args.state_file,
    )

    if not Path(config.state_file).exists():
        print("SCS not initialized. Run 'scs init' first.")
        return 1

    calibrator = SeraphicCalibrator(config)
    calibrator.load_state()

    print("Seraphic Calibration Shell Status")
    print("=" * 50)
    print(f"Step count: {calibrator.step_count}")
    print("\nCurrent configuration:")
    print(f"  Algorithm: {calibrator.current_config.algorithm}")
    print(
        f"  Ansatz: {calibrator.current_config.ansatz_type} (depth {calibrator.current_config.ansatz_depth})"
    )
    print(
        f"  Optimizer: {calibrator.current_config.optimizer} (lr {calibrator.current_config.learning_rate})"
    )

    print("\nCurrent performance:")
    print(f"  ψ (quality):     {calibrator.current_performance.psi:.4f}")
    print(f"  ρ (stability):   {calibrator.current_performance.rho:.4f}")
    print(f"  ω (efficiency):  {calibrator.current_performance.omega:.4f}")
    print(f"  Harmonic mean:   {calibrator.current_performance.harmonic_mean():.4f}")

    print("\nCRI diagnostics:")
    diagnostics = calibrator.cri.get_diagnostics()
    print(f"  Steps since impulse: {diagnostics['steps_since_impulse']}")
    print(f"  Current J(t): {diagnostics['current_j_t']:.4f}")
    print(f"  Stagnating: {diagnostics['is_stagnating']}")
    print(f"  Degrading: {diagnostics['is_degrading']}")

    return 0


def cmd_export(args) -> int:
    """Export current configuration in various formats."""
    config = CalibratorConfig(state_file=args.state_file)

    if not Path(config.state_file).exists():
        print("SCS not initialized. Run 'scs init' first.")
        return 1

    calibrator = SeraphicCalibrator(config)
    calibrator.load_state()

    best_config = calibrator.get_best_configuration()

    # Export to specified file
    output_path = args.output or "scs_config_export.json"
    best_config.to_file(output_path)
    print(f"Exported configuration to {output_path}")

    # Also print to stdout if requested
    if args.stdout:
        print("\nConfiguration:")
        print(best_config.to_json())

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Seraphic Calibration Shell (SCS) for Q⊗DASH (Metatron VM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize SCS
  python -m scs.cli init

  # Run 5 calibration steps
  python -m scs.cli step -n 5

  # Check status
  python -m scs.cli status

  # Export best configuration
  python -m scs.cli export -o best_config.json
        """,
    )

    # Global options
    parser.add_argument(
        "--benchmark-dir",
        default="metatron-qso-rs/ci",
        help="Directory containing benchmark JSON files",
    )
    parser.add_argument(
        "--state-file", default="scs_state.json", help="State file path"
    )
    parser.add_argument(
        "--history-file", default="scs_history.json", help="History file path"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Init command
    parser_init = subparsers.add_parser("init", help="Initialize SCS")
    parser_init.add_argument("--config-file", help="Initial configuration file (JSON)")
    parser_init.add_argument(
        "-o", "--output", help="Output path for best configuration"
    )
    parser_init.set_defaults(func=cmd_init)

    # Step command
    parser_step = subparsers.add_parser("step", help="Run calibration steps")
    parser_step.add_argument(
        "-n",
        "--num-steps",
        type=int,
        default=1,
        help="Number of calibration steps to run",
    )
    parser_step.add_argument(
        "-o", "--output", help="Output path for best configuration"
    )
    parser_step.set_defaults(func=cmd_step)

    # Status command
    parser_status = subparsers.add_parser("status", help="Show SCS status")
    parser_status.set_defaults(func=cmd_status)

    # Export command
    parser_export = subparsers.add_parser("export", help="Export configuration")
    parser_export.add_argument("-o", "--output", help="Output file path")
    parser_export.add_argument(
        "--stdout", action="store_true", help="Also print to stdout"
    )
    parser_export.set_defaults(func=cmd_export)

    # Parse and execute
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if "--debug" in sys.argv:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
