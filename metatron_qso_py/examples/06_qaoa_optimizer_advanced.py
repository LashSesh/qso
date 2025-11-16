#!/usr/bin/env python3
"""
Advanced QAOA MaxCut Optimizer Demo

This example demonstrates the advanced QAOA MaxCut solver with full control
over optimization parameters, including deterministic results via random seed.

Use Cases:
- Network partitioning / community detection
- Load balancing across servers
- Circuit design (minimizing wire crossings)
- Resource allocation problems
"""

import metatron_qso

def main():
    print("=" * 60)
    print("Advanced QAOA MaxCut Optimizer")
    print("=" * 60)
    print()

    # Create the Metatron Cube graph
    graph = metatron_qso.MetatronGraph()
    print(f"Optimizing graph: {graph}")
    print()

    print("MaxCut Problem:")
    print("  Goal: Partition nodes into two sets to maximize edges between them")
    print("  Method: QAOA (Quantum Approximate Optimization Algorithm)")
    print()

    # Run with different configurations
    configs = [
        {"depth": 1, "max_iters": 50, "seed": 42, "name": "Quick (p=1)"},
        {"depth": 2, "max_iters": 100, "seed": 42, "name": "Standard (p=2)"},
        {"depth": 3, "max_iters": 150, "seed": 42, "name": "High-quality (p=3)"},
    ]

    results = []

    for config in configs:
        print(f"Running {config['name']}...")
        print(f"  Depth (p):      {config['depth']}")
        print(f"  Max iterations: {config['max_iters']}")
        print(f"  Random seed:    {config['seed']}")
        print()

        result = metatron_qso.solve_maxcut_qaoa_advanced(
            graph,
            depth=config["depth"],
            max_iters=config["max_iters"],
            seed=config["seed"]
        )

        results.append((config["name"], result))

        # Display results
        print(f"Results for {config['name']}:")
        print("-" * 60)
        print(f"  Cut value:           {result['cut_value']:.4f}")
        print(f"  Approximation ratio: {result['approximation_ratio']:.4f}")
        print(f"  Converged:           {result['meta']['converged']}")
        print(f"  Iterations used:     {result['meta']['iterations']}")
        print(f"  Final cost:          {result['meta']['final_cost']:.6f}")

        # Show partition
        assignment = result['assignment']
        set_0 = [i for i, val in enumerate(assignment) if not val]
        set_1 = [i for i, val in enumerate(assignment) if val]

        print(f"  Partition sizes:     {result['meta']['partition_sizes']}")
        print(f"  Set 0: {set_0}")
        print(f"  Set 1: {set_1}")
        print()

    # Comparison
    print("=" * 60)
    print("Configuration Comparison:")
    print("=" * 60)
    print()
    print(f"{'Config':<20} {'Cut Value':<12} {'Approx Ratio':<15} {'Iterations':<12} {'Quality':<10}")
    print("-" * 75)

    for name, result in results:
        quality = "EXCELLENT" if result['approximation_ratio'] > 0.95 else \
                 "GOOD" if result['approximation_ratio'] > 0.8 else \
                 "FAIR" if result['approximation_ratio'] > 0.6 else "POOR"

        print(f"{name:<20} {result['cut_value']:<12.4f} {result['approximation_ratio']:<15.4f} "
              f"{result['meta']['iterations']:<12} {quality:<10}")

    print()

    # Best result
    best_result = max(results, key=lambda x: x[1]['cut_value'])
    print(f"Best Result: {best_result[0]}")
    print(f"  Cut value: {best_result[1]['cut_value']:.4f}")
    print(f"  Approximation ratio: {best_result[1]['approximation_ratio']:.4f}")
    print()

    # Deterministic behavior
    print("=" * 60)
    print("Deterministic Behavior (same seed):")
    print("=" * 60)
    print()

    # Run twice with same seed
    result1 = metatron_qso.solve_maxcut_qaoa_advanced(graph, depth=2, max_iters=100, seed=123)
    result2 = metatron_qso.solve_maxcut_qaoa_advanced(graph, depth=2, max_iters=100, seed=123)

    print(f"Run 1 - Cut value: {result1['cut_value']:.6f}")
    print(f"Run 2 - Cut value: {result2['cut_value']:.6f}")
    print(f"Difference: {abs(result1['cut_value'] - result2['cut_value']):.10f}")
    print()

    if abs(result1['cut_value'] - result2['cut_value']) < 1e-6:
        print("✓ Results are deterministic (same seed produces same result)")
    else:
        print("Note: Small numerical variations may occur due to optimization")

    print()

    # Performance guidance
    print("=" * 60)
    print("Performance Guidance:")
    print("=" * 60)
    print()
    print("  Depth (p):")
    print("    • p=1: Fastest, reasonable for quick estimates")
    print("    • p=2-3: Good balance of quality and speed")
    print("    • p≥4: Diminishing returns, much slower")
    print()
    print("  Max iterations:")
    print("    • 50-100: Quick optimization")
    print("    • 100-200: Standard (recommended)")
    print("    • >200: High-precision (check convergence)")
    print()
    print("  Random seed:")
    print("    • Use for reproducible results")
    print("    • Omit (None) for non-deterministic runs")
    print("    • Try multiple seeds to find best solution")
    print()

    print("Real-World Applications:")
    print("-" * 60)
    print("  1. Network Partitioning:")
    print("     - Divide network into balanced communities")
    print("     - Minimize inter-community traffic")
    print()
    print("  2. Load Balancing:")
    print("     - Partition tasks across servers")
    print("     - Maximize independent operations")
    print()
    print("  3. Circuit Design:")
    print("     - Partition logic blocks")
    print("     - Minimize wire crossings")
    print()

    print("=" * 60)
    print("Advanced QAOA optimization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
