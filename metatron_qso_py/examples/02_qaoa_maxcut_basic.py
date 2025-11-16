#!/usr/bin/env python3
"""
QAOA MaxCut Example

This example demonstrates how to solve the MaxCut optimization problem
on the Metatron Cube graph using QAOA (Quantum Approximate Optimization Algorithm).
"""

import metatron_qso

def main():
    print("=" * 60)
    print("Metatron QSO - QAOA MaxCut Demo")
    print("=" * 60)
    print()

    # Create the Metatron Cube graph
    print("Creating Metatron Cube graph...")
    graph = metatron_qso.MetatronGraph()
    print(f"  {graph}")
    print()

    # Explain the MaxCut problem
    print("MaxCut Problem:")
    print("  Goal: Partition graph nodes into two sets to maximize")
    print("        the number of edges between the sets")
    print()

    # Solve with different QAOA depths
    depths = [1, 2, 3]

    for depth in depths:
        print(f"Running QAOA with depth p={depth}...")
        print(f"  Circuit depth: {depth}")
        print(f"  Max iterations: 100")
        print()

        result = metatron_qso.solve_maxcut_qaoa(
            graph=graph,
            depth=depth,
            max_iters=100
        )

        # Display results
        print(f"Results (depth p={depth}):")
        print(f"  Cut value: {result['cut_value']:.4f}")
        print(f"  Approximation ratio: {result['approximation_ratio']:.4f}")
        print(f"  Optimization iterations: {result['meta']['iterations']}")
        print()

        # Show the partition
        assignment = result['assignment']
        set_0 = [i for i, val in enumerate(assignment) if val == 0]
        set_1 = [i for i, val in enumerate(assignment) if val == 1]

        print(f"  Partition:")
        print(f"    Set 0: {set_0}")
        print(f"    Set 1: {set_1}")
        print()

        # Quality assessment
        if result['approximation_ratio'] > 0.95:
            quality = "EXCELLENT"
        elif result['approximation_ratio'] > 0.8:
            quality = "GOOD"
        elif result['approximation_ratio'] > 0.6:
            quality = "FAIR"
        else:
            quality = "POOR"

        print(f"  Quality: {quality}")
        print(f"  Mean cost: {result['meta']['mean_cost']:.4f}")
        print(f"  Std deviation: {result['meta']['std_dev']:.4f}")
        print()
        print("-" * 60)
        print()

    # Summary
    print("=" * 60)
    print("QAOA MaxCut Optimization Complete!")
    print()
    print("Key Observations:")
    print("  - Higher depth (p) generally gives better approximation")
    print("  - QAOA provides good solutions for combinatorial problems")
    print("  - The Metatron Cube's symmetry affects cut quality")
    print("=" * 60)


if __name__ == "__main__":
    main()
