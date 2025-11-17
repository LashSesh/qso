#!/usr/bin/env python3
"""
VQE Ground State Example

This example demonstrates how to use VQE (Variational Quantum Eigensolver)
to find the ground state energy of the Metatron Hamiltonian.
"""

import metatron_qso


def main():
    print("=" * 60)
    print("Metatron QSO - VQE Ground State Demo")
    print("=" * 60)
    print()

    # Create the Metatron Cube graph
    print("Creating Metatron Cube graph...")
    graph = metatron_qso.MetatronGraph()
    print(f"  {graph}")
    print()

    # Explain VQE
    print("VQE (Variational Quantum Eigensolver):")
    print("  Goal: Find the ground state energy of a quantum system")
    print("  Method: Variational optimization with parameterized circuits")
    print()

    # Test different ansätze
    ansatz_types = [
        ("hardware_efficient", "Hardware-Efficient Ansatz"),
        ("metatron", "Metatron-Optimized Ansatz"),
        ("efficient_su2", "Efficient SU(2) Ansatz"),
    ]

    results = []

    for ansatz_type, ansatz_name in ansatz_types:
        print(f"Running VQE with {ansatz_name}...")
        print(f"  Ansatz: {ansatz_type}")
        print("  Depth: 2")
        print("  Max iterations: 150")
        print()

        result = metatron_qso.run_vqe(
            graph=graph, depth=2, max_iters=150, ansatz_type=ansatz_type
        )

        results.append((ansatz_name, result))

        # Display results
        print(f"Results ({ansatz_name}):")
        print(f"  Ground state energy (VQE): {result['ground_state_energy']:.10f}")
        print(f"  Classical ground energy:   {result['classical_ground_energy']:.10f}")
        print(f"  Absolute error:            {result['error']:.10e}")
        print(
            f"  Relative error:            {(result['error'] / abs(result['classical_ground_energy']) * 100):.6f}%"
        )
        print(f"  Iterations:                {result['iterations']}")
        print()

        # Show first few probabilities
        final_probs = result["final_state"]
        print("  Final state (first 5 nodes):")
        for i in range(min(5, len(final_probs))):
            print(f"    Node {i}: P = {final_probs[i]:.6f}")
        print()
        print("-" * 60)
        print()

    # Comparison
    print("=" * 60)
    print("Ansatz Comparison:")
    print()

    print(f"{'Ansatz':<30} {'Energy':>15} {'Error':>15} {'Iters':>8}")
    print("-" * 70)
    for name, result in results:
        print(
            f"{name:<30} {result['ground_state_energy']:>15.10f} "
            f"{result['error']:>15.10e} {result['iterations']:>8}"
        )
    print()

    # Find best result
    best_idx = min(range(len(results)), key=lambda i: results[i][1]["error"])
    best_name, best_result = results[best_idx]
    print(f"Best Result: {best_name}")
    print(f"  Energy: {best_result['ground_state_energy']:.10f}")
    print(f"  Error: {best_result['error']:.10e}")
    print()

    print("=" * 60)
    print("VQE Ground State Computation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
