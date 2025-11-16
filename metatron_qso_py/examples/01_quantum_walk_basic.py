#!/usr/bin/env python3
"""
Basic Quantum Walk Example

This example demonstrates how to run a continuous-time quantum walk
on the Metatron Cube graph using the Python SDK.
"""

import metatron_qso

def main():
    print("=" * 60)
    print("Metatron QSO - Quantum Walk Demo")
    print("=" * 60)
    print()

    # Create the Metatron Cube graph
    print("Creating Metatron Cube graph...")
    graph = metatron_qso.MetatronGraph()
    print(f"  {graph}")
    print(f"  Nodes: {graph.num_nodes()}")
    print(f"  Edges: {graph.num_edges()}")
    print()

    # Run quantum walk starting from central node (node 0)
    print("Running quantum walk...")
    print("  Initial state: Node 0 (central node)")
    print("  Evolution time: 0.0 -> 5.0")
    print("  Time step: 0.1")
    print()

    result = metatron_qso.run_quantum_walk(
        graph=graph,
        source_nodes=[0],
        t_max=5.0,
        dt=0.1
    )

    # Display results
    print("Quantum Walk Results:")
    print(f"  Number of time steps: {len(result['times'])}")
    print()

    # Show probability distribution at key times
    times_to_show = [0.0, 1.0, 2.5, 5.0]
    for t in times_to_show:
        # Find closest time index
        idx = min(range(len(result['times'])),
                 key=lambda i: abs(result['times'][i] - t))
        actual_t = result['times'][idx]
        probs = result['probabilities'][idx]

        print(f"Time t = {actual_t:.2f}:")
        # Show probabilities for central node and first few neighbors
        print(f"  Node 0 (center):  P = {probs[0]:.6f}")
        print(f"  Node 1:           P = {probs[1]:.6f}")
        print(f"  Node 2:           P = {probs[2]:.6f}")
        print(f"  Node 7:           P = {probs[7]:.6f}")
        print()

    # Final state analysis
    final_probs = result['final_state']
    print("Final State Analysis (t = 5.0):")
    print(f"  Total probability: {sum(final_probs):.6f} (should be 1.0)")
    print(f"  Max probability: {max(final_probs):.6f} at node {final_probs.index(max(final_probs))}")
    print(f"  Min probability: {min(final_probs):.6f} at node {final_probs.index(min(final_probs))}")
    print()

    # Calculate and display entropy (measure of spreading)
    import math
    entropy = -sum(p * math.log(p) if p > 0 else 0 for p in final_probs)
    max_entropy = math.log(graph.num_nodes())
    print(f"  Entropy: {entropy:.4f} / {max_entropy:.4f} (max)")
    print(f"  Spreading: {(entropy/max_entropy)*100:.2f}%")
    print()

    print("=" * 60)
    print("Quantum walk complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
