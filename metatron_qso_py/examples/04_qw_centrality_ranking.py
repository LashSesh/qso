#!/usr/bin/env python3
"""
Quantum Walk Centrality - Node Ranking Demo

This example demonstrates how to use quantum walk centrality for node ranking.
Quantum walk centrality provides a quantum-mechanical perspective on node
importance based on wave-like propagation through the network.

Use Cases:
- Social network influence ranking
- Network infrastructure criticality analysis
- Resource allocation prioritization
"""

import metatron_qso

def main():
    print("=" * 60)
    print("Quantum Walk Centrality - Node Ranking")
    print("=" * 60)
    print()

    # Create the Metatron Cube graph
    graph = metatron_qso.MetatronGraph()
    print(f"Analyzing graph: {graph}")
    print()

    # Compute quantum walk centrality
    print("Computing quantum walk centrality...")
    print("  (This measures node importance via quantum walk dynamics)")
    print()

    centrality = metatron_qso.quantum_walk_centrality(
        graph,
        t_max=10.0,  # Evolution time
        dt=0.1,      # Time step
        samples=128  # Number of samples for averaging
    )

    print("Centrality Scores (normalized to [0, 1]):")
    print("-" * 60)

    # Create a list of (node_id, centrality_score) and sort by score
    node_scores = [(i, score) for i, score in enumerate(centrality)]
    node_scores.sort(key=lambda x: x[1], reverse=True)

    # Display rankings
    print(f"{'Rank':<6} {'Node':<6} {'Score':<10} {'Bar Chart':<30}")
    print("-" * 60)

    for rank, (node_id, score) in enumerate(node_scores, 1):
        bar_length = int(score * 30)  # Scale to 30 characters
        bar = "█" * bar_length
        print(f"{rank:<6} {node_id:<6} {score:<10.6f} {bar}")

    print()

    # Identify node types
    central_node = 0
    hexagon_nodes = list(range(1, 7))
    cube_nodes = list(range(7, 13))

    print("Node Type Analysis:")
    print("-" * 60)

    avg_central = centrality[central_node]
    avg_hexagon = sum(centrality[i] for i in hexagon_nodes) / len(hexagon_nodes)
    avg_cube = sum(centrality[i] for i in cube_nodes) / len(cube_nodes)

    print(f"  Central node (0):      {avg_central:.6f}")
    print(f"  Hexagon layer (avg):   {avg_hexagon:.6f}")
    print(f"  Cube layer (avg):      {avg_cube:.6f}")
    print()

    # Interpretation
    print("Interpretation:")
    print("-" * 60)
    print("  • Higher centrality = more 'central' in quantum walk dynamics")
    print("  • Central node typically has highest score (hub position)")
    print("  • Score reflects how often the quantum state visits each node")
    print("  • Useful for identifying critical nodes in networks")
    print()

    # Top 5 most central nodes
    print("Top 5 Most Central Nodes:")
    print("-" * 60)
    for rank, (node_id, score) in enumerate(node_scores[:5], 1):
        node_type = "Central" if node_id == 0 else \
                   ("Hexagon" if node_id in hexagon_nodes else "Cube")
        print(f"  {rank}. Node {node_id:2d} ({node_type:8s}): {score:.6f}")

    print()
    print("=" * 60)
    print("Centrality analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
