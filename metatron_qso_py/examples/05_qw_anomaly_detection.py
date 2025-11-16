#!/usr/bin/env python3
"""
Quantum Walk Anomaly Detection Demo

This example demonstrates how to detect structural anomalies in graphs
using quantum walk dynamics. By comparing a baseline graph to a modified
version, we can identify nodes that exhibit unusual behavior.

Use Cases:
- Network intrusion detection
- Infrastructure health monitoring
- Supply chain disruption detection
- Social network manipulation detection
"""

import metatron_qso

def main():
    print("=" * 60)
    print("Quantum Walk Anomaly Detection")
    print("=" * 60)
    print()

    # Create the baseline (normal) graph
    print("Creating baseline graph (normal Metatron Cube)...")
    base_graph = metatron_qso.MetatronGraph()
    print(f"  {base_graph}")
    print()

    # In a real scenario, current_graph would be different
    # Here we use the same graph for demonstration
    # (In practice, you'd modify edges/weights to simulate changes)
    print("Creating current graph (same as baseline for demo)...")
    current_graph = metatron_qso.MetatronGraph()
    print(f"  {current_graph}")
    print()

    print("Note: In real usage, current_graph would reflect")
    print("      current network state (potentially with changes)")
    print()

    # Compute anomaly scores
    print("Computing anomaly scores...")
    print("  (Comparing quantum walk dynamics between graphs)")
    print()

    anomaly_scores = metatron_qso.quantum_walk_anomaly_score(
        base_graph,
        current_graph,
        t_max=10.0,
        dt=0.1,
        samples=128
    )

    # Display results
    print("Anomaly Scores (higher = more anomalous):")
    print("-" * 60)
    print(f"{'Node':<6} {'Score':<12} {'Status':<15} {'Bar Chart':<25}")
    print("-" * 60)

    # Define threshold for anomaly detection
    threshold = 0.05  # Adjust based on sensitivity needs
    max_score = max(anomaly_scores) if max(anomaly_scores) > 0 else 1.0

    anomalous_nodes = []

    for node_id, score in enumerate(anomaly_scores):
        status = "ANOMALOUS" if score > threshold else "Normal"
        normalized_score = score / max_score if max_score > 0 else 0
        bar_length = int(normalized_score * 25)
        bar = "█" * bar_length

        print(f"{node_id:<6} {score:<12.6f} {status:<15} {bar}")

        if score > threshold:
            anomalous_nodes.append((node_id, score))

    print()

    # Summary
    print("Detection Summary:")
    print("-" * 60)
    print(f"  Total nodes analyzed:     {len(anomaly_scores)}")
    print(f"  Anomaly threshold:        {threshold:.6f}")
    print(f"  Anomalous nodes detected: {len(anomalous_nodes)}")
    print()

    if anomalous_nodes:
        print("Detected Anomalies (ranked by severity):")
        print("-" * 60)
        anomalous_nodes.sort(key=lambda x: x[1], reverse=True)
        for rank, (node_id, score) in enumerate(anomalous_nodes, 1):
            severity = "HIGH" if score > 0.1 else "MEDIUM" if score > threshold else "LOW"
            print(f"  {rank}. Node {node_id}: score={score:.6f}, severity={severity}")
    else:
        print("✓ No significant anomalies detected")
        print("  All nodes behave consistently with baseline")

    print()

    # Interpretation
    print("Interpretation Guide:")
    print("-" * 60)
    print("  • Anomaly score = |centrality_base - centrality_current|")
    print("  • High scores indicate structural changes affecting that node")
    print("  • Can detect:")
    print("    - Modified connectivity patterns")
    print("    - Unusual traffic/activity  ")
    print("    - Network topology changes")
    print("  • Threshold tuning:")
    print("    - Lower threshold = more sensitive (more false positives)")
    print("    - Higher threshold = less sensitive (may miss anomalies)")
    print()

    print("Real-World Application Example:")
    print("-" * 60)
    print("  1. Establish baseline from normal network operation")
    print("  2. Periodically compare current state to baseline")
    print("  3. Investigate nodes with high anomaly scores")
    print("  4. Update baseline as network evolves normally")
    print()

    print("=" * 60)
    print("Anomaly detection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
