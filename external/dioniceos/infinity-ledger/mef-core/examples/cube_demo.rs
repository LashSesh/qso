/*!
 * Cube Module Example - Demonstrating MetatronCube API
 *
 * This example demonstrates the high-level MetatronCube API, showing how to:
 * - Create a cube instance
 * - Query nodes and edges
 * - Work with platonic solids
 * - Apply symmetry operators
 * - Work with quantum states
 */

use mef_core::{MetatronCube, QuantumState};
use num_complex::Complex64;

fn main() {
    println!("=== Metatron Cube API Demo ===\n");

    // Create a default Metatron Cube
    let mut cube = MetatronCube::default();
    println!("✓ Created default Metatron Cube");
    println!("  Nodes: {}", cube.nodes.len());
    println!(
        "  Edges: {} (raw), {} (unique)",
        cube.edges.len(),
        cube.graph.num_edges()
    );
    println!("  Operators: {}", cube.operators.len());

    // Query nodes
    println!("\n=== Node Queries ===");
    let center = cube.get_node("1").unwrap();
    println!(
        "✓ Center node: {} (type: {})",
        center.label, center.node_type
    );
    println!("  Coordinates: {:?}", center.coordinates);
    println!("  Membership: {:?}", center.membership);

    let hexagons = cube.list_nodes(Some("hexagon"));
    println!("✓ Hexagon nodes: {}", hexagons.len());
    for node in hexagons.iter().take(3) {
        println!("  - {} at {:?}", node.label, node.coordinates);
    }

    // Query edges
    println!("\n=== Edge Queries ===");
    let edge = cube.get_edge_by_index(1).unwrap();
    println!(
        "✓ Edge 1: {} -> {} ({})",
        edge.from, edge.to, edge.edge_type
    );
    println!("  Label: {}", edge.label);

    let hex_edges = cube.list_edges(Some("hex"));
    println!("✓ Hexagon edges: {}", hex_edges.len());

    // Platonic solids
    println!("\n=== Platonic Solids ===");
    let solids = cube.list_solids();
    println!("✓ Available solids: {:?}", solids);

    if let Some(cube_nodes) = cube.get_solid_nodes("cube") {
        println!("✓ Cube nodes: {:?}", cube_nodes[0]);
    }

    if let Some(octahedron_nodes) = cube.get_solid_nodes("octahedron") {
        println!("✓ Octahedron nodes: {:?}", octahedron_nodes[0]);
    }

    // Symmetry operators
    println!("\n=== Symmetry Operators ===");
    if let Ok(c6_ops) = cube.enumerate_group("C6", None) {
        println!("✓ C6 group has {} operators", c6_ops.len());
        let op = &c6_ops[1]; // 60° rotation
        println!("  Example: {} (group: {:?})", op.id, op.group);
    }

    // Apply operator to adjacency
    if let Ok(rotated_adj) = cube.apply_operator_to_adjacency("C6_rot_0") {
        println!("✓ Applied identity rotation to adjacency matrix");
        println!("  Shape: {:?}", rotated_adj.shape());
    }

    // Quantum state operations
    println!("\n=== Quantum State Operations ===");
    let amplitudes = vec![Complex64::new(1.0, 0.0)];
    let state = QuantumState::new(amplitudes, true).unwrap();
    println!("✓ Created quantum state with 13 amplitudes");

    let probs = state.probabilities();
    println!("  Probability at center: {:.4}", probs[0]);

    if let Ok(qop) = cube.get_quantum_operator("C6_rot_60") {
        println!("✓ Created quantum operator from C6 rotation");
        println!("  Is unitary: {}", qop.is_unitary(1e-10));

        if let Ok(transformed) = cube.apply_operator_to_state("C6_rot_60", &state) {
            println!("✓ Applied operator to quantum state");
            let new_probs = transformed.probabilities();
            println!("  New probability at center: {:.4}", new_probs[0]);
        }
    }

    // Serialization
    println!("\n=== Serialization ===");
    let json = cube.serialize();
    println!("✓ Serialized cube to JSON ({} bytes)", json.len());
    println!("  Contains nodes: {}", json.contains("nodes"));
    println!("  Contains edges: {}", json.contains("edges"));
    println!("  Contains adjacency: {}", json.contains("adjacency"));
    println!("  Contains operators: {}", json.contains("operators"));

    // Validation
    println!("\n=== Validation ===");
    let valid_perm = vec![2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    println!(
        "✓ Valid permutation: {}",
        cube.validate_permutation(&valid_perm)
    );

    let invalid_perm = vec![1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    println!(
        "✓ Invalid permutation: {}",
        cube.validate_permutation(&invalid_perm)
    );

    let adj = cube.graph.get_adjacency_matrix();
    println!(
        "✓ Valid adjacency matrix: {}",
        cube.validate_adjacency(&adj)
    );

    println!("\n=== Demo Complete ===");
}
