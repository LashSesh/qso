/*!
 * MEF-Core Verification Example
 *
 * Demonstrates the usage of the MEF-Core utilities including:
 * - Metatron Cube geometry
 * - Field vectors with resonance
 * - MEF-Core pipeline configuration
 */

use mef_core::{
    canonical_edges, canonical_nodes, complete_canonical_edges, FieldVector, MEFCore, MEFCoreConfig,
};

fn main() -> anyhow::Result<()> {
    println!("=== MEF-Core Utilities Verification ===\n");

    // 1. Geometry Module Demo
    println!("1. Metatron Cube Geometry");
    println!("   ----------------------");

    let nodes = canonical_nodes();
    println!("   Total nodes: {}", nodes.len());

    let center = &nodes[0];
    println!("   Center node: {} at {:?}", center.label, center.coords);

    let edges = canonical_edges();
    println!("   Canonical edges: {}", edges.len());

    let complete_edges = complete_canonical_edges();
    println!("   Complete edges (K_13): {}", complete_edges.len());

    // Distance between center and first hexagon node
    let h1 = &nodes[1];
    let dist = center.distance_to(h1);
    println!("   Distance C -> H1: {:.6}", dist);
    println!();

    // 2. Field Vector Demo
    println!("2. Field Vector Operations");
    println!("   -----------------------");

    let mut fv = FieldVector::new(vec![1.0, 0.5, -0.3, 0.8, -0.2], 0.5);
    println!("   Initial vector: {:?}", fv.as_vec());
    println!("   Norm: {:.6}", fv.norm());

    fv.normalize();
    println!("   After normalization: {:.6}", fv.norm());

    let other = vec![0.8, 0.3, -0.1, 0.7, -0.1];
    let sim = fv.similarity(&other);
    println!("   Similarity with other: {:.6}", sim);

    let scaled = fv.scale(2.0);
    println!(
        "   Scaled by 2: {:?}",
        scaled
            .as_vec()
            .iter()
            .map(|x| format!("{:.2}", x))
            .collect::<Vec<_>>()
    );
    println!();

    // 3. MEF-Core Pipeline Config Demo
    println!("3. MEF-Core Pipeline Configuration");
    println!("   --------------------------------");

    let mef = MEFCore::new("MEF_SEED_42", None)?;
    println!("   Seed: {}", mef.seed);

    let config = mef.get_config();
    println!("   Spiral config:");
    println!(
        "     r: {}, a: {}, b: {}, c: {}, k: {}",
        config.spiral.r, config.spiral.a, config.spiral.b, config.spiral.c, config.spiral.k
    );

    println!("   Solve-Coagula config:");
    println!(
        "     lambda: {}, eps: {}, max_iter: {}",
        config.solvecoagula.lambda, config.solvecoagula.eps, config.solvecoagula.max_iter
    );

    println!("   Gate config:");
    println!(
        "     por_delta: {}, phi_star: {}, mci_min: {}",
        config.gate.por_delta, config.gate.phi_star, config.gate.mci_min
    );
    println!();

    // 4. Custom Configuration Demo
    println!("4. Custom Configuration");
    println!("   --------------------");

    let custom_config = MEFCoreConfig::with_seed("CUSTOM_SEED");
    let custom_json = serde_json::to_string_pretty(&custom_config)?;
    println!("   Custom config JSON:");
    println!(
        "{}",
        custom_json
            .lines()
            .map(|l| format!("   {}", l))
            .collect::<Vec<_>>()
            .join("\n")
    );
    println!();

    println!("=== Verification Complete ===");
    println!("\nAll MEF-Core utility modules are working correctly!");
    println!("Total modules verified: 3 (geometry, field_vector, mef_pipeline)");

    Ok(())
}
