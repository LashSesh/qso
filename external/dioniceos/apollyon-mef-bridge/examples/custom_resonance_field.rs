//! Example: Custom Resonance Field
//!
//! Demonstrates how to create and use a custom resonance field
//! with the UnifiedCognitiveEngine.

use apollyon_mef_bridge::unified::{CognitiveInput, UnifiedCognitiveEngine};
use bridge::ResonanceField;
use core_5d::{State5D, SystemParameters};

/// Custom time-dependent resonance field with exponential decay
#[derive(Debug, Clone)]
struct ExponentialDecayField {
    initial_strength: f64,
    decay_rate: f64,
}

impl ExponentialDecayField {
    fn new(initial_strength: f64, decay_rate: f64) -> Self {
        Self {
            initial_strength,
            decay_rate,
        }
    }
}

impl ResonanceField for ExponentialDecayField {
    fn modulation(&self, t: f64, _node_i: usize, _node_j: usize) -> f64 {
        self.initial_strength * (-self.decay_rate * t).exp()
    }
}

/// Custom node-specific resonance field
#[derive(Debug, Clone)]
struct NodeSpecificField {
    coupling_strengths: [[f64; 5]; 5],
}

impl NodeSpecificField {
    fn new() -> Self {
        // Define custom coupling strengths between nodes
        let mut strengths = [[1.0; 5]; 5];

        // Stronger coupling between adjacent dimensions
        for i in 0..4 {
            strengths[i][i + 1] = 1.5;
            strengths[i + 1][i] = 1.5;
        }

        // Weaker coupling between distant dimensions
        strengths[0][4] = 0.5;
        strengths[4][0] = 0.5;

        Self {
            coupling_strengths: strengths,
        }
    }
}

impl ResonanceField for NodeSpecificField {
    fn modulation(&self, _t: f64, node_i: usize, node_j: usize) -> f64 {
        if node_i < 5 && node_j < 5 {
            self.coupling_strengths[node_i][node_j]
        } else {
            1.0
        }
    }
}

fn main() {
    println!("=== Custom Resonance Field Examples ===\n");

    // Example 1: Exponential Decay Field
    println!("1. Exponential Decay Field");
    println!("   - Initial strength: 1.0");
    println!("   - Decay rate: 0.5");
    println!();

    let decay_field = Box::new(ExponentialDecayField::new(1.0, 0.5));
    let mut engine1 = UnifiedCognitiveEngine::new_with_field(decay_field);

    let input1 = CognitiveInput {
        initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
        parameters: SystemParameters::default(),
        t_final: 2.0,
        tic_id: "TIC-DECAY-001".to_string(),
        seed: "decay_seed".to_string(),
        seed_path: "MEF/examples/decay/0001".to_string(),
    };

    match engine1.process(input1) {
        Ok(output) => {
            println!("   ✓ Processing completed successfully");
            println!("   - Trajectory length: {}", output.trajectory.len());
            println!("   - Gate decision: {:?}", output.gate_decision);
            println!("   - PoR δPI: {:.6}", output.proof.delta_pi);
            println!("   - PoR Φ: {:.6}", output.proof.phi);
        }
        Err(e) => println!("   ✗ Error: {}", e),
    }
    println!();

    // Example 2: Node-Specific Field
    println!("2. Node-Specific Field");
    println!("   - Varies coupling by node pairs");
    println!("   - Adjacent dimensions: 1.5x coupling");
    println!("   - Distant dimensions: 0.5x coupling");
    println!();

    let node_field = Box::new(NodeSpecificField::new());
    let mut engine2 = UnifiedCognitiveEngine::new_with_field(node_field);

    let input2 = CognitiveInput {
        initial_state: State5D::new(0.8, 0.6, 0.4, 0.3, 0.2),
        parameters: SystemParameters::default(),
        t_final: 1.5,
        tic_id: "TIC-NODE-001".to_string(),
        seed: "node_seed".to_string(),
        seed_path: "MEF/examples/node/0001".to_string(),
    };

    match engine2.process(input2) {
        Ok(output) => {
            println!("   ✓ Processing completed successfully");
            println!("   - Trajectory length: {}", output.trajectory.len());
            println!("   - Gate decision: {:?}", output.gate_decision);
            println!("   - PoR δPI: {:.6}", output.proof.delta_pi);
            println!("   - PoR Φ: {:.6}", output.proof.phi);
        }
        Err(e) => println!("   ✗ Error: {}", e),
    }
    println!();

    // Example 3: Comparing fields
    println!("3. Comparing Standard vs Custom Fields");
    println!();

    use bridge::ConstantResonanceField;

    let standard_field = Box::new(ConstantResonanceField::new(0.8));
    let mut standard_engine = UnifiedCognitiveEngine::new_with_field(standard_field);

    let test_input = CognitiveInput {
        initial_state: State5D::new(1.0, 0.5, 0.3, 0.2, 0.1),
        parameters: SystemParameters::default(),
        t_final: 1.0,
        tic_id: "TIC-COMPARE-001".to_string(),
        seed: "compare_seed".to_string(),
        seed_path: "MEF/examples/compare/0001".to_string(),
    };

    let standard_result = standard_engine.process(test_input.clone()).unwrap();

    let custom_field = Box::new(ExponentialDecayField::new(1.0, 0.3));
    let mut custom_engine = UnifiedCognitiveEngine::new_with_field(custom_field);
    let custom_result = custom_engine.process(test_input).unwrap();

    println!("   Standard Field (constant 0.8):");
    println!("   - PoR Φ: {:.6}", standard_result.proof.phi);
    println!("   - Gate: {:?}", standard_result.gate_decision);
    println!();
    println!("   Custom Field (exponential decay):");
    println!("   - PoR Φ: {:.6}", custom_result.proof.phi);
    println!("   - Gate: {:?}", custom_result.gate_decision);

    println!("\n=== Complete! ===");
}
