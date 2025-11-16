//! VQA Suite Demonstration
//!
//! This example demonstrates all three VQA algorithms:
//! - VQE (Variational Quantum Eigensolver)
//! - QAOA (Quantum Approximate Optimization Algorithm)
//! - VQC (Variational Quantum Classifier)

use metatron_qso::prelude::*;
use std::sync::Arc;

fn main() {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║   VQA SUITE DEMONSTRATION - Metatron QSO              ║");
    println!("║   Variational Quantum Algorithms in Pure Rust         ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // Demo 1: VQE for Ground State Energy
    demo_vqe();

    println!("\n────────────────────────────────────────────────────────\n");

    // Demo 2: QAOA for Optimization
    demo_qaoa();

    println!("\n────────────────────────────────────────────────────────\n");

    // Demo 3: VQC for Classification
    demo_vqc();

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║   VQA SUITE DEMO COMPLETE                              ║");
    println!("╚════════════════════════════════════════════════════════╝\n");
}

/// Demonstrate VQE (Variational Quantum Eigensolver)
fn demo_vqe() {
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║   DEMO 1: VQE - GROUND STATE ENERGY FINDER            ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    // Create Metatron system
    let graph = MetatronGraph::new();
    let params = QSOParameters::default();
    let hamiltonian = Arc::new(MetatronHamiltonian::new(&graph, &params));

    println!("System Configuration:");
    println!("  • Metatron Cube: 13 nodes, 30 edges");
    println!("  • Hilbert Space Dimension: 13");
    println!("  • Hamiltonian: Tight-binding on Metatron graph");
    println!();

    // Run VQE with different ansätze
    println!("Running VQE with Hardware-Efficient Ansatz...\n");

    let vqe = VQEBuilder::new()
        .hamiltonian(hamiltonian.clone())
        .ansatz_type(AnsatzType::HardwareEfficient)
        .ansatz_depth(2)
        .optimizer(OptimizerType::Adam)
        .max_iterations(100)
        .learning_rate(0.01)
        .tolerance(1e-6)
        .verbose(true)
        .build();

    let result = vqe.run();

    println!("\n✓ VQE Verification:");
    if vqe.verify_result(&result) {
        println!("  • Result quality: PASSED");
        println!("  • Ground state energy within expected bounds");
        println!("  • State normalization: OK");
    } else {
        println!("  ⚠ Some verification checks failed");
    }

    // Compare with Metatron ansatz
    println!("\n\nRunning VQE with Metatron-Optimized Ansatz...\n");

    let vqe_metatron = VQEBuilder::new()
        .hamiltonian(hamiltonian)
        .ansatz_type(AnsatzType::Metatron)
        .ansatz_depth(1)
        .optimizer(OptimizerType::Adam)
        .max_iterations(100)
        .learning_rate(0.01)
        .verbose(true)
        .build();

    let result_metatron = vqe_metatron.run();

    println!("\n✓ Comparison:");
    println!(
        "  • Hardware-Efficient: E = {:.10}",
        result.ground_state_energy
    );
    println!(
        "  • Metatron-Optimized: E = {:.10}",
        result_metatron.ground_state_energy
    );
    println!(
        "  • Relative difference: {:.6}%",
        ((result.ground_state_energy - result_metatron.ground_state_energy).abs()
            / result.classical_ground_energy.abs())
            * 100.0
    );
}

/// Demonstrate QAOA (Quantum Approximate Optimization)
fn demo_qaoa() {
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║   DEMO 2: QAOA - COMBINATORIAL OPTIMIZATION           ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    println!("Problem: MaxCut on a small graph");
    println!("  • Graph: Triangle (3 nodes, 3 edges)");
    println!("  • Objective: Maximize cut size");
    println!();

    // Create MaxCut problem on triangle graph
    let edges = vec![(0, 1), (1, 2), (2, 0)];
    let cost_hamiltonian = Arc::new(metatron_qso::vqa::qaoa::create_maxcut_hamiltonian(&edges));

    println!("Running QAOA with depth p=3...\n");

    let qaoa = QAOABuilder::new()
        .cost_hamiltonian(cost_hamiltonian)
        .depth(3)
        .optimizer(OptimizerType::NelderMead)
        .max_iterations(200)
        .verbose(true)
        .build();

    let result = qaoa.run();

    println!("\n✓ QAOA Performance:");
    println!("  • Approximation ratio: {:.4}", result.approximation_ratio);
    if result.approximation_ratio > 0.5 {
        println!("  • Quality: GOOD (better than random)");
    } else {
        println!("  • Quality: POOR (not better than random)");
    }

    // Sample solutions
    println!("\n✓ Solution Sampling:");
    let (mean_cost, std_dev, _costs) = qaoa.analyze_samples(&result.optimal_state, 100);
    println!("  • Mean cost: {:.6}", mean_cost);
    println!("  • Std deviation: {:.6}", std_dev);
}

/// Demonstrate VQC (Variational Quantum Classifier)
fn demo_vqc() {
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║   DEMO 3: VQC - QUANTUM MACHINE LEARNING              ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    println!("Problem: Binary classification on synthetic data");
    println!("  • Features: 4-dimensional");
    println!("  • Classes: 2 (binary)");
    println!("  • Training samples: 8");
    println!();

    // Generate synthetic training data
    let training_data = vec![
        // Class 0 samples
        vec![0.1, 0.1, 0.0, 0.0],
        vec![0.2, 0.15, 0.0, 0.0],
        vec![0.15, 0.2, 0.0, 0.0],
        vec![0.05, 0.1, 0.0, 0.0],
        // Class 1 samples
        vec![0.8, 0.85, 0.0, 0.0],
        vec![0.9, 0.9, 0.0, 0.0],
        vec![0.85, 0.8, 0.0, 0.0],
        vec![0.95, 0.85, 0.0, 0.0],
    ];

    let training_labels = vec![0, 0, 0, 0, 1, 1, 1, 1];

    println!("Building VQC with Hardware-Efficient Ansatz...\n");

    let mut vqc = VQCBuilder::new()
        .ansatz_type(AnsatzType::HardwareEfficient)
        .ansatz_depth(2)
        .encoding(metatron_qso::vqa::vqc::EncodingType::Angle)
        .optimizer(OptimizerType::Adam)
        .max_iterations(200)
        .learning_rate(0.02)
        .tolerance(1e-4)
        .verbose(true)
        .build();

    println!("Training VQC...\n");
    let training_result = vqc.train(training_data.clone(), training_labels.clone());

    println!("\n✓ Training Complete:");
    println!(
        "  • Final training accuracy: {:.2}%",
        training_result.training_accuracy * 100.0
    );
    println!("  • Final loss: {:.6}", training_result.training_loss);

    // Test predictions
    println!("\n✓ Testing Predictions:");
    let test_samples = vec![
        (vec![0.12, 0.13, 0.0, 0.0], 0, "Class 0 sample"),
        (vec![0.88, 0.87, 0.0, 0.0], 1, "Class 1 sample"),
    ];

    for (data, expected, description) in test_samples {
        let prediction = vqc.predict(&data);
        let correct = if prediction.predicted_class == expected {
            "✓"
        } else {
            "✗"
        };

        println!(
            "  {} {} → Predicted: {}, Confidence: {:.2}%",
            correct,
            description,
            prediction.predicted_class,
            prediction.confidence * 100.0
        );
    }

    // Evaluate on training set (should be high)
    let train_accuracy = vqc.evaluate(training_data, training_labels);
    println!(
        "\n✓ Final Training Set Accuracy: {:.2}%",
        train_accuracy * 100.0
    );
}
