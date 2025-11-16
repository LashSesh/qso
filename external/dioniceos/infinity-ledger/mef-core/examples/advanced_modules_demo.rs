/*!
 * Advanced MEF-Core Modules Example
 *
 * This example demonstrates the four advanced modules added to mef-core:
 * - MandorlaField: Global decision and resonance field
 * - GabrielCell: Modular feedback resonator
 * - QLogicEngine: Quantum logic and spectral processing
 * - ResonanceTensorField: 3D multidimensional resonance dynamics
 */

use mef_core::{
    couple_cells, neighbor_feedback, GabrielCell, MandorlaField, QLogicEngine, ResonanceTensorField,
};
use ndarray::{array, Array3};
use std::f64::consts::PI;

fn main() {
    println!("=== Advanced MEF-Core Modules Demo ===\n");

    // ===================================================================
    // 1. MandorlaField - Global Decision and Resonance Field
    // ===================================================================
    println!("=== 1. MandorlaField Demo ===");

    let mut mandorla = MandorlaField::new(0.7, 0.5, 0.5);
    println!(
        "✓ Created MandorlaField (threshold: {}, α: {}, β: {})",
        mandorla.threshold, mandorla.alpha, mandorla.beta
    );

    // Add some input vectors
    mandorla.add_input(array![1.0, 2.0, 3.0, 4.0]);
    mandorla.add_input(array![1.1, 2.1, 3.1, 4.1]);
    mandorla.add_input(array![1.2, 1.9, 3.2, 3.9]);
    println!("✓ Added {} input vectors", mandorla.inputs.len());

    // Calculate resonance
    let resonance = mandorla.calc_resonance();
    println!("  Resonance: {:.4}", resonance);

    // Calculate entropy and variance
    let entropy = mandorla.calc_entropy();
    let variance = mandorla.calc_variance();
    println!("  Entropy: {:.4}", entropy);
    println!("  Variance: {:.4}", variance);

    // Check decision trigger
    let triggered = mandorla.decision_trigger();
    println!("  Dynamic threshold θ(t): {:.4}", mandorla.current_theta);
    println!("  Decision triggered: {}", triggered);

    // ===================================================================
    // 2. GabrielCell - Modular Feedback Resonator
    // ===================================================================
    println!("\n=== 2. GabrielCell Demo ===");

    // Create a network of cells
    let mut cells = vec![
        GabrielCell::new(1.0, 1.0, 1.0, 0.15),
        GabrielCell::new(0.8, 1.2, 0.9, 0.15),
        GabrielCell::new(1.2, 0.9, 1.1, 0.15),
    ];
    println!("✓ Created {} Gabriel cells", cells.len());

    // Couple cells in a chain: 0 <-> 1 <-> 2
    couple_cells(&mut cells, 0, 1);
    couple_cells(&mut cells, 1, 2);
    println!("✓ Coupled cells in chain topology");

    // Activate cells with inputs
    for (i, cell) in cells.iter_mut().enumerate() {
        let input = 0.5 + (i as f64) * 0.2;
        cell.activate(Some(input));
        println!(
            "  Cell {} activated: psi={:.3}, output={:.3}",
            i, cell.psi, cell.output
        );
    }

    // Apply neighbor feedback
    println!("\n  Applying neighbor feedback:");
    neighbor_feedback(&mut cells, 1); // Cell 1 learns from neighbors 0 and 2
    println!(
        "  Cell 1 after feedback: psi={:.3}, rho={:.3}, omega={:.3}",
        cells[1].psi, cells[1].rho, cells[1].omega
    );

    // Apply direct feedback to cell 0
    cells[0].feedback(1.5);
    println!("  Cell 0 after feedback(1.5): psi={:.3}", cells[0].psi);

    // ===================================================================
    // 3. QLogicEngine - Quantum Logic and Spectral Processing
    // ===================================================================
    println!("\n=== 3. QLogicEngine Demo ===");

    let engine = QLogicEngine::new(16);
    println!(
        "✓ Created QLogicEngine with {} nodes",
        engine.osc_core.num_nodes
    );

    // Run multiple time steps
    println!("\n  Time evolution:");
    for t in [0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI] {
        let result = engine.step(t);
        println!(
            "  t={:.4}π: entropy={:.4}, centroid={:.2}, sparsity={:.4}",
            t / PI,
            result.entropy,
            result.spectral_centroid.unwrap_or(0.0),
            result.sparsity.unwrap_or(0.0)
        );
    }

    // Show field details for one step
    let result = engine.step(0.5);
    println!("\n  Field at t=0.5:");
    println!(
        "    Field values: [{:.3}, {:.3}, {:.3}, ...]",
        result.field[0], result.field[1], result.field[2]
    );
    println!(
        "    Spectrum magnitudes: [{:.3}, {:.3}, {:.3}, ...]",
        result.spectrum[0], result.spectrum[1], result.spectrum[2]
    );

    // ===================================================================
    // 4. ResonanceTensorField - 3D Multidimensional Resonance
    // ===================================================================
    println!("\n=== 4. ResonanceTensorField Demo ===");

    let mut tensor = ResonanceTensorField::new(
        (4, 4, 4), // 4x4x4 grid
        1.0,       // amplitude
        2.0,       // frequency
        0.0,       // initial phase
        1e-3,      // gradient threshold
    );
    println!(
        "✓ Created {}x{}x{} ResonanceTensorField",
        tensor.shape.0, tensor.shape.1, tensor.shape.2
    );

    // Get initial state
    let state0 = tensor.get_state();
    println!("  Initial state at (0,0,0): {:.4}", state0[[0, 0, 0]]);

    // Evolve the field
    println!("\n  Time evolution:");
    for step in 1..=5 {
        let dt = 0.1;
        let state = tensor.step(dt, None);
        let coherence = tensor.coherence();
        let grad_norm = tensor.gradient_norm();
        let singularity = tensor.detect_singularity();

        println!(
            "  Step {}: t={:.2}, state(0,0,0)={:.4}, coherence={:.4}, grad={:.6}, sing={}",
            step,
            tensor.time,
            state[[0, 0, 0]],
            coherence,
            grad_norm,
            singularity
        );
    }

    // Apply input modulation
    println!("\n  Applying phase modulation:");
    let mut modulation = Array3::zeros((4, 4, 4));
    modulation[[2, 2, 2]] = PI / 4.0; // Perturb center
    tensor.step(0.1, Some(&modulation));
    println!("  ✓ Applied modulation at (2,2,2)");

    let state_after = tensor.get_state();
    println!(
        "  State after modulation at (2,2,2): {:.4}",
        state_after[[2, 2, 2]]
    );

    // Set custom parameters for specific cells
    tensor.set_amplitude(0, 0, 0, 2.0);
    tensor.set_frequency(1, 1, 1, 3.0);
    println!("  ✓ Customized cell parameters");
    println!(
        "  Amplitude at (0,0,0): {:.1}",
        tensor.get_amplitude()[[0, 0, 0]]
    );
    println!(
        "  Frequency at (1,1,1): {:.1}",
        tensor.get_frequency()[[1, 1, 1]]
    );

    // ===================================================================
    // Integration Example: Combining Multiple Modules
    // ===================================================================
    println!("\n=== 5. Integration Example ===");

    // Use QLogic to generate patterns for Mandorla
    let mut integrated_mandorla = MandorlaField::default();
    let qlogic = QLogicEngine::new(8);

    println!("✓ Feeding QLogic patterns into MandorlaField:");
    for t in [0.0, 0.5, 1.0, 1.5] {
        let result = qlogic.step(t);
        integrated_mandorla.add_input(result.field.clone());
    }

    let final_resonance = integrated_mandorla.calc_resonance();
    println!(
        "  Added {} QLogic patterns",
        integrated_mandorla.inputs.len()
    );
    println!("  Combined resonance: {:.4}", final_resonance);

    // Use ResonanceTensor coherence to modulate GabrielCells
    let mut tensor2 = ResonanceTensorField::new((3, 3, 3), 1.0, 1.5, 0.0, 1e-3);
    tensor2.step(0.2, None);
    let tensor_coherence = tensor2.coherence();

    let mut cell = GabrielCell::default();
    cell.activate(Some(tensor_coherence));
    println!(
        "\n✓ Used ResonanceTensor coherence ({:.4}) to modulate GabrielCell",
        tensor_coherence
    );
    println!("  Cell output: {:.4}", cell.output);

    println!("\n=== Demo Complete ===");
    println!("\nSummary:");
    println!("  • MandorlaField: Global resonance decision system");
    println!("  • GabrielCell: Adaptive feedback resonator network");
    println!("  • QLogicEngine: Spectral analysis and entropy measurement");
    println!("  • ResonanceTensorField: 3D oscillatory field dynamics");
    println!("\nAll modules working together for advanced MEF processing! ✓");
}
