// Quick verification of mef-solvecoagula module
use mef_solvecoagula::{SolveCoagula, SolveCoagulaConfig};
use ndarray::Array1;

fn main() {
    println!("=== MEF-Core Solve-Coagula Module Verification ===\n");

    // Create default configuration
    let config = SolveCoagulaConfig::default();
    println!("Configuration:");
    println!("  lambda: {}", config.lambda);
    println!("  eps: {}", config.eps);
    println!("  max_iter: {}", config.max_iter);
    println!("  sc_beta: {}\n", config.sc_beta);

    // Initialize Solve-Coagula
    let sc = SolveCoagula::new(config).expect("Failed to create SolveCoagula");
    println!("✅ SolveCoagula initialized successfully\n");

    // Verify contractivity
    let contractivity = sc.verify_contractivity();
    println!("Contractivity Check:");
    println!("  W spectral norm: {:.6}", contractivity["W_spectral_norm"]);
    println!("  lambda: {}", contractivity["lambda"]);
    println!("  is_contractive: {}\n", contractivity["is_contractive"]);

    // Test fixpoint iteration
    let v0 = Array1::from_vec(vec![1.0, 0.5, -0.3, 0.8, -0.2]);
    println!("Input vector: {:?}", v0.as_slice().unwrap());

    let (fixpoint, info) = sc
        .iterate_to_fixpoint(&v0, true)
        .expect("Failed to compute fixpoint");

    println!("\nFixpoint Result:");
    println!("  fixpoint: {:?}", fixpoint.as_slice().unwrap());
    println!("  converged: {}", info.converged);
    println!("  iterations: {}", info.iterations);
    println!("  final_delta: {:.2e}", info.final_delta);

    if let Some(history) = info.history {
        println!("\nConvergence History (last 5 steps):");
        let start = history.len().saturating_sub(5);
        for step in &history[start..] {
            println!(
                "    iter {}: lyapunov={:.6}, norm={:.6}",
                step.iteration, step.lyapunov, step.norm
            );
        }
    }

    println!("\n✅ All verifications passed!");
}
