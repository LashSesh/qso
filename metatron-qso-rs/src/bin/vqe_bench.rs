use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use metatron_qso::prelude::*;
use serde::{Deserialize, Serialize};

/// Generic benchmark metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub version: String,
    pub commit_hash: String,
    pub system_info: String,
}

/// VQE Benchmark Results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VQEBenchmarkSuite {
    pub metadata: BenchmarkMetadata,
    pub results: Vec<VQEBenchmarkResult>,
    pub performance_metrics: PerformanceMetrics,
    pub quality_metrics: QualityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VQEBenchmarkResult {
    pub ansatz_type: String,
    pub ansatz_depth: usize,
    pub optimizer: String,
    pub ground_energy: f64,
    pub classical_ground: f64,
    pub approximation_error: f64,
    pub iterations: usize,
    pub quantum_evaluations: usize,
    pub converged: bool,
    pub execution_time_ms: f64,
    pub final_gradient_norm: f64,
    pub quality_score: f64,
    pub num_random_starts: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_execution_time_ms: f64,
    pub avg_iteration_time_ms: f64,
    pub total_quantum_evaluations: usize,
    pub evaluations_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub best_ground_energy: f64,
    pub worst_ground_energy: f64,
    pub avg_ground_energy: f64,
    pub energy_variance: f64,
    pub convergence_rate: f64,
}

/// Calculate quality score based on energy accuracy and convergence
/// Score = convergence_factor * (1 - normalized_error)
/// where normalized_error = min(approximation_error / |classical_ground|, 1.0)
/// and convergence_factor = 1.0 if converged, 0.5 otherwise
fn calculate_quality_score(
    approximation_error: f64,
    classical_ground: f64,
    converged: bool,
) -> f64 {
    let normalized_error = (approximation_error / classical_ground.abs()).min(1.0);
    let energy_score = 1.0 - normalized_error;
    let convergence_factor = if converged { 1.0 } else { 0.5 };
    convergence_factor * energy_score
}

fn benchmark_vqe_ansatz(
    ansatz_type: AnsatzType,
    ansatz_name: &str,
    depth: usize,
    optimizer_type: OptimizerType,
    optimizer_name: &str,
    num_random_starts: usize,
    hamiltonian: std::sync::Arc<MetatronHamiltonian>,
) -> VQEBenchmarkResult {
    println!(
        "Benchmarking: {} (depth={}) with {} optimizer (starts={})...",
        ansatz_name, depth, optimizer_name, num_random_starts
    );

    let start = Instant::now();

    let vqe = VQEBuilder::new()
        .hamiltonian(hamiltonian)
        .ansatz_type(ansatz_type)
        .ansatz_depth(depth)
        .optimizer(optimizer_type)
        .max_iterations(100)
        .learning_rate(0.01)
        .tolerance(1e-6)
        .energy_tolerance(1e-3)
        .num_random_starts(num_random_starts)
        .verbose(false)
        .build();

    let result = vqe.run();
    let execution_time = start.elapsed().as_secs_f64() * 1000.0;

    // Calculate final gradient norm from history
    let final_gradient_norm = result
        .optimization_result
        .history
        .entries
        .last()
        .and_then(|e| e.gradient_norm)
        .unwrap_or(0.0);

    // Calculate quality score
    let quality_score = calculate_quality_score(
        result.approximation_error,
        result.classical_ground_energy,
        result.optimization_result.converged,
    );

    println!(
        "  → Energy: {:.6}, Converged: {}, Quality: {:.3}, Time: {:.2}ms",
        result.ground_state_energy,
        result.optimization_result.converged,
        quality_score,
        execution_time
    );

    VQEBenchmarkResult {
        ansatz_type: ansatz_name.to_string(),
        ansatz_depth: depth,
        optimizer: optimizer_name.to_string(),
        ground_energy: result.ground_state_energy,
        classical_ground: result.classical_ground_energy,
        approximation_error: result.approximation_error,
        iterations: result.optimization_result.iterations,
        quantum_evaluations: result.optimization_result.history.total_quantum_evaluations,
        converged: result.optimization_result.converged,
        execution_time_ms: execution_time,
        final_gradient_norm,
        quality_score,
        num_random_starts,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║   VQE BENCHMARK SUITE - Metatron QSO                  ║");
    println!("║   Enhanced with Multi-Start & Convergence Criteria    ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    let overall_start = Instant::now();

    // Create Metatron system
    let graph = MetatronGraph::new();
    let params = QSOParameters::default();
    let hamiltonian = std::sync::Arc::new(MetatronHamiltonian::new(&graph, &params));

    let mut results = Vec::new();

    // Baseline benchmarks (matching original)
    println!("\n══ Baseline Benchmarks ══\n");

    results.push(benchmark_vqe_ansatz(
        AnsatzType::HardwareEfficient,
        "HardwareEfficient",
        2,
        OptimizerType::Adam,
        "Adam",
        1,
        hamiltonian.clone(),
    ));

    results.push(benchmark_vqe_ansatz(
        AnsatzType::EfficientSU2,
        "EfficientSU2",
        2,
        OptimizerType::Adam,
        "Adam",
        1,
        hamiltonian.clone(),
    ));

    results.push(benchmark_vqe_ansatz(
        AnsatzType::Metatron,
        "Metatron",
        1,
        OptimizerType::Adam,
        "Adam",
        1,
        hamiltonian.clone(),
    ));

    // Enhanced Metatron benchmarks with higher depth
    println!("\n══ Enhanced Metatron Benchmarks ══\n");

    results.push(benchmark_vqe_ansatz(
        AnsatzType::Metatron,
        "Metatron",
        2,
        OptimizerType::Adam,
        "Adam",
        1,
        hamiltonian.clone(),
    ));

    results.push(benchmark_vqe_ansatz(
        AnsatzType::Metatron,
        "Metatron",
        3,
        OptimizerType::Adam,
        "Adam",
        1,
        hamiltonian.clone(),
    ));

    // Metatron with multi-start
    println!("\n══ Metatron with Multi-Start ══\n");

    results.push(benchmark_vqe_ansatz(
        AnsatzType::Metatron,
        "Metatron",
        2,
        OptimizerType::Adam,
        "Adam",
        3,
        hamiltonian.clone(),
    ));

    // Metatron with L-BFGS optimizer
    println!("\n══ Metatron with L-BFGS Optimizer ══\n");

    results.push(benchmark_vqe_ansatz(
        AnsatzType::Metatron,
        "Metatron",
        2,
        OptimizerType::LBFGS,
        "LBFGS",
        1,
        hamiltonian.clone(),
    ));

    results.push(benchmark_vqe_ansatz(
        AnsatzType::Metatron,
        "Metatron",
        2,
        OptimizerType::LBFGS,
        "LBFGS",
        3,
        hamiltonian.clone(),
    ));

    let total_time = overall_start.elapsed().as_secs_f64() * 1000.0;

    // Calculate performance metrics
    let total_evals: usize = results.iter().map(|r| r.quantum_evaluations).sum();
    let total_iterations: usize = results.iter().map(|r| r.iterations).sum();

    let performance_metrics = PerformanceMetrics {
        total_execution_time_ms: total_time,
        avg_iteration_time_ms: total_time / total_iterations as f64,
        total_quantum_evaluations: total_evals,
        evaluations_per_second: (total_evals as f64 / total_time) * 1000.0,
    };

    // Calculate quality metrics
    let energies: Vec<f64> = results.iter().map(|r| r.ground_energy).collect();

    let best_energy = energies.iter().cloned().fold(f64::INFINITY, f64::min);
    let worst_energy = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let avg_energy = energies.iter().sum::<f64>() / energies.len() as f64;
    let energy_variance = energies
        .iter()
        .map(|e| (e - avg_energy).powi(2))
        .sum::<f64>()
        / energies.len() as f64;

    let convergence_count = results.iter().filter(|r| r.converged).count();
    let convergence_rate = convergence_count as f64 / results.len() as f64;

    let quality_metrics = QualityMetrics {
        best_ground_energy: best_energy,
        worst_ground_energy: worst_energy,
        avg_ground_energy: avg_energy,
        energy_variance,
        convergence_rate,
    };

    let metadata = BenchmarkMetadata {
        timestamp: chrono::Utc::now(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        commit_hash: option_env!("GIT_HASH").unwrap_or("unknown").to_string(),
        system_info: format!(
            "Metatron QSO VQE Benchmarks - 13D Hilbert Space - {} configurations",
            results.len()
        ),
    };

    let suite = VQEBenchmarkSuite {
        metadata,
        results,
        performance_metrics,
        quality_metrics,
    };

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║   BENCHMARK SUMMARY                                    ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!("Configurations Tested:  {}", suite.results.len());
    println!(
        "Best Ground Energy:     {:.10}",
        suite.quality_metrics.best_ground_energy
    );
    println!(
        "Convergence Rate:       {:.1}%",
        suite.quality_metrics.convergence_rate * 100.0
    );
    println!(
        "Total Time:             {:.2}ms",
        suite.performance_metrics.total_execution_time_ms
    );
    println!(
        "Evaluations/sec:        {:.2}",
        suite.performance_metrics.evaluations_per_second
    );
    println!();

    // Accept optional output file path argument
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        // Write to specified file
        let output_path = &args[1];
        
        // Create parent directory if it doesn't exist
        if let Some(parent) = Path::new(output_path).parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create parent directory for '{}': {}", output_path, e))?;
        }
        
        let file = File::create(output_path)
            .map_err(|e| format!("Failed to create output file '{}': {}", output_path, e))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &suite)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        println!("✓ Results written to: {}", output_path);
    } else {
        // Write to stdout (default behavior)
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        serde_json::to_writer_pretty(&mut handle, &suite)?;
        handle.write_all(b"\n")?;
    }

    Ok(())
}
