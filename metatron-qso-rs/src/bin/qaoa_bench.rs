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

/// QAOA Benchmark Results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAOABenchmarkSuite {
    pub metadata: BenchmarkMetadata,
    pub triangle_maxcut: QAOABenchmarkResult,
    pub square_maxcut: QAOABenchmarkResult,
    pub pentagram_maxcut: QAOABenchmarkResult,
    pub performance_metrics: PerformanceMetrics,
    pub quality_metrics: QualityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAOABenchmarkResult {
    pub problem_name: String,
    pub problem_size: usize,
    pub depth: usize,
    pub optimal_cost: f64,
    pub approximation_ratio: f64,
    pub mean_sampled_cost: f64,
    pub std_dev_cost: f64,
    pub iterations: usize,
    pub quantum_evaluations: usize,
    pub converged: bool,
    pub execution_time_ms: f64,
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
    pub best_approximation_ratio: f64,
    pub worst_approximation_ratio: f64,
    pub avg_approximation_ratio: f64,
    pub ratio_variance: f64,
    pub convergence_rate: f64,
}

fn benchmark_qaoa_problem(
    problem_name: &str,
    edges: &[(usize, usize)],
    depth: usize,
) -> QAOABenchmarkResult {
    println!("Benchmarking QAOA on {} (depth={})...", problem_name, depth);

    let start = Instant::now();

    let cost_hamiltonian =
        std::sync::Arc::new(metatron_qso::vqa::qaoa::create_maxcut_hamiltonian(edges));

    let qaoa = QAOABuilder::new()
        .cost_hamiltonian(cost_hamiltonian)
        .depth(depth)
        .optimizer(OptimizerType::NelderMead)
        .max_iterations(200)
        .verbose(false)
        .build();

    let result = qaoa.run();

    // Sample solutions to get statistics
    let (mean_cost, std_dev, _costs) = qaoa.analyze_samples(&result.optimal_state, 100);

    let execution_time = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "  → Approx Ratio: {:.4}, Cost: {:.4}, Time: {:.2}ms",
        result.approximation_ratio, result.optimal_cost, execution_time
    );

    QAOABenchmarkResult {
        problem_name: problem_name.to_string(),
        problem_size: edges.len(),
        depth,
        optimal_cost: result.optimal_cost,
        approximation_ratio: result.approximation_ratio,
        mean_sampled_cost: mean_cost,
        std_dev_cost: std_dev,
        iterations: result.optimization_result.iterations,
        quantum_evaluations: result.optimization_result.history.total_quantum_evaluations,
        converged: result.optimization_result.converged,
        execution_time_ms: execution_time,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║   QAOA BENCHMARK SUITE - Metatron QSO                 ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    let overall_start = Instant::now();

    // Define benchmark problems
    let triangle_edges = vec![(0, 1), (1, 2), (2, 0)];
    let square_edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
    let pentagram_edges = vec![
        (0, 2),
        (2, 4),
        (4, 1),
        (1, 3),
        (3, 0), // Star shape
    ];

    // Benchmark different problems
    let triangle = benchmark_qaoa_problem("Triangle MaxCut", &triangle_edges, 3);
    let square = benchmark_qaoa_problem("Square MaxCut", &square_edges, 3);
    let pentagram = benchmark_qaoa_problem("Pentagram MaxCut", &pentagram_edges, 3);

    let total_time = overall_start.elapsed().as_secs_f64() * 1000.0;

    // Calculate performance metrics
    let total_evals =
        triangle.quantum_evaluations + square.quantum_evaluations + pentagram.quantum_evaluations;

    let total_iterations = triangle.iterations + square.iterations + pentagram.iterations;

    let performance_metrics = PerformanceMetrics {
        total_execution_time_ms: total_time,
        avg_iteration_time_ms: total_time / total_iterations as f64,
        total_quantum_evaluations: total_evals,
        evaluations_per_second: (total_evals as f64 / total_time) * 1000.0,
    };

    // Calculate quality metrics
    let ratios = [triangle.approximation_ratio,
        square.approximation_ratio,
        pentagram.approximation_ratio];

    let best_ratio = ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let worst_ratio = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
    let avg_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let ratio_variance =
        ratios.iter().map(|r| (r - avg_ratio).powi(2)).sum::<f64>() / ratios.len() as f64;

    let convergence_rate = [triangle.converged as u32,
        square.converged as u32,
        pentagram.converged as u32]
    .iter()
    .sum::<u32>() as f64
        / 3.0;

    let quality_metrics = QualityMetrics {
        best_approximation_ratio: best_ratio,
        worst_approximation_ratio: worst_ratio,
        avg_approximation_ratio: avg_ratio,
        ratio_variance,
        convergence_rate,
    };

    let metadata = BenchmarkMetadata {
        timestamp: chrono::Utc::now(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        commit_hash: option_env!("GIT_HASH").unwrap_or("unknown").to_string(),
        system_info: "Metatron QSO QAOA Benchmarks - MaxCut Problems".to_string(),
    };

    let suite = QAOABenchmarkSuite {
        metadata,
        triangle_maxcut: triangle,
        square_maxcut: square,
        pentagram_maxcut: pentagram,
        performance_metrics,
        quality_metrics,
    };

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║   BENCHMARK SUMMARY                                    ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!(
        "Best Approx Ratio:      {:.4}",
        suite.quality_metrics.best_approximation_ratio
    );
    println!(
        "Avg Approx Ratio:       {:.4}",
        suite.quality_metrics.avg_approximation_ratio
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
