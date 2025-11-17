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

/// VQC Benchmark Results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VQCBenchmarkSuite {
    pub metadata: BenchmarkMetadata,
    pub binary_classification: VQCBenchmarkResult,
    pub linearly_separable: VQCBenchmarkResult,
    pub performance_metrics: PerformanceMetrics,
    pub quality_metrics: QualityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VQCBenchmarkResult {
    pub problem_name: String,
    pub training_samples: usize,
    pub feature_dim: usize,
    pub training_accuracy: f64,
    pub training_loss: f64,
    pub test_accuracy: f64,
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
    pub best_training_accuracy: f64,
    pub worst_training_accuracy: f64,
    pub avg_training_accuracy: f64,
    pub avg_test_accuracy: f64,
    pub convergence_rate: f64,
}

fn benchmark_vqc_problem(
    problem_name: &str,
    training_data: Vec<Vec<f64>>,
    training_labels: Vec<usize>,
    test_data: Vec<Vec<f64>>,
    test_labels: Vec<usize>,
) -> VQCBenchmarkResult {
    println!("Benchmarking VQC on {}...", problem_name);

    let start = Instant::now();
    let feature_dim = training_data[0].len();
    let training_samples = training_data.len();

    let mut vqc = VQCBuilder::new()
        .ansatz_type(AnsatzType::HardwareEfficient)
        .ansatz_depth(3) // Increased depth for better expressiveness
        .encoding(metatron_qso::vqa::vqc::EncodingType::Angle)
        .optimizer(OptimizerType::Adam)
        .max_iterations(300) // More iterations for convergence
        .learning_rate(0.03) // Slightly higher learning rate
        .tolerance(1e-5) // Tighter tolerance
        .verbose(false)
        .build();

    let training_result = vqc.train(training_data.clone(), training_labels.clone());

    // Evaluate on test set
    let test_accuracy = vqc.evaluate(test_data, test_labels);

    let execution_time = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "  → Train Acc: {:.2}%, Test Acc: {:.2}%, Time: {:.2}ms",
        training_result.training_accuracy * 100.0,
        test_accuracy * 100.0,
        execution_time
    );

    VQCBenchmarkResult {
        problem_name: problem_name.to_string(),
        training_samples,
        feature_dim,
        training_accuracy: training_result.training_accuracy,
        training_loss: training_result.training_loss,
        test_accuracy,
        iterations: training_result.optimization_result.iterations,
        quantum_evaluations: training_result
            .optimization_result
            .history
            .total_quantum_evaluations,
        converged: training_result.optimization_result.converged,
        execution_time_ms: execution_time,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║   VQC BENCHMARK SUITE - Metatron QSO                  ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    let overall_start = Instant::now();

    // Problem 1: Binary classification with clear separation
    let binary_train_data = vec![
        vec![0.1, 0.1, 0.0, 0.0],
        vec![0.2, 0.15, 0.0, 0.0],
        vec![0.15, 0.2, 0.0, 0.0],
        vec![0.05, 0.1, 0.0, 0.0],
        vec![0.8, 0.85, 0.0, 0.0],
        vec![0.9, 0.9, 0.0, 0.0],
        vec![0.85, 0.8, 0.0, 0.0],
        vec![0.95, 0.85, 0.0, 0.0],
    ];
    let binary_train_labels = vec![0, 0, 0, 0, 1, 1, 1, 1];

    let binary_test_data = vec![vec![0.12, 0.13, 0.0, 0.0], vec![0.88, 0.87, 0.0, 0.0]];
    let binary_test_labels = vec![0, 1];

    // Problem 2: Linearly separable with margin
    let linear_train_data = vec![
        vec![0.0, 0.0, 0.1, 0.1],
        vec![0.1, 0.0, 0.15, 0.2],
        vec![0.0, 0.1, 0.2, 0.15],
        vec![0.05, 0.05, 0.1, 0.05],
        vec![0.5, 0.5, 0.85, 0.8],
        vec![0.6, 0.5, 0.9, 0.9],
        vec![0.5, 0.6, 0.8, 0.85],
        vec![0.55, 0.55, 0.85, 0.95],
    ];
    let linear_train_labels = vec![0, 0, 0, 0, 1, 1, 1, 1];

    let linear_test_data = vec![vec![0.03, 0.02, 0.13, 0.12], vec![0.52, 0.53, 0.87, 0.88]];
    let linear_test_labels = vec![0, 1];

    // Run benchmarks
    let binary = benchmark_vqc_problem(
        "Binary Classification",
        binary_train_data,
        binary_train_labels,
        binary_test_data,
        binary_test_labels,
    );

    let linear = benchmark_vqc_problem(
        "Linearly Separable",
        linear_train_data,
        linear_train_labels,
        linear_test_data,
        linear_test_labels,
    );

    let total_time = overall_start.elapsed().as_secs_f64() * 1000.0;

    // Calculate performance metrics
    let total_evals = binary.quantum_evaluations + linear.quantum_evaluations;
    let total_iterations = binary.iterations + linear.iterations;

    let performance_metrics = PerformanceMetrics {
        total_execution_time_ms: total_time,
        avg_iteration_time_ms: total_time / total_iterations as f64,
        total_quantum_evaluations: total_evals,
        evaluations_per_second: (total_evals as f64 / total_time) * 1000.0,
    };

    // Calculate quality metrics
    let train_accuracies = vec![binary.training_accuracy, linear.training_accuracy];
    let test_accuracies = vec![binary.test_accuracy, linear.test_accuracy];

    let best_train_acc = train_accuracies
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let worst_train_acc = train_accuracies
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let avg_train_acc = train_accuracies.iter().sum::<f64>() / train_accuracies.len() as f64;
    let avg_test_acc = test_accuracies.iter().sum::<f64>() / test_accuracies.len() as f64;

    let convergence_rate = vec![binary.converged as u32, linear.converged as u32]
        .iter()
        .sum::<u32>() as f64
        / 2.0;

    let quality_metrics = QualityMetrics {
        best_training_accuracy: best_train_acc,
        worst_training_accuracy: worst_train_acc,
        avg_training_accuracy: avg_train_acc,
        avg_test_accuracy: avg_test_acc,
        convergence_rate,
    };

    let metadata = BenchmarkMetadata {
        timestamp: chrono::Utc::now(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        commit_hash: option_env!("GIT_HASH").unwrap_or("unknown").to_string(),
        system_info: format!("Metatron QSO VQC Benchmarks - Classification Tasks"),
    };

    let suite = VQCBenchmarkSuite {
        metadata,
        binary_classification: binary,
        linearly_separable: linear,
        performance_metrics,
        quality_metrics,
    };

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║   BENCHMARK SUMMARY                                    ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!(
        "Avg Training Accuracy:  {:.2}%",
        suite.quality_metrics.avg_training_accuracy * 100.0
    );
    println!(
        "Avg Test Accuracy:      {:.2}%",
        suite.quality_metrics.avg_test_accuracy * 100.0
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
