use std::env;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use serde_json::Value;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: {} <baseline.json> <current.json> [threshold_percent]",
            args[0]
        );
        eprintln!("  Compares benchmark results and fails if performance degrades");
        std::process::exit(1);
    }

    let baseline_path = &args[1];
    let current_path = &args[2];
    let threshold_percent = if args.len() > 3 {
        args[3].parse::<f64>().unwrap_or(10.0)
    } else {
        10.0 // Default 10% threshold
    };

    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║   BENCHMARK COMPARISON                                 ║");
    println!("╚════════════════════════════════════════════════════════╝\n");
    println!("Baseline: {}", baseline_path);
    println!("Current:  {}", current_path);
    println!("Threshold: ±{:.1}%\n", threshold_percent);

    // Load JSON files
    let baseline_file = File::open(baseline_path)?;
    let current_file = File::open(current_path)?;

    let baseline: Value = serde_json::from_reader(BufReader::new(baseline_file))?;
    let current: Value = serde_json::from_reader(BufReader::new(current_file))?;

    // Determine benchmark type from metadata
    let benchmark_type = current
        .get("metadata")
        .and_then(|m| m.get("system_info"))
        .and_then(|s| s.as_str())
        .unwrap_or("Unknown");

    println!("Benchmark Type: {}\n", benchmark_type);

    let mut regressions = Vec::new();
    let mut improvements = Vec::new();
    let mut stable = Vec::new();

    // Compare based on benchmark type
    if benchmark_type.contains("VQE") {
        compare_vqe_benchmarks(
            &baseline,
            &current,
            threshold_percent,
            &mut regressions,
            &mut improvements,
            &mut stable,
        )?;
    } else if benchmark_type.contains("QAOA") {
        compare_qaoa_benchmarks(
            &baseline,
            &current,
            threshold_percent,
            &mut regressions,
            &mut improvements,
            &mut stable,
        )?;
    } else if benchmark_type.contains("VQC") {
        compare_vqc_benchmarks(
            &baseline,
            &current,
            threshold_percent,
            &mut regressions,
            &mut improvements,
            &mut stable,
        )?;
    } else if benchmark_type.contains("Integration") {
        compare_integration_benchmarks(
            &baseline,
            &current,
            threshold_percent,
            &mut regressions,
            &mut improvements,
            &mut stable,
        )?;
    } else if benchmark_type.contains("Cross-System") {
        compare_cross_system_benchmarks(
            &baseline,
            &current,
            threshold_percent,
            &mut regressions,
            &mut improvements,
            &mut stable,
        )?;
    } else if benchmark_type.contains("Quantum Walk") {
        compare_quantum_walk_benchmarks(
            &baseline,
            &current,
            threshold_percent,
            &mut regressions,
            &mut improvements,
            &mut stable,
        )?;
    } else if benchmark_type.contains("Advanced Algorithms") {
        compare_advanced_algorithms_benchmarks(
            &baseline,
            &current,
            threshold_percent,
            &mut regressions,
            &mut improvements,
            &mut stable,
        )?;
    } else {
        eprintln!("⚠ Unknown benchmark type: {}", benchmark_type);
        return Ok(());
    }

    // Print results
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║   COMPARISON RESULTS                                   ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    if !improvements.is_empty() {
        println!("✓ IMPROVEMENTS ({}):", improvements.len());
        for imp in &improvements {
            println!("  {}", imp);
        }
        println!();
    }

    if !stable.is_empty() {
        println!("→ STABLE ({}):", stable.len());
        for stb in &stable {
            println!("  {}", stb);
        }
        println!();
    }

    if !regressions.is_empty() {
        println!("✗ REGRESSIONS ({}):", regressions.len());
        for reg in &regressions {
            println!("  {}", reg);
        }
        println!();
        println!("❌ FAILED: Performance regressions detected!");
        std::process::exit(1);
    }

    println!("✅ PASSED: All benchmarks within acceptable range!");
    Ok(())
}

fn compare_vqe_benchmarks(
    baseline: &Value,
    current: &Value,
    threshold: f64,
    regressions: &mut Vec<String>,
    improvements: &mut Vec<String>,
    stable: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    // Compare quality metrics
    compare_metric(
        baseline,
        current,
        &["quality_metrics", "best_ground_energy"],
        "Best Ground Energy",
        threshold,
        true, // lower is better
        regressions,
        improvements,
        stable,
    );

    compare_metric(
        baseline,
        current,
        &["quality_metrics", "convergence_rate"],
        "Convergence Rate",
        threshold,
        false, // higher is better
        regressions,
        improvements,
        stable,
    );

    // Compare performance
    compare_metric(
        baseline,
        current,
        &["performance_metrics", "evaluations_per_second"],
        "Evaluations/sec",
        threshold,
        false, // higher is better
        regressions,
        improvements,
        stable,
    );

    Ok(())
}

fn compare_qaoa_benchmarks(
    baseline: &Value,
    current: &Value,
    threshold: f64,
    regressions: &mut Vec<String>,
    improvements: &mut Vec<String>,
    stable: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    compare_metric(
        baseline,
        current,
        &["quality_metrics", "best_approximation_ratio"],
        "Best Approximation Ratio",
        threshold,
        false, // higher is better
        regressions,
        improvements,
        stable,
    );

    compare_metric(
        baseline,
        current,
        &["quality_metrics", "convergence_rate"],
        "Convergence Rate",
        threshold,
        false,
        regressions,
        improvements,
        stable,
    );

    compare_metric(
        baseline,
        current,
        &["performance_metrics", "evaluations_per_second"],
        "Evaluations/sec",
        threshold,
        false,
        regressions,
        improvements,
        stable,
    );

    Ok(())
}

fn compare_vqc_benchmarks(
    baseline: &Value,
    current: &Value,
    threshold: f64,
    regressions: &mut Vec<String>,
    improvements: &mut Vec<String>,
    stable: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    compare_metric(
        baseline,
        current,
        &["quality_metrics", "avg_training_accuracy"],
        "Avg Training Accuracy",
        threshold,
        false,
        regressions,
        improvements,
        stable,
    );

    compare_metric(
        baseline,
        current,
        &["quality_metrics", "avg_test_accuracy"],
        "Avg Test Accuracy",
        threshold,
        false,
        regressions,
        improvements,
        stable,
    );

    Ok(())
}

fn compare_integration_benchmarks(
    baseline: &Value,
    current: &Value,
    threshold: f64,
    regressions: &mut Vec<String>,
    improvements: &mut Vec<String>,
    stable: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    compare_metric(
        baseline,
        current,
        &["cross_module_metrics", "overall_compatibility_score"],
        "Compatibility Score",
        threshold,
        false,
        regressions,
        improvements,
        stable,
    );

    compare_metric(
        baseline,
        current,
        &["cross_module_metrics", "integration_success_rate"],
        "Integration Success Rate",
        threshold,
        false,
        regressions,
        improvements,
        stable,
    );

    Ok(())
}

fn compare_cross_system_benchmarks(
    baseline: &Value,
    current: &Value,
    threshold: f64,
    regressions: &mut Vec<String>,
    improvements: &mut Vec<String>,
    stable: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    compare_metric(
        baseline,
        current,
        &["metatron_qso", "overall_score"],
        "Metatron Overall Score",
        threshold,
        false,
        regressions,
        improvements,
        stable,
    );

    compare_metric(
        baseline,
        current,
        &["comparison_metrics", "systems_outperformed"],
        "Systems Outperformed",
        threshold,
        false,
        regressions,
        improvements,
        stable,
    );

    Ok(())
}

fn compare_quantum_walk_benchmarks(
    baseline: &Value,
    current: &Value,
    threshold: f64,
    regressions: &mut Vec<String>,
    improvements: &mut Vec<String>,
    stable: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    compare_metric(
        baseline,
        current,
        &["hitting_time", "speedup_factor"],
        "Speedup Factor",
        threshold,
        false,
        regressions,
        improvements,
        stable,
    );

    compare_metric(
        baseline,
        current,
        &["hitting_time", "quantum_average_steps"],
        "Quantum Avg Steps",
        threshold,
        true, // lower is better
        regressions,
        improvements,
        stable,
    );

    Ok(())
}

fn compare_advanced_algorithms_benchmarks(
    baseline: &Value,
    current: &Value,
    threshold: f64,
    regressions: &mut Vec<String>,
    improvements: &mut Vec<String>,
    stable: &mut Vec<String>,
) -> Result<(), Box<dyn Error>> {
    // Compare Grover search metrics
    compare_metric(
        baseline,
        current,
        &["grover_search", "success_probability"],
        "Grover Success Probability",
        threshold,
        false, // higher is better
        regressions,
        improvements,
        stable,
    );

    compare_metric(
        baseline,
        current,
        &["grover_search", "speedup"],
        "Grover Speedup",
        threshold,
        false, // higher is better
        regressions,
        improvements,
        stable,
    );

    // Compare multi-target Grover metrics
    compare_metric(
        baseline,
        current,
        &["multi_target_grover", "success_probability"],
        "Multi-Target Grover Success Prob",
        threshold,
        false,
        regressions,
        improvements,
        stable,
    );

    // Compare boson sampling metrics
    compare_metric(
        baseline,
        current,
        &["boson_sampling", "interference_visibility"],
        "Boson Sampling Visibility",
        threshold,
        false,
        regressions,
        improvements,
        stable,
    );

    // Compare quantum ML metrics
    compare_metric(
        baseline,
        current,
        &["quantum_ml", "test_accuracy"],
        "Quantum ML Test Accuracy",
        threshold,
        false,
        regressions,
        improvements,
        stable,
    );

    // Compare performance metrics
    compare_metric(
        baseline,
        current,
        &["performance_metrics", "grover_ops_per_second"],
        "Grover Ops/sec",
        threshold,
        false,
        regressions,
        improvements,
        stable,
    );

    compare_metric(
        baseline,
        current,
        &["performance_metrics", "boson_samples_per_second"],
        "Boson Samples/sec",
        threshold,
        false,
        regressions,
        improvements,
        stable,
    );

    Ok(())
}

fn compare_metric(
    baseline: &Value,
    current: &Value,
    path: &[&str],
    name: &str,
    threshold: f64,
    lower_is_better: bool,
    regressions: &mut Vec<String>,
    improvements: &mut Vec<String>,
    stable: &mut Vec<String>,
) {
    let baseline_val = get_nested_value(baseline, path);
    let current_val = get_nested_value(current, path);

    if baseline_val.is_none() || current_val.is_none() {
        stable.push(format!("{}: N/A (missing data)", name));
        return;
    }

    let baseline_num = baseline_val.unwrap();
    let current_num = current_val.unwrap();

    let change_percent = ((current_num - baseline_num) / baseline_num.abs().max(1e-10)) * 100.0;

    let is_regression = if lower_is_better {
        change_percent > threshold
    } else {
        change_percent < -threshold
    };

    let is_improvement = if lower_is_better {
        change_percent < -threshold
    } else {
        change_percent > threshold
    };

    let msg = format!(
        "{}: {:.6} → {:.6} ({:+.2}%)",
        name, baseline_num, current_num, change_percent
    );

    if is_regression {
        regressions.push(msg);
    } else if is_improvement {
        improvements.push(msg);
    } else {
        stable.push(msg);
    }
}

fn get_nested_value(value: &Value, path: &[&str]) -> Option<f64> {
    let mut current = value;
    for &key in path {
        current = current.get(key)?;
    }
    current.as_f64()
}
