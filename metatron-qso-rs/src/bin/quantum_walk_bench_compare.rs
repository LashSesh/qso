use std::env;
use std::error::Error;
use std::fs;

use metatron_qso::quantum_walk::QuantumWalkBenchmarkSuite;

/// Parse a JSON file with detailed error reporting.
/// Returns helpful diagnostics including filename, file size, and a snippet of the file content.
fn parse_json_file<T>(path: &str) -> Result<T, Box<dyn Error>>
where
    T: serde::de::DeserializeOwned,
{
    // Read the file content
    let content =
        fs::read_to_string(path).map_err(|e| format!("Failed to read file '{}': {}", path, e))?;

    // Get file metadata for size information
    let file_size = content.len();

    // Try to parse JSON
    serde_json::from_str(&content).map_err(|e| -> Box<dyn Error> {
        // Create a helpful error message with context
        let snippet_len = 200.min(content.len());
        let snippet = &content[..snippet_len];
        let snippet_display = if snippet_len < content.len() {
            format!("{}...", snippet)
        } else {
            snippet.to_string()
        };

        format!(
            "Failed to parse JSON file '{}'\n\
             File size: {} bytes\n\
             Parse error: {}\n\
             File snippet (first {} bytes):\n{}",
            path, file_size, e, snippet_len, snippet_display
        )
        .into()
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!(
            "Usage: {} <baseline.json> <candidate.json> [tolerance]",
            args.first()
                .map(String::as_str)
                .unwrap_or("quantum_walk_bench_compare")
        );
        std::process::exit(2);
    }

    let baseline_path = &args[1];
    let candidate_path = &args[2];
    let tolerance: f64 = if args.len() == 4 {
        args[3].parse().map_err(|_| "tolerance must be a float")?
    } else {
        0.05
    };

    let baseline: QuantumWalkBenchmarkSuite = parse_json_file(baseline_path)?;
    let candidate: QuantumWalkBenchmarkSuite = parse_json_file(candidate_path)?;

    if baseline.metadata != candidate.metadata {
        return Err(format!(
            "Benchmark metadata mismatch. Expected {:?} but found {:?}",
            baseline.metadata, candidate.metadata
        )
        .into());
    }

    let mut failures = Vec::new();

    compare_metric(
        "quantum_average_time",
        baseline.hitting_time.quantum_average_time,
        candidate.hitting_time.quantum_average_time,
        tolerance,
        &mut failures,
    );
    compare_metric(
        "quantum_average_steps",
        baseline.hitting_time.quantum_average_steps,
        candidate.hitting_time.quantum_average_steps,
        tolerance,
        &mut failures,
    );
    compare_metric(
        "classical_average_steps",
        baseline.hitting_time.classical_average_steps,
        candidate.hitting_time.classical_average_steps,
        tolerance,
        &mut failures,
    );
    compare_metric(
        "mean_success_probability",
        baseline.hitting_time.mean_success_probability,
        candidate.hitting_time.mean_success_probability,
        tolerance,
        &mut failures,
    );
    compare_metric(
        "speedup_factor",
        baseline.hitting_time.speedup_factor,
        candidate.hitting_time.speedup_factor,
        tolerance,
        &mut failures,
    );

    if baseline.mixing_time.times.len() != candidate.mixing_time.times.len() {
        failures.push(format!(
            "Mixing time samples changed ({} vs {}).",
            baseline.mixing_time.times.len(),
            candidate.mixing_time.times.len()
        ));
    }

    compare_optional_metric(
        "mixing_time",
        &baseline.mixing_time.mixing_time,
        &candidate.mixing_time.mixing_time,
        tolerance,
        &mut failures,
    );

    if !failures.is_empty() {
        eprintln!(
            "Benchmark regression detected (tolerance ±{:.2}%):",
            tolerance * 100.0
        );
        for failure in failures {
            eprintln!("  - {}", failure);
        }
        return Err("benchmark comparison failed".into());
    }

    println!(
        "Benchmark comparison passed within ±{:.2}% tolerance.",
        tolerance * 100.0
    );
    println!(
        "  speedup_factor: {:.4} (baseline {:.4})",
        candidate.hitting_time.speedup_factor, baseline.hitting_time.speedup_factor
    );
    println!(
        "  mean_success_probability: {:.4} (baseline {:.4})",
        candidate.hitting_time.mean_success_probability,
        baseline.hitting_time.mean_success_probability
    );

    Ok(())
}

fn compare_metric(
    name: &str,
    baseline: f64,
    candidate: f64,
    tolerance: f64,
    failures: &mut Vec<String>,
) {
    if !within_tolerance(baseline, candidate, tolerance) {
        failures.push(format!(
            "{} deviated beyond tolerance (baseline {:.6}, candidate {:.6})",
            name, baseline, candidate
        ));
    }
}

fn compare_optional_metric(
    name: &str,
    baseline: &Option<f64>,
    candidate: &Option<f64>,
    tolerance: f64,
    failures: &mut Vec<String>,
) {
    match (baseline, candidate) {
        (Some(b), Some(c)) => compare_metric(name, *b, *c, tolerance, failures),
        (None, None) => {}
        (b, c) => failures.push(format!("{} changed from {:?} to {:?}", name, b, c)),
    }
}

fn within_tolerance(baseline: f64, candidate: f64, tolerance: f64) -> bool {
    if baseline == 0.0 {
        (candidate - baseline).abs() <= tolerance
    } else {
        let denom = baseline.abs().max(1e-9);
        ((candidate - baseline).abs() / denom) <= tolerance
    }
}
