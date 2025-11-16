use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufWriter, Write};

use metatron_qso::prelude::*;
use metatron_qso::quantum_walk::QuantumWalkBenchmarkSuite;

fn main() -> Result<(), Box<dyn Error>> {
    let epsilon = 0.05;
    let mixing_dt = 0.5;
    let mixing_samples = 40; // Extended to allow more time for mixing
    let hitting_dt = 0.25;
    let hitting_steps = 24;

    // Optimized dephasing to achieve mixing while preserving hitting-time speedup
    let params = QSOParameters::default().with_dephasing(0.032);
    let qso = QuantumStateOperator::new(params);
    let benchmarker = qso.quantum_walk_benchmarker();
    let initial = qso.basis_state(0);

    let suite: QuantumWalkBenchmarkSuite = benchmarker.benchmark_suite(
        &initial,
        mixing_dt,
        mixing_samples,
        epsilon,
        hitting_dt,
        hitting_steps,
    );

    // Accept optional output file path argument
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        // Write to specified file
        let output_path = &args[1];
        let file = File::create(output_path)
            .map_err(|e| format!("Failed to create output file '{}': {}", output_path, e))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &suite)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    } else {
        // Write to stdout (default behavior)
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        serde_json::to_writer_pretty(&mut handle, &suite)?;
        handle.write_all(b"\n")?;
    }

    Ok(())
}
