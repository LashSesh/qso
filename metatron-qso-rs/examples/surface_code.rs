//! Quantum error correction: repetition and rotated surface-code memory
//! experiments under circuit-level depolarizing noise, sampled with Pauli
//! frames and decoded by union-find.
//!
//!     cargo run --release --example surface_code

use std::time::Instant;

use metatron_qso::Result;
use metatron_qso::qec::{
    CircuitNoise, detector_error_model, logical_error_rate, repetition_code_memory,
    surface_code_memory,
};

fn main() -> Result<()> {
    let shots = 100_000;
    println!("rotated surface code, d rounds, circuit-level noise p, {shots} shots");
    println!(
        "{:>7} {:>3} {:>12} {:>10} {:>8}",
        "p", "d", "logical", "± (1σ)", "time"
    );
    for p in [0.001, 0.003, 0.01, 0.02] {
        for d in [3, 5, 7] {
            let circuit = surface_code_memory(d, d, &CircuitNoise::uniform(p))?;
            let start = Instant::now();
            let r = logical_error_rate(&circuit, shots, 7)?;
            println!(
                "{p:>7} {d:>3} {:>12.5} {:>10.5} {:>7.2}s",
                r.rate,
                r.standard_error,
                start.elapsed().as_secs_f64()
            );
        }
    }

    println!("\nrepetition code, d rounds, circuit-level noise p, {shots} shots");
    for p in [0.01, 0.05] {
        for d in [3, 5, 9, 15] {
            let circuit = repetition_code_memory(d, d, &CircuitNoise::uniform(p))?;
            let r = logical_error_rate(&circuit, shots, 7)?;
            println!("{p:>7} {d:>3} {:>12.5} {:>10.5}", r.rate, r.standard_error);
        }
    }

    let circuit = surface_code_memory(5, 5, &CircuitNoise::uniform(0.001))?;
    let dem = detector_error_model(&circuit, true)?;
    println!(
        "\nsurface code d = 5, 5 rounds: {} qubits, {} detectors, {} error mechanisms",
        circuit.n_qubits(),
        dem.n_detectors,
        dem.errors.len()
    );
    Ok(())
}
