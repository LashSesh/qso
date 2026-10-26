//! Quantum error correction: repetition and rotated surface-code memory
//! experiments under circuit-level depolarizing noise, sampled with Pauli
//! frames and decoded by union-find and by minimum-weight matching (the
//! same shots for both).
//!
//!     cargo run --release --example surface_code

use std::time::Instant;

use metatron_qso::Result;
use metatron_qso::qec::{
    CircuitNoise, DecoderKind, MemoryBasis, detector_error_model, logical_error_rate,
    logical_error_rate_with, repetition_code_memory, surface_code_memory, surface_code_memory_with,
};

fn main() -> Result<()> {
    println!("rotated surface code, d rounds, circuit-level noise p");
    println!(
        "{:>7} {:>3} {:>9} {:>11} {:>8} {:>11} {:>8}",
        "p", "d", "shots", "union-find", "time", "matching", "time"
    );
    for p in [0.001, 0.003, 0.01, 0.02] {
        // Rare failures need more shots.
        let shots = if p <= 0.001 { 1_000_000 } else { 100_000 };
        for d in [3, 5, 7] {
            let circuit = surface_code_memory(d, d, &CircuitNoise::uniform(p))?;
            let start = Instant::now();
            let uf = logical_error_rate(&circuit, shots, 7)?;
            let t_uf = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let mwpm = logical_error_rate_with(&circuit, shots, 7, DecoderKind::Matching)?;
            println!(
                "{p:>7} {d:>3} {shots:>9} {:>11.6} {t_uf:>7.2}s {:>11.6} {:>7.2}s",
                uf.rate,
                mwpm.rate,
                start.elapsed().as_secs_f64()
            );
        }
    }

    let shots = 100_000;
    println!(
        "
X basis (|+⟩_L), matching, {shots} shots"
    );
    for p in [0.003, 0.01] {
        for d in [3, 5, 7] {
            let noise = CircuitNoise::uniform(p);
            let circuit = surface_code_memory_with(d, d, &noise, MemoryBasis::X)?;
            let r = logical_error_rate_with(&circuit, shots, 7, DecoderKind::Matching)?;
            println!("{p:>7} {d:>3} {:>12.6} {:>10.6}", r.rate, r.standard_error);
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
