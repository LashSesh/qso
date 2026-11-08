//! Expectation values of a 115-qubit noisy circuit by Pauli propagation:
//! the kicked Ising model of Kim et al. (Nature 618, 500 (2023)) on
//! IBM's heavy-hex lattice.
//!
//! One step is `RX(θ_h)` on every qubit followed by `RZZ(−π/2)` on every
//! edge (three layers, one per colour of the edges). The observable is
//! `⟨Z₅₇⟩` on the central qubit. Up to five steps the propagation is exact
//! (about 1.35·10⁵ strings); beyond that coefficients below a threshold
//! are dropped, and the result carries a rigorous bound on the error this
//! makes. Finally a sparse Pauli–Lindblad noise model after every ZZ layer
//! is amplified (`α > 0`) and inverted (`α = −1`), the ingredients of
//! zero-noise extrapolation by probabilistic error amplification and of
//! probabilistic error cancellation.
//!
//! Run with `cargo run --release --example pauli_propagation`; set
//! `RAYON_NUM_THREADS` to choose the number of threads (the results are
//! bit-identical for every count).

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
use std::time::Instant;

use metatron_qso::propagation::{Layer, PauliOperator, PropagationOptions, propagate};
use metatron_qso::quantum::{Circuit, Pauli};

/// The heavy-hex lattice of Qiskit's `CouplingMap.from_heavy_hex(7)`: 115
/// qubits and its 132 edges in three colour classes (written by
/// `benchmarks/cross_framework/propagation_data.py`).
fn heavy_hex() -> (usize, Vec<Vec<(usize, usize)>>) {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("../tests/data/heavy_hex_7.json")).expect("lattice");
    let colours = data["colours"]
        .as_array()
        .expect("colour classes")
        .iter()
        .map(|class| {
            class
                .as_array()
                .expect("edges")
                .iter()
                .map(|e| {
                    (
                        e[0].as_u64().unwrap() as usize,
                        e[1].as_u64().unwrap() as usize,
                    )
                })
                .collect()
        })
        .collect();
    (data["n_qubits"].as_u64().expect("size") as usize, colours)
}

/// `steps` kicked-Ising steps; with `rates`, the Pauli–Lindblad layer of
/// colour `k` (scaled by `alpha`) follows the ZZ layer of colour `k`.
fn kicked_ising(
    n: usize,
    colours: &[Vec<(usize, usize)>],
    steps: usize,
    theta: f64,
    noise: Option<(&[PauliOperator], f64)>,
) -> Vec<Layer> {
    let mut layers = Vec::new();
    let mut c = Circuit::new(n);
    for _ in 0..steps {
        for q in 0..n {
            c.rx(q, theta);
        }
        for (k, class) in colours.iter().enumerate() {
            for &(a, b) in class {
                c.rzz(a, b, -FRAC_PI_2);
            }
            if let Some((rates, alpha)) = noise {
                layers.push(Layer::gates(std::mem::replace(&mut c, Circuit::new(n))));
                layers.push(Layer::pauli_lindblad(rates[k].clone(), alpha));
            }
        }
    }
    if !c.instructions().is_empty() {
        layers.push(Layer::gates(c));
    }
    layers
}

/// Peak resident memory of this process (Linux only).
fn peak_memory_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

fn main() -> metatron_qso::Result<()> {
    let (n, colours) = heavy_hex();
    let centre = n / 2;
    let z = PauliOperator::parse(n, &format!("Z{centre}"))?;
    println!("Kicked Ising on heavy-hex(7): {n} qubits, <Z{centre}>");

    println!("\nExact propagation (no truncation)");
    println!("theta_h  steps  value               strings   time");
    for theta in [0.3, FRAC_PI_4, 1.0] {
        for steps in 1..=5 {
            let layers = kicked_ising(n, &colours, steps, theta, None);
            let t = Instant::now();
            let r = propagate(&layers, &z, None, &PropagationOptions::default())?;
            println!(
                "{theta:<7.4}  {steps:>5}  {:<18.15}  {:>7}  {:>7.1} ms",
                r.value,
                r.terms,
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    println!("\nTruncated propagation at theta_h = 0.3");
    println!("steps  min_coefficient  value               bound      strings   time");
    for steps in [6, 7] {
        for tau in [1e-3, 1e-4, 1e-5] {
            let layers = kicked_ising(n, &colours, steps, 0.3, None);
            let t = Instant::now();
            let o = PropagationOptions::default().with_min_coefficient(tau);
            let r = propagate(&layers, &z, None, &o)?;
            println!(
                "{steps:>5}  {tau:>15.0e}  {:<18.15}  {:<9.3e}  {:>7}  {:>7.1} ms",
                r.value,
                r.truncation_bound,
                r.terms,
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    // A uniform sparse Pauli–Lindblad model after the ZZ layer of colour
    // k: 10⁻³ (X + Y + Z) on every qubit, 2·10⁻⁴ XX + 5·10⁻⁴ ZZ on its edges.
    let rates: Vec<PauliOperator> = colours
        .iter()
        .map(|class| {
            let mut rates = PauliOperator::new(n)?;
            for q in 0..n {
                for p in [Pauli::X, Pauli::Y, Pauli::Z] {
                    rates.add_term(1e-3, &[(q, p)])?;
                }
            }
            for &(a, b) in class {
                rates.add_term(2e-4, &[(a, Pauli::X), (b, Pauli::X)])?;
                rates.add_term(5e-4, &[(a, Pauli::Z), (b, Pauli::Z)])?;
            }
            Ok(rates)
        })
        .collect::<metatron_qso::Result<_>>()?;
    println!("\nNoisy kicked Ising, theta_h = 1, 3 steps (alpha = noise factor)");
    println!("alpha  value");
    for alpha in [0.0, 1.0, 1.5, 2.0, -1.0] {
        let layers = kicked_ising(n, &colours, 3, 1.0, Some((&rates, alpha)));
        let r = propagate(&layers, &z, None, &PropagationOptions::default())?;
        println!("{alpha:>5}  {:.15}", r.value);
    }
    if let Some(mb) = peak_memory_mb() {
        println!("\npeak resident memory: {mb:.0} MB");
    }
    Ok(())
}
