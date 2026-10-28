//! Circuits: Bell/GHZ states, Grover, phase estimation and OpenQASM.
//!
//! Run with `cargo run --release --example circuits`.

use metatron_qso::prelude::*;

fn main() -> Result<()> {
    // ANCHOR: ghz
    // GHZ state and its entanglement.
    let n = 5;
    let mut ghz = Circuit::new(n);
    ghz.h(0);
    for q in 1..n {
        ghz.cx(q - 1, q);
    }
    let psi = StateVector::from_circuit(&ghz, &[])?;
    let counts = psi.sample(1000, &mut Rng::seed_from_u64(7));
    println!(
        "GHZ({n}) counts: {:?}",
        counts
            .iter()
            .map(|(b, c)| (bitstring(*b, n), *c))
            .collect::<Vec<_>>()
    );
    println!(
        "entanglement entropy of qubit 0: {:.3} bit",
        psi.entanglement_entropy(&[0])?
    );
    println!("OpenQASM:\n{}", to_qasm2(&ghz, &[], true)?);
    // ANCHOR_END: ghz

    // ANCHOR: grover
    // Grover search follows sin²((2k+1)θ).
    for k in 0..=6 {
        let r = grover::grover_search(6, &[42], Some(k))?;
        println!(
            "Grover k={k}: success {:.6} (theory {:.6})",
            r.success_probability, r.theoretical_success_probability
        );
    }
    // ANCHOR_END: grover

    // ANCHOR: qpe
    // Phase estimation of P(2π·0.3125) with 4 bits recovers 0.3125 exactly.
    let phi: f64 = 0.3125;
    let u = [
        C64::new(1.0, 0.0),
        C64::new(0.0, 0.0),
        C64::new(0.0, 0.0),
        C64::from_polar(1.0, 2.0 * std::f64::consts::PI * phi),
    ];
    let est = qft::estimate_phase(4, u, [C64::new(0.0, 0.0), C64::new(1.0, 0.0)])?;
    println!(
        "phase estimate {} with probability {:.6}",
        est.phase, est.probability
    );
    // ANCHOR_END: qpe
    Ok(())
}
