//! Matrix-product states: a quench of an XX chain on 200 qubits and QAOA
//! on a ring of 200 nodes, far beyond a state vector, with the truncation
//! error bounded.
//!
//!     cargo run --release --example mps

use std::time::Instant;

use metatron_qso::Result;
use metatron_qso::algorithms::ansatz::qaoa_maxcut_circuit;
use metatron_qso::algorithms::qaoa::{QaoaConfig, optimize_layerwise};
use metatron_qso::graph::generators;
use metatron_qso::mps::{Mps, MpsOptions};
use metatron_qso::quantum::Circuit;

fn main() -> Result<()> {
    // ANCHOR: quench
    // Néel state, then Trotter steps of the XX chain: the excitations
    // spread ballistically and the entanglement grows linearly.
    let n = 200;
    let theta = 0.25;
    let mut state = Mps::zero(n, MpsOptions::default())?;
    let mut neel = Circuit::new(n);
    for q in (0..n).step_by(2) {
        neel.x(q);
    }
    state.apply_circuit(&neel, &[])?;
    let mut step = Circuit::new(n);
    for start in [0, 1] {
        for j in (start..n - 1).step_by(2) {
            step.rxx(j, j + 1, theta).ryy(j, j + 1, theta);
        }
    }
    println!("XX chain, {n} qubits, Néel state, θ = {theta}");
    println!(
        "{:>6} {:>10} {:>14} {:>9} {:>8}",
        "steps", "⟨Z_100⟩", "S(100|100)", "max χ", "time"
    );
    let start = Instant::now();
    for k in 1..=8 {
        state.apply_circuit(&step, &[])?;
        if k % 2 == 0 {
            let z = state.expectation_str("Z100")?;
            let s = state.bond_entropies()?[n / 2 - 1];
            println!(
                "{k:>6} {z:>10.6} {s:>14.6} {:>9} {:>7.2}s",
                state.max_bond_dimension(),
                start.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "fidelity ≥ {:.12}, discarded weight {:.1e}",
        state.fidelity_lower_bound().unwrap_or(0.0),
        state.discarded_weight()
    );
    // ANCHOR_END: quench

    // ANCHOR: qaoa
    // QAOA for MaxCut on a ring of 200 nodes with angles optimized on a
    // ring of 10: the cut fraction is the proven optimum (2p + 1)/(2p + 2).
    println!("\nQAOA on a ring of {n} nodes");
    let ring = generators::cycle(n);
    for r in optimize_layerwise(&generators::cycle(10), 3, &QaoaConfig::default())? {
        let mut theta = r.gammas.clone();
        theta.extend(&r.betas);
        let start = Instant::now();
        let mut s = Mps::from_circuit(
            &qaoa_maxcut_circuit(&ring, r.p)?,
            &theta,
            MpsOptions::default(),
        )?;
        let mut cut = 0.0;
        for &(a, b, _) in ring.edges() {
            cut += (1.0 - s.expectation_str(&format!("Z{a} Z{b}"))?) / 2.0;
        }
        println!(
            "  p = {}: cut fraction {:.6} (optimum {:.6}), max χ {}, {:.2}s",
            r.p,
            cut / n as f64,
            (2 * r.p + 1) as f64 / (2 * r.p + 2) as f64,
            s.max_bond_dimension(),
            start.elapsed().as_secs_f64()
        );
    }
    // ANCHOR_END: qaoa
    Ok(())
}
