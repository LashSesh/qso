//! VQE for H₂ (STO-3G, 0.735 Å): exact, noiseless VQE, and shot-based
//! energy estimation.
//!
//! Run with `cargo run --release --example vqe_h2`.

use metatron_qso::prelude::*;

fn main() -> Result<()> {
    let h2 = models::h2_sto3g();
    println!("{}", h2.name);
    println!("H = {}", h2.hamiltonian);

    let exact = h2.hamiltonian.ground_state_energy(2)?;
    println!("exact electronic ground energy  {exact:.10} Ha");
    println!(
        "exact total energy              {:.10} Ha",
        exact + h2.nuclear_repulsion
    );

    let ansatz = ansatz::real_amplitudes(2, 1, Entanglement::Linear);
    for method in [
        Method::lbfgs(),
        Method::adam(0.1),
        Method::nelder_mead(),
        Method::spsa(),
    ] {
        let r = Vqe::new(h2.hamiltonian.clone(), ansatz.clone())?
            .with_method(method.clone())
            .with_options(Options {
                max_iterations: 2000,
                ..Default::default()
            })
            .with_restarts(4)
            .run()?;
        println!(
            "VQE {:<13} E = {:.10}  |ΔE| = {:.2e} Ha  ({} evaluations)",
            method.name(),
            r.energy,
            (r.energy - exact).abs(),
            r.evaluations
        );
    }

    // Measure the optimised state with finite shots, as on hardware.
    let best = Vqe::new(h2.hamiltonian.clone(), ansatz.clone())?
        .with_restarts(4)
        .run()?;
    for shots in [100, 1_000, 10_000, 100_000] {
        let e = estimate_expectation(
            &StatevectorBackend,
            &ansatz,
            &best.parameters,
            &h2.hamiltonian,
            shots,
            1,
        )?;
        println!("shot estimate with {shots:>6} shots per basis: {e:.6} Ha");
    }

    // And with depolarising noise.
    let noisy = DensityMatrixBackend::new(NoiseModel::depolarizing(0.001, 0.01)?);
    let e = estimate_expectation(
        &noisy,
        &ansatz,
        &best.parameters,
        &h2.hamiltonian,
        100_000,
        2,
    )?;
    println!("noisy estimate (p1 = 0.1 %, p2 = 1 %):    {e:.6} Ha");
    Ok(())
}
