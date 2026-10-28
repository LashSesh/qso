//! The quick-start snippet from the README (kept compiling by CI).
//!
//! Run with `cargo run --release --example quickstart`.

use metatron_qso::prelude::*;

fn main() -> Result<()> {
    // ANCHOR: vqe
    // VQE for H₂.
    let h2 = models::h2_sto3g();
    let ansatz = ansatz::real_amplitudes(2, 1, Entanglement::Linear);
    let vqe = Vqe::new(h2.hamiltonian.clone(), ansatz)?
        .with_restarts(4)
        .run()?;
    println!("E = {:.6} Ha", vqe.energy + h2.nuclear_repulsion); // −1.137306
    // ANCHOR_END: vqe

    // ANCHOR: qaoa
    // QAOA for MaxCut on Metatron's Cube.
    let qaoa = MaxCutQaoa::new(&metatron::metatron_cube(), 2)?;
    let r = qaoa.optimize(&QaoaConfig::default())?;
    println!("ratio = {:.4}", r.approximation_ratio.unwrap());
    // ANCHOR_END: qaoa

    // ANCHOR: gradient
    // A parametrised circuit with exact gradients.
    let mut c = Circuit::new(2);
    let theta = c.new_parameter();
    c.h(0).cx(0, 1).ry(1, theta);
    let zz: PauliSum = "Z0 Z1".parse()?;
    let (e, grad) = metatron_qso::algorithms::gradients::expectation_and_gradient(
        &c,
        &zz,
        &[0.3],
        None,
        GradientMethod::Adjoint,
    )?;
    println!("<ZZ> = {e:.4}, d<ZZ>/dθ = {:.4}", grad[0]);
    // ANCHOR_END: gradient

    // ANCHOR: walk
    // Quantum walk and spatial search.
    let walk = ContinuousWalk::new(&metatron::fruit_of_life(), Generator::Laplacian, 1.0)?;
    println!("P(t=2) = {:?}", walk.probabilities(0, 2.0)?);
    // ANCHOR_END: walk
    Ok(())
}
