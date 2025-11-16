use metatron_qso::prelude::*;

fn main() {
    let params = QSOParameters::default();
    let qso = QuantumStateOperator::new(params);
    let benchmarker = qso.quantum_walk_benchmarker();

    let initial = qso.basis_state(0);
    let times: Vec<f64> = (0..24).map(|k| k as f64 * 0.5).collect();
    let mixing = benchmarker.mixing_time(&initial, &times, 0.05);
    println!("Mixing time ε=0.05: {:?}", mixing.mixing_time);

    let hitting = benchmarker.hitting_time_benchmark(0.25, 20);
    println!(
        "Average hitting steps (quantum/classical): {:.3} / {:.3}",
        hitting.quantum_average_steps, hitting.classical_average_steps
    );
    println!(
        "Mean success probability: {:.3}",
        hitting.mean_success_probability
    );
    println!("Speedup factor: {:.2}x", hitting.speedup_factor);

    let projection = metatron_qso::quantum_walk::krylov::krylov_projection(
        qso.hamiltonian(),
        &initial,
        8,
        1e-10,
    );
    let krylov_state = projection.evolve(1.0);
    let exact = qso.evolve_state(&initial, 1.0);
    let diff = krylov_state.state.amplitudes() - exact.amplitudes();
    println!("Krylov residual norm: {:.3e}", krylov_state.residual_norm);
    println!("Krylov vs exact state error: {:.3e}", diff.norm());

    let scattering_channel = metatron_qso::quantum_walk::scattering::ScatteringChannel::new(
        metatron_qso::quantum::state::StateVector::from_element(num_complex::Complex64::new(
            1.0, 0.0,
        )),
        0.15,
    );
    let scattering = metatron_qso::quantum_walk::scattering::scattering_matrix(
        qso.hamiltonian(),
        &[scattering_channel],
        0.5,
        0.05,
        0.1,
    );
    println!(
        "Scattering trace @ E=0.5: {:.3} + {:.3}i",
        scattering.matrix.trace().re,
        scattering.matrix.trace().im
    );
}
