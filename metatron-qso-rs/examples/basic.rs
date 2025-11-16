use metatron_qso::prelude::*;

fn main() {
    let params = QSOParameters::default();
    let qso = QuantumStateOperator::new(params);

    let report = qso.analyze();
    println!("Metatron QSO analysis:\n{:#?}", report);

    let initial = qso.basis_state(0);
    let evolved = qso.evolve_state(&initial, 1.0);
    println!(
        "Initial probabilities (first 5): {:?}",
        &initial.probabilities()[..5]
    );
    println!(
        "Evolved probabilities (first 5): {:?}",
        &evolved.probabilities()[..5]
    );

    let (_times, phases) = qso.simulate_resonators((0.0, 1.0), 0.05);
    println!("Simulated {} resonator snapshots", phases.len());
}
