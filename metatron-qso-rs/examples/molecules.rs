//! Molecules on qubits: mappings, full CI, UCCSD, ADAPT-VQE, excited
//! states and time evolution.
//!
//! Run with `cargo run --release --example molecules`. The FCIDUMP files in
//! `examples/molecules/` were written by PySCF (see `generate.py` there),
//! which also computed the reference energies in `reference.json`.

use metatron_qso::Result;
use metatron_qso::algorithms::Vqe;
use metatron_qso::algorithms::ansatz::{Entanglement, efficient_su2};
use metatron_qso::algorithms::excited::{ExcitedStateOptions, vqd};
use metatron_qso::algorithms::time_evolution::{trotter_circuit, trotter_error_bound};
use metatron_qso::chemistry::{
    AdaptOptions, Mapping, MolecularIntegrals, QubitEncoding, adapt_vqe_molecule, k_upccgsd, uccsd,
};
use metatron_qso::linalg::expm_hermitian;
use metatron_qso::quantum::{PauliSum, StateVector};

fn main() -> Result<()> {
    let lih = MolecularIntegrals::from_fcidump(include_str!("molecules/lih_sto3g.fcidump"))?;
    let fci = lih.fci_energies(3)?;
    println!(
        "LiH / STO-3G at 1.595 Å: {} spatial orbitals, {}+{} electrons\n",
        lih.n_orbitals(),
        lih.n_alpha(),
        lih.n_beta()
    );
    println!("| encoding | qubits | Pauli terms | lowest energy in the determinant space |");
    println!("|---|---:|---:|---:|");
    for (label, encoding) in [
        ("Jordan–Wigner", QubitEncoding::new(Mapping::JordanWigner)),
        ("parity", QubitEncoding::new(Mapping::Parity)),
        ("Bravyi–Kitaev", QubitEncoding::new(Mapping::BravyiKitaev)),
        (
            "parity, two-qubit reduction",
            QubitEncoding::parity_reduced(),
        ),
    ] {
        let h = lih.qubit_hamiltonian(encoding)?;
        println!(
            "| {label} | {} | {} | {:.10} |",
            encoding.n_qubits(lih.n_spin_orbitals()),
            h.num_terms(),
            lih.fci_energies_with(encoding, 1)?[0]
        );
    }

    let encoding = QubitEncoding::parity_reduced();
    let h = lih.qubit_hamiltonian(encoding)?;
    println!("\n| method | parameters | energy (Ha) | error vs full CI |");
    println!("|---|---:|---:|---:|");
    println!(
        "| Hartree–Fock | 0 | {:.8} | {:.2e} |",
        lih.hartree_fock_energy(),
        lih.hartree_fock_energy() - fci[0]
    );
    for (label, ansatz) in [
        ("UCCSD", uccsd(&lih, encoding)?),
        ("1-UpCCGSD", k_upccgsd(&lih, 1, encoding)?),
    ] {
        let p = ansatz.num_parameters();
        let r = Vqe::new(h.clone(), ansatz)?
            .with_initial_parameters(vec![0.0; p])?
            .run()?;
        println!(
            "| {label} | {p} | {:.8} | {:.2e} |",
            r.energy,
            r.energy - fci[0]
        );
    }
    let adapt = adapt_vqe_molecule(&lih, encoding, false, &AdaptOptions::default())?;
    println!(
        "| ADAPT-VQE (UCCSD pool) | {} | {:.8} | {:.2e} |",
        adapt.parameters.len(),
        adapt.energy,
        adapt.energy - fci[0]
    );
    println!(
        "| full CI (PySCF reference −7.882401932) | — | {:.8} | 0 |",
        fci[0]
    );

    // Excited states of H2 on two qubits (parity, two-qubit reduction).
    let h2 = MolecularIntegrals::from_fcidump(include_str!("molecules/h2_sto3g.fcidump"))?;
    let h2_ham = h2.qubit_hamiltonian(encoding)?;
    let exact = h2.fci_energies(4)?;
    let states = vqd(
        &h2_ham,
        &efficient_su2(2, 2, Entanglement::Linear),
        4,
        &ExcitedStateOptions::default().with_restarts(6).with_seed(2),
    )?;
    println!("\nH2 levels (Sz = 0, two qubits): VQD against full CI");
    for (k, (e, f)) in states.energies.iter().zip(&exact).enumerate() {
        println!("  E{k} = {e:.10}  (full CI {f:.10})");
    }

    // Time evolution under the H2 Hamiltonian: product formulas.
    let traceless = PauliSum::from_terms(
        h2_ham
            .terms()
            .iter()
            .filter(|(_, s)| !s.is_identity())
            .copied(),
    );
    let exact_u = expm_hermitian(&traceless.to_dense(2)?, 2.0);
    println!("\nexp(−iHt) for H2, t = 2, 4 steps: largest column error and bound");
    for order in [1, 2, 4] {
        let c = trotter_circuit(&h2_ham, 2, 2.0, 4, order)?;
        let mut worst: f64 = 0.0;
        for col in 0..4 {
            let mut psi = StateVector::basis(2, col)?;
            psi.apply_circuit(&c, &[])?;
            let err: f64 = psi
                .amplitudes()
                .iter()
                .enumerate()
                .map(|(r, a)| (a - exact_u[(r, col)]).norm_sqr())
                .sum::<f64>()
                .sqrt();
            worst = worst.max(err);
        }
        println!(
            "  order {order}: {} gates, error {worst:.2e}, bound {:.2e}",
            c.len(),
            trotter_error_bound(&h2_ham, 2.0, 4, order)?
        );
    }
    Ok(())
}
