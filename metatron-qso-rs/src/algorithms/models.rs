//! Model Hamiltonians as [`PauliSum`]s.

use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::graph::Graph;
use crate::quantum::{Pauli, PauliString, PauliSum};

fn pair(a: usize, b: usize, p: Pauli) -> PauliString {
    PauliString::new(&[(a, p), (b, p)]).expect("distinct qubits below 64")
}

fn bonds(n: usize, periodic: bool) -> Vec<(usize, usize)> {
    let mut b: Vec<(usize, usize)> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
    if periodic && n > 2 {
        b.push((n - 1, 0));
    }
    b
}

/// Transverse-field Ising chain `H = −J Σ Z_i Z_{i+1} − h Σ X_i`.
pub fn transverse_field_ising(n: usize, j: f64, h: f64, periodic: bool) -> PauliSum {
    let mut terms = Vec::new();
    for (a, b) in bonds(n, periodic) {
        terms.push((-j, pair(a, b, Pauli::Z)));
    }
    for q in 0..n {
        terms.push((-h, PauliString::single(q, Pauli::X)));
    }
    PauliSum::from_terms(terms)
}

/// XXZ Heisenberg chain `H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Δ Z_i Z_{i+1})`.
pub fn heisenberg_xxz(n: usize, j: f64, delta: f64, periodic: bool) -> PauliSum {
    let mut terms = Vec::new();
    for (a, b) in bonds(n, periodic) {
        terms.push((j, pair(a, b, Pauli::X)));
        terms.push((j, pair(a, b, Pauli::Y)));
        terms.push((j * delta, pair(a, b, Pauli::Z)));
    }
    PauliSum::from_terms(terms)
}

/// MaxCut cost operator `C = Σ_{(u,v)} w_uv (1 − Z_u Z_v)/2`, whose
/// eigenvalue on basis state `|b⟩` is the weight of the cut `b`.
pub fn maxcut_cost(graph: &Graph) -> Result<PauliSum> {
    if graph.num_nodes() > 64 {
        return Err(QsoError::TooManyQubits {
            what: "Pauli-sum MaxCut cost",
            max: 64,
            requested: graph.num_nodes(),
        });
    }
    let mut terms = Vec::new();
    for &(u, v, w) in graph.edges() {
        terms.push((w / 2.0, PauliString::identity()));
        terms.push((-w / 2.0, pair(u, v, Pauli::Z)));
    }
    Ok(PauliSum::from_terms(terms))
}

/// A qubit Hamiltonian of a molecule with its reference data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MolecularHamiltonian {
    pub name: String,
    pub n_qubits: usize,
    /// Electronic Hamiltonian (Hartree).
    pub hamiltonian: PauliSum,
    /// Nuclear repulsion energy to add to electronic energies (Hartree).
    pub nuclear_repulsion: f64,
    /// Literature ground-state total energy (Hartree).
    pub reference_total_energy: f64,
}

/// H₂ in the STO-3G basis at bond length 0.735 Å, parity mapping with
/// two-qubit reduction (the standard two-qubit H₂ benchmark, e.g.
/// O'Malley et al., Phys. Rev. X 6, 031007, 2016).
///
/// Exact diagonalisation gives an electronic ground energy of
/// −1.857275 Ha; adding the nuclear repulsion 0.719969 Ha yields the full-CI
/// total energy −1.137306 Ha.
pub fn h2_sto3g() -> MolecularHamiltonian {
    let hamiltonian: PauliSum = "-1.052373245772859 \
         + 0.39793742484318045 Z0 \
         - 0.39793742484318045 Z1 \
         - 0.01128010425623538 Z0 Z1 \
         + 0.18093119978423156 X0 X1"
        .parse()
        .expect("valid Pauli expression");
    MolecularHamiltonian {
        name: "H2 / STO-3G / 0.735 Å (parity, 2 qubits)".into(),
        n_qubits: 2,
        hamiltonian,
        nuclear_repulsion: 0.7199689944489797,
        reference_total_energy: -1.137306,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::generators;
    use crate::linalg::eigh;

    #[test]
    fn h2_matches_full_ci_reference() {
        let h2 = h2_sto3g();
        let (vals, _) = eigh(&h2.hamiltonian.to_dense(2).unwrap());
        assert!((vals[0] - (-1.8572750302023795)).abs() < 1e-12);
        assert!((vals[0] + h2.nuclear_repulsion - h2.reference_total_energy).abs() < 1e-6);
    }

    #[test]
    fn maxcut_cost_diagonal_counts_cut_edges() {
        let g = generators::cycle(5);
        let c = maxcut_cost(&g).unwrap();
        let diag = c.diagonal(5).unwrap().unwrap();
        // b = 0b01010 cuts edges (0,1),(1,2),(2,3),(3,4) → 4.
        assert!((diag[0b01010] - 4.0).abs() < 1e-12);
        assert_eq!(diag.iter().cloned().fold(0.0, f64::max), 4.0);
        assert_eq!(diag[0], 0.0);
    }

    #[test]
    fn tfim_limits() {
        // h = 0: classical ferromagnet, E0 = −J(n−1) (open chain).
        let e0 = transverse_field_ising(6, 1.0, 0.0, false)
            .ground_state_energy(6)
            .unwrap();
        assert!((e0 + 5.0).abs() < 1e-10);
        // J = 0: free spins, E0 = −h·n.
        let e0 = transverse_field_ising(6, 0.0, 0.7, false)
            .ground_state_energy(6)
            .unwrap();
        assert!((e0 + 4.2).abs() < 1e-8);
        // Two-site Heisenberg: singlet energy −3J.
        let e0 = heisenberg_xxz(2, 1.0, 1.0, false)
            .ground_state_energy(2)
            .unwrap();
        assert!((e0 + 3.0).abs() < 1e-10);
    }
}
