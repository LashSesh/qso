//! ADAPT-VQE (Grimsley, Economou, Barnes and Mayhall, Nat. Commun. 10,
//! 3007 (2019)).
//!
//! Starting from a reference circuit, every iteration computes the energy
//! gradient `∂E/∂θ = ⟨ψ|[H, τ_k]|ψ⟩ = 2 Re⟨Hψ|τ_kψ⟩` of appending
//! `exp(θτ_k)` for every operator `τ_k` of a pool, appends the operator
//! with the largest `|∂E/∂θ|`, and re-optimises all parameters, starting
//! from the previous ones and θ = 0 for the new operator. It stops when
//! the largest gradient falls below a tolerance or after a maximum number
//! of operators. An operator may be chosen more than once.

use serde::{Deserialize, Serialize};

use super::ansatz::{
    Excitation, append_generator, apply_operator, uccsd_excitations, upccgsd_excitations,
};
use super::mapping::QubitEncoding;
use super::molecule::MolecularIntegrals;
use crate::algorithms::Vqe;
use crate::error::{QsoError, Result};
use crate::linalg::inner;
use crate::optimize::Options;
use crate::quantum::{Circuit, PauliSum, QubitOperator, StateVector};

/// Settings of [`adapt_vqe`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AdaptOptions {
    /// Stop when every pool gradient is below this (Hartree).
    pub gradient_tolerance: f64,
    /// Stop after this many operators.
    pub max_operators: usize,
    /// Optimiser settings of each re-optimisation (L-BFGS).
    pub optimizer: Options,
}

impl Default for AdaptOptions {
    fn default() -> Self {
        Self {
            gradient_tolerance: 1e-3,
            max_operators: 40,
            optimizer: Options::default(),
        }
    }
}

impl AdaptOptions {
    pub fn with_gradient_tolerance(mut self, tol: f64) -> Self {
        self.gradient_tolerance = tol;
        self
    }

    pub fn with_max_operators(mut self, n: usize) -> Self {
        self.max_operators = n;
        self
    }
}

/// Outcome of [`adapt_vqe`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AdaptResult {
    pub energy: f64,
    pub parameters: Vec<f64>,
    /// Labels of the chosen pool operators, in order.
    pub operators: Vec<String>,
    /// Energy after each iteration.
    pub energies: Vec<f64>,
    /// Largest pool gradient at the start of each iteration (and at the end).
    pub max_gradients: Vec<f64>,
    /// Whether the gradient tolerance was reached.
    pub converged: bool,
    /// The final ansatz (reference circuit plus chosen operators).
    pub circuit: Circuit,
}

/// A pool operator: label and anti-Hermitian qubit generator.
pub type PoolOperator = (String, QubitOperator);

/// The pool of `excitations` mapped under `encoding` for `mol`.
pub fn excitation_pool(
    mol: &MolecularIntegrals,
    excitations: &[Excitation],
    encoding: QubitEncoding,
) -> Result<Vec<PoolOperator>> {
    excitations
        .iter()
        .map(|e| {
            Ok((
                e.label.clone(),
                e.qubit_generator(encoding, mol.n_spin_orbitals(), mol.n_alpha(), mol.n_beta())?,
            ))
        })
        .collect()
}

/// Gradients `2 Re⟨Hψ|τ_kψ⟩` of all pool operators at `psi`.
pub fn pool_gradients(
    hamiltonian: &PauliSum,
    pool: &[PoolOperator],
    psi: &StateVector,
) -> Result<Vec<f64>> {
    let h_psi = hamiltonian.apply(psi)?;
    let n = psi.n_qubits();
    Ok(pool
        .iter()
        .map(|(_, tau)| 2.0 * inner(&h_psi, &apply_operator(tau, psi.amplitudes(), n)).re)
        .collect())
}

/// ADAPT-VQE for `hamiltonian` from the parameter-free `reference` circuit
/// with operators from `pool`.
pub fn adapt_vqe(
    hamiltonian: &PauliSum,
    reference: &Circuit,
    pool: &[PoolOperator],
    options: &AdaptOptions,
) -> Result<AdaptResult> {
    if reference.num_parameters() != 0 {
        return Err(QsoError::invalid("reference", "must not have parameters"));
    }
    if pool.is_empty() {
        return Err(QsoError::invalid("pool", "needs at least one operator"));
    }
    let n = reference.n_qubits();
    if hamiltonian.min_qubits() > n || pool.iter().any(|(_, t)| t.min_qubits() > n) {
        return Err(QsoError::invalid(
            "pool",
            "operators act on more qubits than the reference circuit",
        ));
    }
    if options.gradient_tolerance.is_nan() || options.gradient_tolerance < 0.0 {
        return Err(QsoError::invalid(
            "gradient_tolerance",
            "must be non-negative",
        ));
    }
    let mut circuit = reference.clone();
    let mut theta: Vec<f64> = Vec::new();
    let mut energy = hamiltonian.expectation(&StateVector::from_circuit(&circuit, &[])?)?;
    let mut result = AdaptResult {
        energy,
        parameters: Vec::new(),
        operators: Vec::new(),
        energies: Vec::new(),
        max_gradients: Vec::new(),
        converged: false,
        circuit: circuit.clone(),
    };
    loop {
        let psi = StateVector::from_circuit(&circuit, &theta)?;
        let grads = pool_gradients(hamiltonian, pool, &psi)?;
        let (best, g) = grads
            .iter()
            .enumerate()
            .map(|(k, g)| (k, g.abs()))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("pool is not empty");
        result.max_gradients.push(g);
        if g < options.gradient_tolerance {
            result.converged = true;
            break;
        }
        if result.operators.len() >= options.max_operators {
            break;
        }
        let p = circuit.new_parameter();
        append_generator(&mut circuit, &pool[best].1, p)?;
        theta.push(0.0);
        let vqe = Vqe::new(hamiltonian.clone(), circuit.clone())?
            .with_options(options.optimizer.clone())
            .with_initial_parameters(theta.clone())?;
        let run = vqe.run()?;
        theta = run.parameters;
        energy = run.energy;
        result.operators.push(pool[best].0.clone());
        result.energies.push(energy);
    }
    result.energy = energy;
    result.parameters = theta;
    result.circuit = circuit;
    Ok(result)
}

/// ADAPT-VQE for a molecule: Hartree–Fock reference and the pool of
/// UCCSD excitations (`generalized = false`) or of generalised pair
/// doubles and spin-adapted singles.
pub fn adapt_vqe_molecule(
    mol: &MolecularIntegrals,
    encoding: QubitEncoding,
    generalized: bool,
    options: &AdaptOptions,
) -> Result<AdaptResult> {
    let h = mol.qubit_hamiltonian(encoding)?;
    let excitations = if generalized {
        upccgsd_excitations(mol.n_orbitals())
    } else {
        uccsd_excitations(mol.n_orbitals(), mol.n_alpha(), mol.n_beta())
    };
    let pool = excitation_pool(mol, &excitations, encoding)?;
    adapt_vqe(&h, &mol.hartree_fock_circuit(encoding)?, &pool, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chemistry::mapping::Mapping;

    #[test]
    fn pool_gradients_match_finite_differences() {
        let mol = MolecularIntegrals::from_fcidump(include_str!(
            "../../examples/molecules/h2_sto3g.fcidump"
        ))
        .unwrap();
        let encoding = QubitEncoding::new(Mapping::JordanWigner);
        let h = mol.qubit_hamiltonian(encoding).unwrap();
        let ex = upccgsd_excitations(2);
        let pool = excitation_pool(&mol, &ex, encoding).unwrap();
        // A generic state, so that every operator has a gradient.
        let mut base = mol.hartree_fock_circuit(encoding).unwrap();
        for q in 0..4 {
            base.ry(q, 0.3 + 0.2 * q as f64).rz(q, 0.7 - 0.3 * q as f64);
        }
        base.cx(0, 1).cx(2, 3).cx(1, 2);
        let psi = StateVector::from_circuit(&base, &[]).unwrap();
        let grads = pool_gradients(&h, &pool, &psi).unwrap();
        for ((_, tau), g) in pool.iter().zip(&grads) {
            let mut c = base.clone();
            let p = c.new_parameter();
            append_generator(&mut c, tau, p).unwrap();
            let e = |t: f64| {
                h.expectation(&StateVector::from_circuit(&c, &[t]).unwrap())
                    .unwrap()
            };
            let fd = (e(1e-5) - e(-1e-5)) / 2e-5;
            assert!((fd - g).abs() < 1e-7, "{fd} vs {g}");
        }
        assert!(grads.iter().all(|g| g.abs() > 1e-3), "{grads:?}");
    }

    #[test]
    fn adapt_reaches_full_ci_for_h2_with_one_operator() {
        let mol = MolecularIntegrals::from_fcidump(include_str!(
            "../../examples/molecules/h2_sto3g.fcidump"
        ))
        .unwrap();
        let fci = mol.fci_energies(1).unwrap()[0];
        let r = adapt_vqe_molecule(
            &mol,
            QubitEncoding::parity_reduced(),
            false,
            &AdaptOptions::default().with_gradient_tolerance(1e-6),
        )
        .unwrap();
        assert!(r.converged);
        assert_eq!(r.operators, vec!["0,2->1,3".to_string()]);
        assert!((r.energy - fci).abs() < 1e-9);
        assert!(
            adapt_vqe(
                &PauliSum::new(),
                &Circuit::new(2),
                &[],
                &AdaptOptions::default()
            )
            .is_err()
        );
    }
}
