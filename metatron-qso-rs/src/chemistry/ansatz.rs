//! Chemistry-inspired ansätze: unitary coupled cluster.
//!
//! An excitation moves electrons from the spin orbitals `from` to `to`:
//! `T = a†_to… a_from…`, and `exp(θ(T − T†))` is unitary. Mapped to qubits,
//! `T − T† = i Σ_k a_k P_k` with real `a_k`, and QSO applies
//! `Π_k exp(iθ a_k P_k)` — one Pauli rotation per string, all sharing the
//! parameter θ. For single and double excitations the strings commute, so
//! the product is exact.
//!
//! * [`uccsd`]: all spin-conserving single and double excitations from the
//!   occupied to the virtual orbitals of the Hartree–Fock determinant, one
//!   parameter each (Peruzzo et al., Nat. Commun. 5, 4213 (2014); Romero et
//!   al., Quantum Sci. Technol. 4, 014008 (2019)).
//! * [`k_upccgsd`]: `k` layers of generalised pair doubles
//!   `a†_pα a†_pβ a_qβ a_qα` and spin-adapted generalised singles
//!   `a†_pα a_qα + a†_pβ a_qβ` between all orbitals `p > q` (Lee, Huggins,
//!   Head-Gordon and Whaley, J. Chem. Theory Comput. 15, 311 (2019)).
//!
//! Both circuits start with the Hartree–Fock state, so θ = 0 gives the
//! Hartree–Fock energy.

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::fermion::{FermionOperator, Ladder};
use super::mapping::QubitEncoding;
use super::molecule::MolecularIntegrals;
use crate::error::{QsoError, Result};
use crate::quantum::{Circuit, Parameter, QubitOperator};

/// Largest number of excitations in one circuit.
pub const MAX_EXCITATIONS: usize = 100_000;

/// A fermionic excitation generator `Σ_j c_j T_j − h.c.` (several products
/// share one parameter in spin-adapted generators).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Excitation {
    /// `(coefficient, from, to)`: `T = a†_to[0] a†_to[1] … a_from[…] a_from[0]`.
    pub parts: Vec<(f64, Vec<usize>, Vec<usize>)>,
    /// Human-readable label, e.g. `"0,6->2,8"`.
    pub label: String,
}

impl Excitation {
    /// A single product `a†_to… a_from…` with coefficient 1.
    pub fn new(from: &[usize], to: &[usize]) -> Result<Self> {
        if from.len() != to.len() || from.is_empty() {
            return Err(QsoError::invalid(
                "excitation",
                "need equally many (and at least one) source and target orbitals",
            ));
        }
        let label = format!(
            "{}->{}",
            from.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(","),
            to.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        Ok(Self {
            parts: vec![(1.0, from.to_vec(), to.to_vec())],
            label,
        })
    }

    /// `T − T†` as a fermionic operator.
    pub fn generator(&self) -> Result<FermionOperator> {
        let mut t = FermionOperator::new();
        for (c, from, to) in &self.parts {
            let ops: Vec<Ladder> = to
                .iter()
                .map(|&p| Ladder::create(p))
                .chain(from.iter().rev().map(|&p| Ladder::annihilate(p)))
                .collect();
            t.add_term(*c, &ops)?;
        }
        Ok(&t - &t.adjoint())
    }

    /// The qubit form `T − T†` under `encoding`; anti-Hermitian.
    pub fn qubit_generator(
        &self,
        encoding: QubitEncoding,
        n_spin_orbitals: usize,
        n_alpha: usize,
        n_beta: usize,
    ) -> Result<QubitOperator> {
        encoding.map(&self.generator()?, n_spin_orbitals, n_alpha, n_beta)
    }
}

/// Append `exp(θ τ)` for an anti-Hermitian `τ = i Σ a_k P_k` with the
/// circuit parameter `theta`.
pub fn append_generator(c: &mut Circuit, tau: &QubitOperator, theta: Parameter) -> Result<()> {
    for (coefficient, string) in tau.terms() {
        if coefficient.re.abs() > 1e-10 {
            return Err(QsoError::invalid(
                "generator",
                "must be anti-Hermitian (imaginary coefficients)",
            ));
        }
        if string.is_identity() || coefficient.im == 0.0 {
            continue;
        }
        // exp(iθaP) = exp(−i(−2aθ)P/2)
        c.pauli_rotation(&string, theta * (-2.0 * coefficient.im))?;
    }
    Ok(())
}

/// The spin-conserving single and double excitations from occupied to
/// virtual spin orbitals of the Hartree–Fock determinant (block order).
pub fn uccsd_excitations(n_orbitals: usize, n_alpha: usize, n_beta: usize) -> Vec<Excitation> {
    let m = n_orbitals;
    let occ_a: Vec<usize> = (0..n_alpha).collect();
    let vir_a: Vec<usize> = (n_alpha..m).collect();
    let occ_b: Vec<usize> = (m..m + n_beta).collect();
    let vir_b: Vec<usize> = (m + n_beta..2 * m).collect();
    let mut out = Vec::new();
    for (occ, vir) in [(&occ_a, &vir_a), (&occ_b, &vir_b)] {
        for &i in occ {
            for &a in vir {
                out.push(Excitation::new(&[i], &[a]).expect("valid"));
            }
        }
    }
    // Same-spin doubles i < j → a < b.
    for (occ, vir) in [(&occ_a, &vir_a), (&occ_b, &vir_b)] {
        for (x, &i) in occ.iter().enumerate() {
            for &j in &occ[x + 1..] {
                for (y, &a) in vir.iter().enumerate() {
                    for &b in &vir[y + 1..] {
                        out.push(Excitation::new(&[i, j], &[a, b]).expect("valid"));
                    }
                }
            }
        }
    }
    // Opposite-spin doubles iα jβ → aα bβ.
    for &i in &occ_a {
        for &j in &occ_b {
            for &a in &vir_a {
                for &b in &vir_b {
                    out.push(Excitation::new(&[i, j], &[a, b]).expect("valid"));
                }
            }
        }
    }
    out
}

/// Generalised pair doubles and spin-adapted generalised singles for all
/// orbital pairs `p > q` (one layer of k-UpCCGSD).
pub fn upccgsd_excitations(n_orbitals: usize) -> Vec<Excitation> {
    let m = n_orbitals;
    let mut out = Vec::new();
    for p in 0..m {
        for q in 0..p {
            out.push(Excitation {
                parts: vec![(1.0, vec![q, q + m], vec![p, p + m])],
                label: format!("pair {q}->{p}"),
            });
        }
    }
    for p in 0..m {
        for q in 0..p {
            out.push(Excitation {
                parts: vec![(1.0, vec![q], vec![p]), (1.0, vec![q + m], vec![p + m])],
                label: format!("single {q}->{p}"),
            });
        }
    }
    out
}

/// The Hartree–Fock circuit followed by `exp(θ_k τ_k)` for `excitations`
/// (one parameter each, in order).
pub fn excitation_circuit(
    mol: &MolecularIntegrals,
    excitations: &[Excitation],
    encoding: QubitEncoding,
) -> Result<Circuit> {
    let n = mol.n_spin_orbitals();
    let nq = encoding.n_qubits(n);
    if nq > crate::quantum::MAX_QUBITS {
        return Err(QsoError::TooManyQubits {
            what: "excitation ansatz",
            max: crate::quantum::MAX_QUBITS,
            requested: nq,
        });
    }
    if excitations.len() > MAX_EXCITATIONS {
        return Err(QsoError::invalid(
            "excitations",
            format!("at most {MAX_EXCITATIONS} excitations"),
        ));
    }
    let mut c = mol.hartree_fock_circuit(encoding)?;
    for e in excitations {
        let tau = e.qubit_generator(encoding, n, mol.n_alpha(), mol.n_beta())?;
        let theta = c.new_parameter();
        append_generator(&mut c, &tau, theta)?;
    }
    Ok(c)
}

/// UCCSD circuit for `mol` under `encoding`.
pub fn uccsd(mol: &MolecularIntegrals, encoding: QubitEncoding) -> Result<Circuit> {
    let ex = uccsd_excitations(mol.n_orbitals(), mol.n_alpha(), mol.n_beta());
    excitation_circuit(mol, &ex, encoding)
}

/// k-UpCCGSD circuit for `mol` under `encoding` (`k ≥ 1` layers, each
/// with its own parameters).
pub fn k_upccgsd(mol: &MolecularIntegrals, k: usize, encoding: QubitEncoding) -> Result<Circuit> {
    if k == 0 {
        return Err(QsoError::invalid("k", "need at least one layer"));
    }
    let layer = upccgsd_excitations(mol.n_orbitals());
    if layer.len().saturating_mul(k) > MAX_EXCITATIONS {
        return Err(QsoError::invalid(
            "k",
            format!("{k} layers exceed {MAX_EXCITATIONS} excitations"),
        ));
    }
    let ex: Vec<Excitation> = (0..k).flat_map(|_| layer.iter().cloned()).collect();
    excitation_circuit(mol, &ex, encoding)
}

/// `op|ψ⟩` for a qubit operator on the amplitudes `psi`.
pub(crate) fn apply_operator(op: &QubitOperator, psi: &[C64], n_qubits: usize) -> Vec<C64> {
    let mut out = vec![C64::new(0.0, 0.0); psi.len()];
    for (c, s) in op.terms() {
        let (x, z) = (s.x_mask() as usize, s.z_mask() as usize);
        let y = (x & z).count_ones();
        let base = c * match y % 4 {
            0 => C64::new(1.0, 0.0),
            1 => C64::new(0.0, 1.0),
            2 => C64::new(-1.0, 0.0),
            _ => C64::new(0.0, -1.0),
        };
        debug_assert!(s.min_qubits() <= n_qubits);
        for (b, amp) in psi.iter().enumerate() {
            if *amp == C64::new(0.0, 0.0) {
                continue;
            }
            let sign = if (b & z).count_ones() % 2 == 1 {
                -1.0
            } else {
                1.0
            };
            out[b ^ x] += base * sign * amp;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::Vqe;
    use crate::chemistry::mapping::Mapping;
    use crate::chemistry::molecule::MolecularIntegrals;
    use crate::quantum::StateVector;

    fn h2() -> MolecularIntegrals {
        MolecularIntegrals::from_fcidump(include_str!("../../examples/molecules/h2_sto3g.fcidump"))
            .unwrap()
    }

    #[test]
    fn generator_strings_commute_and_hartree_fock_is_theta_zero() {
        let mol = MolecularIntegrals::from_fcidump(include_str!(
            "../../examples/molecules/lih_sto3g.fcidump"
        ))
        .unwrap();
        let ex = uccsd_excitations(6, 2, 2);
        assert_eq!(ex.len(), 2 * 2 * 4 + 2 * 6 + 2 * 4 * 2 * 4);
        for encoding in [
            QubitEncoding::new(Mapping::JordanWigner),
            QubitEncoding::new(Mapping::BravyiKitaev),
            QubitEncoding::parity_reduced(),
        ] {
            for e in ex.iter().chain(&upccgsd_excitations(6)) {
                let tau = e.qubit_generator(encoding, 12, 2, 2).unwrap();
                assert!(!tau.is_empty());
                let strings: Vec<_> = tau.terms().map(|(_, s)| s).collect();
                for a in &strings {
                    for b in &strings {
                        assert!(a.commutes_with(b), "{} {encoding:?}", e.label);
                    }
                }
            }
        }
        let c = uccsd(&mol, QubitEncoding::parity_reduced()).unwrap();
        let h = mol
            .qubit_hamiltonian(QubitEncoding::parity_reduced())
            .unwrap();
        let psi = StateVector::from_circuit(&c, &vec![0.0; c.num_parameters()]).unwrap();
        assert!((h.expectation(&psi).unwrap() - mol.hartree_fock_energy()).abs() < 1e-9);
    }

    #[test]
    fn excitation_rotation_matches_the_matrix_exponential() {
        // exp(θ(T − T†)) applied to the HF state equals the dense exponential.
        let mol = h2();
        let encoding = QubitEncoding::new(Mapping::BravyiKitaev);
        let e = Excitation::new(&[0, 2], &[1, 3]).unwrap();
        let tau = e.qubit_generator(encoding, 4, 1, 1).unwrap();
        let mut c = mol.hartree_fock_circuit(encoding).unwrap();
        let theta = c.new_parameter();
        append_generator(&mut c, &tau, theta).unwrap();
        let t = 0.37;
        let psi = StateVector::from_circuit(&c, &[t]).unwrap();
        // Dense: G = iτ Hermitian, exp(θτ) = exp(−iθG).
        let g = tau.scaled(C64::new(0.0, 1.0)).to_pauli_sum(1e-12).unwrap();
        let u = crate::linalg::expm_hermitian(&g.to_dense(4).unwrap(), t);
        let hf =
            StateVector::from_circuit(&mol.hartree_fock_circuit(encoding).unwrap(), &[]).unwrap();
        let expected = &u * nalgebra::DVector::from_column_slice(hf.amplitudes());
        for (a, b) in psi.amplitudes().iter().zip(expected.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
        // apply_operator agrees with the dense matrix.
        let tpsi = apply_operator(&tau, hf.amplitudes(), 4);
        let dense = g.to_dense(4).unwrap() * nalgebra::DVector::from_column_slice(hf.amplitudes());
        for (a, b) in tpsi.iter().zip(dense.iter()) {
            assert!((a - b * C64::new(0.0, -1.0)).norm() < 1e-12);
        }
    }

    #[test]
    fn uccsd_and_upccgsd_reach_full_ci_for_h2() {
        let mol = h2();
        let fci = mol.fci_energies(1).unwrap()[0];
        for encoding in [
            QubitEncoding::new(Mapping::JordanWigner),
            QubitEncoding::parity_reduced(),
        ] {
            let h = mol.qubit_hamiltonian(encoding).unwrap();
            for c in [
                uccsd(&mol, encoding).unwrap(),
                k_upccgsd(&mol, 1, encoding).unwrap(),
            ] {
                let n = c.num_parameters();
                let res = Vqe::new(h.clone(), c)
                    .unwrap()
                    .with_initial_parameters(vec![0.0; n])
                    .unwrap()
                    .run()
                    .unwrap();
                assert!(
                    (res.energy - fci).abs() < 1e-8,
                    "{encoding:?}: {} vs {fci}",
                    res.energy
                );
            }
        }
        assert!(k_upccgsd(&mol, 0, QubitEncoding::default()).is_err());
        assert!(k_upccgsd(&mol, usize::MAX, QubitEncoding::default()).is_err());
        assert!(Excitation::new(&[0], &[1, 2]).is_err());
    }
}
