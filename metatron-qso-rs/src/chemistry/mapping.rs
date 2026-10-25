//! Fermion-to-qubit mappings.
//!
//! All three mappings are linear binary transforms (Seeley, Richard and
//! Love, J. Chem. Phys. 137, 224109 (2012)): the occupation vector `n` of
//! the modes is stored as qubit vector `b = A n (mod 2)` with
//!
//! * Jordan–Wigner: `A = I` — qubit `j` holds `n_j`;
//! * parity: `A` lower triangular of ones — qubit `j` holds `n_0 ⊕ … ⊕ n_j`;
//! * Bravyi–Kitaev: qubit `j` holds the parity of the Fenwick-tree range
//!   ending at `j` (modes `j − lowbit(j + 1) + 1, …, j`).
//!
//! A creation operator then flips the qubits of column `j` of `A`
//! (`X_F`), takes the sign `(−1)^{n_0 + … + n_{j−1}}` from the qubits
//! `P(j)` that encode that parity (`Z_P`), and requires `n_j = 0`, a
//! projector `(1 + Z_N)/2` on the qubits `N(j)` that encode `n_j`:
//! `a_j† = X_F Z_P (1 + Z_N)/2`.
//!
//! Spin orbitals of molecules are ordered in blocks: modes `0 … m−1` are the
//! spin-up (α) orbitals, `m … 2m−1` the spin-down (β) ones. Under the parity
//! mapping, qubit `m − 1` then holds the parity of the α electrons and
//! qubit `2m − 1` the parity of all electrons; [`two_qubit_reduction`]
//! replaces both by their eigenvalues and removes them.

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::fermion::{FermionOperator, MAX_MODES};
use crate::error::{QsoError, Result};
use crate::quantum::{Pauli, PauliString, PauliSum, QubitOperator};

/// A fermion-to-qubit mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Mapping {
    #[default]
    JordanWigner,
    Parity,
    BravyiKitaev,
}

impl Mapping {
    /// Parse `"jordan_wigner"` (or `"jw"`), `"parity"` or
    /// `"bravyi_kitaev"` (or `"bk"`).
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().replace(['-', ' '], "_").as_str() {
            "jordan_wigner" | "jw" => Ok(Mapping::JordanWigner),
            "parity" => Ok(Mapping::Parity),
            "bravyi_kitaev" | "bk" => Ok(Mapping::BravyiKitaev),
            _ => Err(QsoError::invalid(
                "mapping",
                format!("unknown mapping '{name}' (jordan_wigner, parity, bravyi_kitaev)"),
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Mapping::JordanWigner => "jordan_wigner",
            Mapping::Parity => "parity",
            Mapping::BravyiKitaev => "bravyi_kitaev",
        }
    }

    /// Row `j` of the transform `A` as a bit mask over modes.
    fn row(self, j: usize) -> u64 {
        match self {
            Mapping::JordanWigner => 1 << j,
            Mapping::Parity => low_mask(j + 1),
            Mapping::BravyiKitaev => {
                let size = (j + 1) & (!j); // lowbit(j + 1)
                low_mask(j + 1) & !low_mask(j + 1 - size)
            }
        }
    }

    /// The qubit vector `A n` of an occupation vector `n` (bit masks).
    pub fn encode(self, occupation: u64, n_modes: usize) -> Result<u64> {
        check_modes(n_modes)?;
        if n_modes < 64 && occupation >> n_modes != 0 {
            return Err(QsoError::invalid(
                "occupation",
                format!("occupies modes beyond {n_modes}"),
            ));
        }
        Ok((0..n_modes).fold(0, |b, j| {
            b | (u64::from((self.row(j) & occupation).count_ones() % 2 == 1) << j)
        }))
    }

    /// The Pauli form of `a_j†` on `n_modes` modes.
    pub fn creation(self, j: usize, n_modes: usize) -> Result<QubitOperator> {
        check_modes(n_modes)?;
        if j >= n_modes {
            return Err(QsoError::invalid(
                "mode",
                format!("mode {j} out of range for {n_modes} modes"),
            ));
        }
        let rows: Vec<u64> = (0..n_modes).map(|i| self.row(i)).collect();
        let inverse = invert_lower_triangular(&rows);
        // Column j of A: qubits i with A_ij = 1.
        let flip = (0..n_modes).fold(0u64, |m, i| m | (((rows[i] >> j) & 1) << i));
        // Σ_{k<j} n_k = Σ_k [k<j] (A⁻¹ b)_k, a parity of qubits.
        let parity = (0..j).fold(0u64, |m, k| m ^ inverse[k]);
        let occupied = inverse[j];
        let x = string(flip, 0)?;
        let zp = string(0, parity)?;
        let zn = string(0, occupied)?;
        let half = C64::new(0.5, 0.0);
        let xz = &QubitOperator::from(x) * &QubitOperator::from(zp);
        let projector = QubitOperator::from_terms([(half, PauliString::identity()), (half, zn)]);
        Ok(&xz * &projector)
    }

    /// Map a fermionic operator on `n_modes` modes to qubits.
    pub fn map(self, op: &FermionOperator, n_modes: usize) -> Result<QubitOperator> {
        check_modes(n_modes)?;
        if op.num_modes() > n_modes {
            return Err(QsoError::invalid(
                "fermion operator",
                format!("acts on {} modes, more than {n_modes}", op.num_modes()),
            ));
        }
        if op
            .terms()
            .any(|(_, ops)| ops.len() > super::fermion::MAX_PRODUCT_LENGTH)
        {
            return Err(QsoError::invalid(
                "fermion operator",
                format!(
                    "products of more than {} ladder operators are not supported",
                    super::fermion::MAX_PRODUCT_LENGTH
                ),
            ));
        }
        let ladders = self.ladder_operators(n_modes)?;
        let mut out = QubitOperator::new();
        for (c, ops) in op.terms() {
            let mut product = QubitOperator::identity(c);
            for l in ops {
                let (create, annihilate) = &ladders[l.mode];
                product = &product * if l.dagger { create } else { annihilate };
            }
            for (c, s) in product.terms() {
                out.add_term(c, s);
            }
        }
        out.simplify(1e-14);
        Ok(out)
    }

    /// `(a_j†, a_j)` for every mode.
    pub(crate) fn ladder_operators(
        self,
        n_modes: usize,
    ) -> Result<Vec<(QubitOperator, QubitOperator)>> {
        (0..n_modes)
            .map(|j| {
                let c = self.creation(j, n_modes)?;
                let a = c.adjoint();
                Ok((c, a))
            })
            .collect()
    }
}

fn low_mask(k: usize) -> u64 {
    if k >= 64 { u64::MAX } else { (1u64 << k) - 1 }
}

fn check_modes(n_modes: usize) -> Result<()> {
    if n_modes == 0 || n_modes > MAX_MODES {
        return Err(QsoError::invalid(
            "n_modes",
            format!("need 1 to {MAX_MODES} modes, got {n_modes}"),
        ));
    }
    Ok(())
}

/// The Pauli string with X on `x` and Z on `z` (no overlap).
fn string(x: u64, z: u64) -> Result<PauliString> {
    let factors: Vec<(usize, Pauli)> = (0..64)
        .filter_map(|q| match ((x >> q) & 1, (z >> q) & 1) {
            (1, 0) => Some((q, Pauli::X)),
            (0, 1) => Some((q, Pauli::Z)),
            (1, 1) => Some((q, Pauli::Y)),
            _ => None,
        })
        .collect();
    PauliString::new(&factors)
}

/// Rows of `A⁻¹` over GF(2) for a lower unitriangular `A` given by rows.
fn invert_lower_triangular(rows: &[u64]) -> Vec<u64> {
    // Forward substitution: (A⁻¹)_j = e_j ⊕ Σ_{k<j, A_jk=1} (A⁻¹)_k.
    let mut inv: Vec<u64> = Vec::with_capacity(rows.len());
    for (j, &row) in rows.iter().enumerate() {
        let mut r = 1u64 << j;
        for (k, inv_k) in inv.iter().enumerate() {
            if (row >> k) & 1 == 1 {
                r ^= inv_k;
            }
        }
        inv.push(r);
    }
    inv
}

fn check_reduction(n_spin_orbitals: usize, n_alpha: usize, n_beta: usize) -> Result<usize> {
    if n_spin_orbitals < 4 || n_spin_orbitals % 2 == 1 || n_spin_orbitals > MAX_MODES {
        return Err(QsoError::invalid(
            "n_spin_orbitals",
            "the two-qubit reduction needs an even number of spin orbitals, at least 4",
        ));
    }
    let m = n_spin_orbitals / 2;
    if n_alpha > m || n_beta > m {
        return Err(QsoError::invalid(
            "electrons",
            "more electrons than orbitals",
        ));
    }
    Ok(m)
}

/// Taper a parity-mapped operator on `2m` spin orbitals in block order
/// that conserves the numbers `n_alpha` and `n_beta` of spin-up and
/// spin-down electrons: qubits `m − 1` (α parity) and `2m − 1` (total
/// parity) may carry only `I` or `Z`; they are replaced by their
/// eigenvalues `(−1)^{n_α}` and `(−1)^{n_α + n_β}` and removed. The result
/// acts on `2m − 2` qubits.
pub fn taper(
    op: &QubitOperator,
    n_spin_orbitals: usize,
    n_alpha: usize,
    n_beta: usize,
) -> Result<QubitOperator> {
    let m = check_reduction(n_spin_orbitals, n_alpha, n_beta)?;
    if op.min_qubits() > n_spin_orbitals {
        return Err(QsoError::DimensionMismatch {
            expected: n_spin_orbitals,
            actual: op.min_qubits(),
        });
    }
    let (qa, qt) = (m - 1, 2 * m - 1);
    let sign_a = if n_alpha.is_multiple_of(2) { 1.0 } else { -1.0 };
    let sign_t = if (n_alpha + n_beta).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    let mut out = QubitOperator::new();
    for (c, s) in op.terms() {
        let mut coefficient = c;
        let mut factors = Vec::new();
        for q in 0..n_spin_orbitals {
            let p = s.get(q);
            if q == qa || q == qt {
                match p {
                    Pauli::I => {}
                    Pauli::Z => coefficient *= if q == qa { sign_a } else { sign_t },
                    _ => {
                        return Err(QsoError::invalid(
                            "operator",
                            format!(
                                "term {s} acts with {p:?} on tapered qubit {q}; \
                                 it does not conserve the electron numbers"
                            ),
                        ));
                    }
                }
            } else if p != Pauli::I {
                factors.push((if q > qa { q - 1 } else { q }, p));
            }
        }
        out.add_term(coefficient, PauliString::new(&factors)?);
    }
    out.simplify(1e-14);
    Ok(out)
}

/// [`taper`] for a Hermitian operator such as a molecular Hamiltonian.
pub fn two_qubit_reduction(
    h: &PauliSum,
    n_spin_orbitals: usize,
    n_alpha: usize,
    n_beta: usize,
) -> Result<PauliSum> {
    taper(&h.into(), n_spin_orbitals, n_alpha, n_beta)?.to_pauli_sum(1e-14)
}

/// How spin orbitals become qubits: a [`Mapping`], optionally followed by
/// the [`two_qubit_reduction`] (parity mapping only).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct QubitEncoding {
    pub mapping: Mapping,
    pub two_qubit_reduction: bool,
}

impl QubitEncoding {
    /// `mapping` without reduction.
    pub fn new(mapping: Mapping) -> Self {
        Self {
            mapping,
            two_qubit_reduction: false,
        }
    }

    /// The parity mapping with the two-qubit reduction.
    pub fn parity_reduced() -> Self {
        Self {
            mapping: Mapping::Parity,
            two_qubit_reduction: true,
        }
    }

    fn check(self, n_spin_orbitals: usize, n_alpha: usize, n_beta: usize) -> Result<()> {
        check_modes(n_spin_orbitals)?;
        if self.two_qubit_reduction {
            if self.mapping != Mapping::Parity {
                return Err(QsoError::invalid(
                    "encoding",
                    "the two-qubit reduction requires the parity mapping",
                ));
            }
            check_reduction(n_spin_orbitals, n_alpha, n_beta)?;
        }
        Ok(())
    }

    /// Qubits needed for `n_spin_orbitals` spin orbitals.
    pub fn n_qubits(self, n_spin_orbitals: usize) -> usize {
        if self.two_qubit_reduction {
            n_spin_orbitals.saturating_sub(2)
        } else {
            n_spin_orbitals
        }
    }

    /// Map a fermionic operator that conserves `n_alpha` and `n_beta`
    /// (needed only with the reduction).
    pub fn map(
        self,
        op: &FermionOperator,
        n_spin_orbitals: usize,
        n_alpha: usize,
        n_beta: usize,
    ) -> Result<QubitOperator> {
        self.check(n_spin_orbitals, n_alpha, n_beta)?;
        let mapped = self.mapping.map(op, n_spin_orbitals)?;
        if self.two_qubit_reduction {
            taper(&mapped, n_spin_orbitals, n_alpha, n_beta)
        } else {
            Ok(mapped)
        }
    }

    /// Map a Hermitian fermionic operator to a [`PauliSum`].
    pub fn map_hermitian(
        self,
        op: &FermionOperator,
        n_spin_orbitals: usize,
        n_alpha: usize,
        n_beta: usize,
    ) -> Result<PauliSum> {
        self.map(op, n_spin_orbitals, n_alpha, n_beta)?
            .to_pauli_sum(1e-12)
    }

    /// The qubit basis state (bit mask) of the occupation `occupation`.
    /// With the reduction, the occupation must hold `n_α` electrons in the
    /// first half and `n_β` in the second, matching the tapered parities.
    pub fn encode(self, occupation: u64, n_spin_orbitals: usize) -> Result<u64> {
        let b = self.mapping.encode(occupation, n_spin_orbitals)?;
        if !self.two_qubit_reduction {
            return Ok(b);
        }
        let m = n_spin_orbitals / 2;
        let alpha = (occupation & low_mask(m)).count_ones() as usize;
        let beta = (occupation >> m).count_ones() as usize;
        self.check(n_spin_orbitals, alpha, beta)?;
        let low = b & low_mask(m - 1);
        let mid = (b >> m) & low_mask(m - 1);
        Ok(low | (mid << (m - 1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::StateVector;

    /// `⟨b'|op|b⟩` applied to a basis state, as (amplitude, index).
    fn apply(op: &QubitOperator, n: usize, b: usize) -> Vec<C64> {
        let mut out = vec![C64::new(0.0, 0.0); 1 << n];
        let psi = StateVector::basis(n, b).unwrap();
        for (c, s) in op.terms() {
            let single = PauliSum::from_terms([(1.0, s)]);
            for (o, a) in out.iter_mut().zip(single.apply(&psi).unwrap()) {
                *o += c * a;
            }
        }
        out
    }

    #[test]
    fn creation_operators_act_on_encoded_occupations() {
        let n = 5;
        for mapping in [
            Mapping::JordanWigner,
            Mapping::Parity,
            Mapping::BravyiKitaev,
        ] {
            for j in 0..n {
                let c = mapping.creation(j, n).unwrap();
                for occ in 0..(1u64 << n) {
                    let out = apply(&c, n, mapping.encode(occ, n).unwrap() as usize);
                    let mut expected = vec![C64::new(0.0, 0.0); 1 << n];
                    if (occ >> j) & 1 == 0 {
                        let sign = if (occ & ((1 << j) - 1)).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        expected[mapping.encode(occ | (1 << j), n).unwrap() as usize] =
                            C64::new(sign, 0.0);
                    }
                    for (a, b) in out.iter().zip(&expected) {
                        assert!((a - b).norm() < 1e-12, "{mapping:?} j={j} occ={occ:b}");
                    }
                }
            }
        }
    }

    #[test]
    fn bravyi_kitaev_matches_the_textbook_matrix() {
        // Seeley et al.: for 4 modes b0 = n0, b1 = n0+n1, b2 = n2, b3 = n0+…+n3.
        let rows: Vec<u64> = (0..4).map(|j| Mapping::BravyiKitaev.row(j)).collect();
        assert_eq!(rows, vec![0b0001, 0b0011, 0b0100, 0b1111]);
        // Parity: b_j = n_0 + … + n_j.
        assert_eq!(Mapping::Parity.row(2), 0b111);
        // Weight of BK creation operators grows as log n.
        let c = Mapping::BravyiKitaev.creation(7, 16).unwrap();
        assert!(c.terms().all(|(_, s)| s.weight() <= 5));
    }

    #[test]
    fn names() {
        for m in [
            Mapping::JordanWigner,
            Mapping::Parity,
            Mapping::BravyiKitaev,
        ] {
            assert_eq!(Mapping::from_name(m.name()).unwrap(), m);
        }
        assert_eq!(Mapping::from_name("BK").unwrap(), Mapping::BravyiKitaev);
        assert!(Mapping::from_name("x").is_err());
        assert!(Mapping::JordanWigner.creation(3, 3).is_err());
        assert!(Mapping::JordanWigner.creation(0, 65).is_err());
        assert!(Mapping::Parity.encode(0b1000, 3).is_err());
        assert!(Mapping::Parity.encode(0, 65).is_err());
        assert!(QubitEncoding::parity_reduced().encode(1, 1).is_err());
        assert!(QubitEncoding::parity_reduced().encode(0b0101, 4).is_ok());
    }
}
