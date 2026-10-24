//! Pauli strings and Hermitian Pauli sums (observables / Hamiltonians).
//!
//! A [`PauliString`] on up to 64 qubits is stored as two bit masks: bit `q`
//! of `x` (`z`) is set if the factor on qubit `q` contains X (Z); Y sets
//! both. Since `Y = i·X·Z`, a string acts on basis states as
//!
//! ```text
//! P|b⟩ = i^{#Y} · (−1)^{popcount(b ∧ z)} · |b ⊕ x⟩
//! ```
//!
//! which makes expectation values and `H|ψ⟩` simple bit manipulations over
//! the 2ⁿ amplitudes — no matrices are ever formed.
//!
//! ```
//! use metatron_qso::quantum::{PauliSum, StateVector};
//!
//! let h: PauliSum = "1.0*Z0 Z1 - 0.5 X0 + 0.25".parse().unwrap();
//! let psi = StateVector::zero(2);
//! // ⟨00|Z0Z1|00⟩ = 1, ⟨00|X0|00⟩ = 0
//! assert!((h.expectation(&psi).unwrap() - 1.25).abs() < 1e-12);
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::str::FromStr;

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::state::StateVector;
use crate::error::{QsoError, Result};
use crate::linalg::{CMatrix, KrylovOptions, LanczosResult, lanczos_ground_state};
use crate::parallel::{for_each_mut, sum_c64, sum_f64};

/// Single-qubit Pauli operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

/// Tensor product of single-qubit Paulis (without coefficient).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct PauliString {
    x: u64,
    z: u64,
}

/// `i^k` for `k mod 4`.
#[inline]
fn i_pow(k: u32) -> C64 {
    match k % 4 {
        0 => C64::new(1.0, 0.0),
        1 => C64::new(0.0, 1.0),
        2 => C64::new(-1.0, 0.0),
        _ => C64::new(0.0, -1.0),
    }
}

impl PauliString {
    /// The identity string.
    pub fn identity() -> Self {
        Self::default()
    }

    /// Build from `(qubit, pauli)` factors; a qubit may appear only once.
    pub fn new(factors: &[(usize, Pauli)]) -> Result<Self> {
        let mut s = Self::identity();
        let mut used = 0u64;
        for &(q, p) in factors {
            if q >= 64 {
                return Err(QsoError::QubitOutOfRange {
                    qubit: q,
                    n_qubits: 64,
                });
            }
            if used & (1 << q) != 0 {
                return Err(QsoError::Parse(format!(
                    "qubit {q} appears twice in Pauli string"
                )));
            }
            used |= 1 << q;
            s.set(q, p);
        }
        Ok(s)
    }

    /// Single-qubit Pauli `p` on qubit `q`.
    ///
    /// # Panics
    /// If `q ≥ 64` (use [`PauliString::new`] for untrusted input).
    pub fn single(q: usize, p: Pauli) -> Self {
        let mut s = Self::identity();
        s.set(q, p);
        s
    }

    fn set(&mut self, q: usize, p: Pauli) {
        assert!(q < 64, "Pauli strings support qubits 0..64, got {q}");
        let bit = 1u64 << q;
        self.x &= !bit;
        self.z &= !bit;
        match p {
            Pauli::I => {}
            Pauli::X => self.x |= bit,
            Pauli::Z => self.z |= bit,
            Pauli::Y => {
                self.x |= bit;
                self.z |= bit;
            }
        }
    }

    /// Factor on qubit `q`.
    pub fn get(&self, q: usize) -> Pauli {
        if q >= 64 {
            return Pauli::I;
        }
        match ((self.x >> q) & 1, (self.z >> q) & 1) {
            (0, 0) => Pauli::I,
            (1, 0) => Pauli::X,
            (0, 1) => Pauli::Z,
            _ => Pauli::Y,
        }
    }

    /// X-mask (bit flips).
    pub fn x_mask(&self) -> u64 {
        self.x
    }

    /// Z-mask (phases).
    pub fn z_mask(&self) -> u64 {
        self.z
    }

    /// Number of non-identity factors.
    pub fn weight(&self) -> u32 {
        (self.x | self.z).count_ones()
    }

    pub fn is_identity(&self) -> bool {
        self.x == 0 && self.z == 0
    }

    /// Diagonal in the computational basis (only I and Z factors).
    pub fn is_diagonal(&self) -> bool {
        self.x == 0
    }

    /// Smallest register size that contains every factor.
    pub fn min_qubits(&self) -> usize {
        64 - (self.x | self.z).leading_zeros() as usize
    }

    fn y_count(&self) -> u32 {
        (self.x & self.z).count_ones()
    }

    /// Whether the two strings commute.
    pub fn commutes_with(&self, other: &PauliString) -> bool {
        ((self.x & other.z).count_ones() + (self.z & other.x).count_ones()).is_multiple_of(2)
    }

    /// Product `self · other = phase · result`.
    pub fn multiply(&self, other: &PauliString) -> (C64, PauliString) {
        // Write P = i^{y} X^x Z^z. Moving Z^{z1} past X^{x2} gives (−1)^{|z1∧x2|}.
        let x = self.x ^ other.x;
        let z = self.z ^ other.z;
        let result = PauliString { x, z };
        let k = self.y_count() + other.y_count() + 2 * (self.z & other.x).count_ones();
        // Re-express X^x Z^z with Y factors: X^x Z^z = i^{−#Y(result)} P_result.
        let total = (k + 4 * 64 - result.y_count()) % 4;
        (i_pow(total), result)
    }

    /// Phase `i^{#Y}·(−1)^{popcount(b∧z)}` and image `b ⊕ x` of basis state `b`.
    #[inline]
    pub(crate) fn act(&self, b: usize) -> (C64, usize) {
        let sign = if ((b as u64) & self.z).count_ones().is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        (i_pow(self.y_count()) * sign, b ^ self.x as usize)
    }

    /// Dense `2ⁿ × 2ⁿ` matrix (small registers only).
    pub fn to_dense(&self, n_qubits: usize) -> CMatrix {
        let dim = 1usize << n_qubits;
        let mut m = CMatrix::zeros(dim, dim);
        for b in 0..dim {
            let (phase, image) = self.act(b);
            m[(image, b)] = phase;
        }
        m
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() {
            return write!(f, "I");
        }
        let mut first = true;
        for q in 0..64 {
            let p = self.get(q);
            if p != Pauli::I {
                if !first {
                    write!(f, " ")?;
                }
                write!(f, "{p:?}{q}")?;
                first = false;
            }
        }
        Ok(())
    }
}

/// Real linear combination of Pauli strings — a Hermitian operator.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PauliSum {
    terms: Vec<(f64, PauliString)>,
}

impl PauliSum {
    /// The zero operator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from `(coefficient, string)` pairs (merged and simplified).
    pub fn from_terms(terms: impl IntoIterator<Item = (f64, PauliString)>) -> Self {
        let mut s = Self {
            terms: terms.into_iter().collect(),
        };
        s.simplify(0.0);
        s
    }

    /// Add `coefficient · string`.
    pub fn add_term(&mut self, coefficient: f64, string: PauliString) -> &mut Self {
        self.terms.push((coefficient, string));
        self
    }

    /// Merge equal strings and drop terms with `|c| ≤ tol`. Terms are
    /// sorted canonically.
    pub fn simplify(&mut self, tol: f64) -> &mut Self {
        let mut merged: BTreeMap<PauliString, f64> = BTreeMap::new();
        for &(c, s) in &self.terms {
            *merged.entry(s).or_insert(0.0) += c;
        }
        self.terms = merged
            .into_iter()
            .filter(|(_, c)| c.abs() > tol)
            .map(|(s, c)| (c, s))
            .collect();
        self
    }

    pub fn terms(&self) -> &[(f64, PauliString)] {
        &self.terms
    }

    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Smallest register size that contains every term.
    pub fn min_qubits(&self) -> usize {
        self.terms
            .iter()
            .map(|(_, s)| s.min_qubits())
            .max()
            .unwrap_or(0)
    }

    /// Coefficient of the identity term.
    pub fn constant(&self) -> f64 {
        self.terms
            .iter()
            .filter(|(_, s)| s.is_identity())
            .map(|(c, _)| c)
            .sum()
    }

    /// All terms diagonal in the computational basis.
    pub fn is_diagonal(&self) -> bool {
        self.terms.iter().all(|(_, s)| s.is_diagonal())
    }

    /// Sum of absolute coefficients (an upper bound on the spectral norm).
    pub fn one_norm(&self) -> f64 {
        self.terms.iter().map(|(c, _)| c.abs()).sum()
    }

    fn check_register(&self, n_qubits: usize) -> Result<()> {
        let needed = self.min_qubits();
        if needed > n_qubits {
            return Err(QsoError::QubitOutOfRange {
                qubit: needed - 1,
                n_qubits,
            });
        }
        Ok(())
    }

    /// `⟨ψ|H|ψ⟩` (real because H is Hermitian).
    pub fn expectation(&self, state: &StateVector) -> Result<f64> {
        self.check_register(state.n_qubits())?;
        Ok(self.expectation_amplitudes(state.amplitudes()))
    }

    pub(crate) fn expectation_amplitudes(&self, amps: &[C64]) -> f64 {
        let n = amps.len();
        let mut total = 0.0;
        for &(c, s) in &self.terms {
            let value = if s.is_identity() {
                sum_f64(n, |b| amps[b].norm_sqr())
            } else if s.is_diagonal() {
                let z = s.z;
                sum_f64(n, |b| {
                    let p = amps[b].norm_sqr();
                    if ((b as u64) & z).count_ones().is_multiple_of(2) {
                        p
                    } else {
                        -p
                    }
                })
            } else {
                sum_c64(n, |b| {
                    let (phase, image) = s.act(b);
                    amps[image].conj() * phase * amps[b]
                })
                .re
            };
            total += c * value;
        }
        total
    }

    /// `out = H·amps` without forming a matrix.
    pub(crate) fn apply_amplitudes(&self, amps: &[C64], out: &mut [C64]) {
        debug_assert_eq!(amps.len(), out.len());
        let terms = &self.terms;
        for_each_mut(out, |c, o| {
            let mut acc = C64::new(0.0, 0.0);
            for &(coef, s) in terms {
                // (Pψ)[c] = i^{#Y} (−1)^{|(c⊕x)∧z|} ψ[c⊕x]
                let (phase, source) = s.act(c ^ s.x as usize);
                debug_assert_eq!(source, c);
                acc += coef * phase * amps[c ^ s.x as usize];
            }
            *o = acc;
        });
    }

    /// `H|ψ⟩` as a new (unnormalised) amplitude vector.
    pub fn apply(&self, state: &StateVector) -> Result<Vec<C64>> {
        self.check_register(state.n_qubits())?;
        let mut out = vec![C64::new(0.0, 0.0); state.dim()];
        self.apply_amplitudes(state.amplitudes(), &mut out);
        Ok(out)
    }

    /// Diagonal of a diagonal operator (`None` if any term flips bits).
    pub fn diagonal(&self, n_qubits: usize) -> Result<Option<Vec<f64>>> {
        self.check_register(n_qubits)?;
        if n_qubits > super::state::MAX_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "diagonal of a Pauli sum",
                max: super::state::MAX_QUBITS,
                requested: n_qubits,
            });
        }
        if !self.is_diagonal() {
            return Ok(None);
        }
        let terms = &self.terms;
        Ok(Some(crate::parallel::map_collect(1 << n_qubits, |b| {
            terms
                .iter()
                .map(|&(c, s)| {
                    if ((b as u64) & s.z).count_ones().is_multiple_of(2) {
                        c
                    } else {
                        -c
                    }
                })
                .sum()
        })))
    }

    /// Dense matrix (small registers only).
    pub fn to_dense(&self, n_qubits: usize) -> Result<CMatrix> {
        self.check_register(n_qubits)?;
        if n_qubits > 14 {
            return Err(QsoError::TooManyQubits {
                what: "dense Pauli-sum matrix",
                max: 14,
                requested: n_qubits,
            });
        }
        let dim = 1usize << n_qubits;
        let mut m = CMatrix::zeros(dim, dim);
        for &(c, s) in &self.terms {
            for b in 0..dim {
                let (phase, image) = s.act(b);
                m[(image, b)] += phase * c;
            }
        }
        Ok(m)
    }

    /// Exact ground state on `n_qubits` qubits by matrix-free Lanczos.
    ///
    /// Works up to ~25 qubits (memory: one state vector per Krylov vector).
    pub fn ground_state(
        &self,
        n_qubits: usize,
        opts: &KrylovOptions,
    ) -> Result<(f64, StateVector)> {
        let res = self.lanczos(n_qubits, opts)?;
        let state = StateVector::from_amplitudes(res.eigenvector)?;
        Ok((res.eigenvalue, state))
    }

    /// Exact ground-state energy (see [`PauliSum::ground_state`]).
    pub fn ground_state_energy(&self, n_qubits: usize) -> Result<f64> {
        if let Some(diag) = self.diagonal(n_qubits)? {
            return Ok(diag.into_iter().fold(f64::INFINITY, f64::min));
        }
        Ok(self
            .lanczos(n_qubits, &KrylovOptions::default())?
            .eigenvalue)
    }

    fn lanczos(&self, n_qubits: usize, opts: &KrylovOptions) -> Result<LanczosResult> {
        self.check_register(n_qubits)?;
        if n_qubits > super::state::MAX_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "Lanczos diagonalisation",
                max: super::state::MAX_QUBITS,
                requested: n_qubits,
            });
        }
        let dim = 1usize << n_qubits;
        lanczos_ground_state(dim, |x, y| self.apply_amplitudes(x, y), None, opts)
    }
}

impl Add for PauliSum {
    type Output = PauliSum;
    fn add(mut self, rhs: PauliSum) -> PauliSum {
        self.terms.extend(rhs.terms);
        self.simplify(0.0);
        self
    }
}

impl Sub for PauliSum {
    type Output = PauliSum;
    fn sub(self, rhs: PauliSum) -> PauliSum {
        self + (-rhs)
    }
}

impl Neg for PauliSum {
    type Output = PauliSum;
    fn neg(self) -> PauliSum {
        self * -1.0
    }
}

impl Mul<f64> for PauliSum {
    type Output = PauliSum;
    fn mul(mut self, rhs: f64) -> PauliSum {
        self.terms.iter_mut().for_each(|(c, _)| *c *= rhs);
        self.simplify(0.0);
        self
    }
}

impl Mul for &PauliSum {
    type Output = PauliSum;
    /// Operator product. The result is Hermitian only if the factors
    /// commute; imaginary parts of the product coefficients are discarded,
    /// so use this for commuting factors (e.g. building Ising terms).
    fn mul(self, rhs: &PauliSum) -> PauliSum {
        let mut out = Vec::with_capacity(self.terms.len() * rhs.terms.len());
        for &(a, sa) in &self.terms {
            for &(b, sb) in &rhs.terms {
                let (phase, s) = sa.multiply(&sb);
                out.push((a * b * phase.re, s));
            }
        }
        PauliSum::from_terms(out)
    }
}

impl From<PauliString> for PauliSum {
    fn from(s: PauliString) -> Self {
        PauliSum::from_terms([(1.0, s)])
    }
}

impl fmt::Display for PauliSum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        for (k, (c, s)) in self.terms.iter().enumerate() {
            if k == 0 {
                write!(f, "{c}")?;
            } else if *c < 0.0 {
                write!(f, " - {}", -c)?;
            } else {
                write!(f, " + {c}")?;
            }
            if !s.is_identity() {
                write!(f, " {s}")?;
            }
        }
        Ok(())
    }
}

impl FromStr for PauliSum {
    type Err = QsoError;

    /// Parse expressions such as `"0.5*Z0 Z1 - 1.2 X0 + 0.3*Y2*Y3 - 1.05"`.
    ///
    /// Terms are separated by `+`/`−`; within a term an optional real
    /// coefficient is followed by Pauli factors `X<q>`, `Y<q>`, `Z<q>` (or a
    /// bare `I`), separated by whitespace or `*`.
    fn from_str(s: &str) -> Result<Self> {
        let chars: Vec<char> = s.chars().collect();
        let mut terms = Vec::new();
        let mut pos = 0usize;
        // State of the term being read.
        let mut sign = 1.0;
        let mut coef: Option<f64> = None;
        let mut factors: Vec<(usize, Pauli)> = Vec::new();
        let mut has_content = false;
        // An operator was read and no term has followed it yet.
        let mut dangling = false;

        let finish = |sign: f64,
                      coef: Option<f64>,
                      factors: &mut Vec<(usize, Pauli)>,
                      terms: &mut Vec<(f64, PauliString)>|
         -> Result<()> {
            let string = PauliString::new(factors)?;
            terms.push((sign * coef.unwrap_or(1.0), string));
            factors.clear();
            Ok(())
        };

        // The previous token was `*`, so a following sign is unary.
        let mut after_star = false;
        while pos < chars.len() {
            let c = chars[pos];
            if c.is_whitespace() {
                pos += 1;
                continue;
            }
            if c == '*' {
                if !has_content || after_star {
                    return Err(QsoError::Parse(format!("unexpected '*' at {pos}")));
                }
                after_star = true;
                dangling = true;
                pos += 1;
                continue;
            }
            if (c == '+' || c == '-') && after_star {
                // `2*-X0`: the sign belongs to the factor, not a new term.
                if c == '-' {
                    sign = -sign;
                }
                pos += 1;
                continue;
            }
            if c == '+' || c == '-' {
                if has_content {
                    // Binary operator: close the current term.
                    finish(sign, coef, &mut factors, &mut terms)?;
                    coef = None;
                    has_content = false;
                    sign = 1.0;
                }
                // Binary or unary, the operator applies to the next term.
                if c == '-' {
                    sign = -sign;
                }
                dangling = true;
                pos += 1;
                continue;
            }
            dangling = false;
            after_star = false;
            if c.is_ascii_digit() || c == '.' {
                if coef.is_some() || !factors.is_empty() {
                    return Err(QsoError::Parse(format!("unexpected number at {pos}")));
                }
                let start = pos;
                pos += 1;
                while pos < chars.len() {
                    let d = chars[pos];
                    let prev = chars[pos - 1];
                    let mantissa = d.is_ascii_digit() || d == '.' || d == 'e' || d == 'E';
                    let exponent_sign = (d == '+' || d == '-') && (prev == 'e' || prev == 'E');
                    if mantissa || exponent_sign {
                        pos += 1;
                    } else {
                        break;
                    }
                }
                let text: String = chars[start..pos].iter().collect();
                let value: f64 = text
                    .parse()
                    .map_err(|_| QsoError::Parse(format!("invalid number '{text}'")))?;
                coef = Some(value);
                has_content = true;
                continue;
            }
            let pauli = match c.to_ascii_uppercase() {
                'I' => Pauli::I,
                'X' => Pauli::X,
                'Y' => Pauli::Y,
                'Z' => Pauli::Z,
                _ => {
                    return Err(QsoError::Parse(format!(
                        "unexpected character '{c}' at {pos}"
                    )));
                }
            };
            pos += 1;
            let start = pos;
            while pos < chars.len() && chars[pos].is_ascii_digit() {
                pos += 1;
            }
            if start == pos {
                if pauli != Pauli::I {
                    return Err(QsoError::Parse(format!(
                        "Pauli {pauli:?} without qubit index"
                    )));
                }
            } else {
                let text: String = chars[start..pos].iter().collect();
                let q: usize = text
                    .parse()
                    .map_err(|_| QsoError::Parse(format!("invalid qubit index '{text}'")))?;
                if pauli != Pauli::I {
                    factors.push((q, pauli));
                }
            }
            has_content = true;
        }
        if dangling {
            return Err(QsoError::Parse("expression ends with an operator".into()));
        }
        if has_content {
            finish(sign, coef, &mut factors, &mut terms)?;
        }
        Ok(PauliSum::from_terms(terms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_pauli(p: Pauli) -> CMatrix {
        let z = C64::new(0.0, 0.0);
        let o = C64::new(1.0, 0.0);
        let i = C64::new(0.0, 1.0);
        match p {
            Pauli::I => CMatrix::from_row_slice(2, 2, &[o, z, z, o]),
            Pauli::X => CMatrix::from_row_slice(2, 2, &[z, o, o, z]),
            Pauli::Y => CMatrix::from_row_slice(2, 2, &[z, -i, i, z]),
            Pauli::Z => CMatrix::from_row_slice(2, 2, &[o, z, z, -o]),
        }
    }

    /// Kronecker reference with qubit n−1 as the most significant factor.
    fn kron_reference(s: &PauliString, n: usize) -> CMatrix {
        let mut m = dense_pauli(s.get(n - 1));
        for q in (0..n - 1).rev() {
            m = m.kronecker(&dense_pauli(s.get(q)));
        }
        m
    }

    #[test]
    fn string_action_matches_kronecker_products() {
        let ps = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];
        for a in ps {
            for b in ps {
                for c in ps {
                    let s = PauliString::new(&[(0, a), (1, b), (2, c)]).unwrap();
                    assert!(
                        (s.to_dense(3) - kron_reference(&s, 3)).norm() < 1e-14,
                        "{s}"
                    );
                }
            }
        }
    }

    #[test]
    fn multiplication_phases_are_correct() {
        let ps = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];
        for a in ps {
            for b in ps {
                for c in ps {
                    for d in ps {
                        let s1 = PauliString::new(&[(0, a), (1, b)]).unwrap();
                        let s2 = PauliString::new(&[(0, c), (1, d)]).unwrap();
                        let (phase, prod) = s1.multiply(&s2);
                        let expected = s1.to_dense(2) * s2.to_dense(2);
                        let got = prod.to_dense(2) * phase;
                        assert!((expected - got).norm() < 1e-14, "{s1} * {s2}");
                        let commute = (s1.to_dense(2) * s2.to_dense(2)
                            - s2.to_dense(2) * s1.to_dense(2))
                        .norm()
                            < 1e-14;
                        assert_eq!(commute, s1.commutes_with(&s2));
                    }
                }
            }
        }
    }

    #[test]
    fn parser_handles_signs_coefficients_and_constants() {
        let h: PauliSum = "0.5*Z0 Z1 - 1.2 X0 + 0.3*Y2*Y3 - 1.05".parse().unwrap();
        assert_eq!(h.num_terms(), 4);
        assert!((h.constant() + 1.05).abs() < 1e-15);
        let zz = PauliString::new(&[(0, Pauli::Z), (1, Pauli::Z)]).unwrap();
        let x0 = PauliString::single(0, Pauli::X);
        let coef = |s: PauliString| h.terms().iter().find(|(_, t)| *t == s).unwrap().0;
        assert_eq!(coef(zz), 0.5);
        assert_eq!(coef(x0), -1.2);
        let h2: PauliSum = "-Z0 + 2e-1 X1 - -0.5 I".parse().unwrap();
        assert_eq!(h2.num_terms(), 3);
        assert_eq!(h2.constant(), 0.5);
        assert_eq!(
            h2.terms()
                .iter()
                .find(|(_, s)| *s == PauliString::single(0, Pauli::Z))
                .unwrap()
                .0,
            -1.0
        );
        assert_eq!(
            h2.terms()
                .iter()
                .find(|(_, s)| *s == PauliString::single(1, Pauli::X))
                .unwrap()
                .0,
            0.2
        );
        assert_eq!("".parse::<PauliSum>().unwrap().num_terms(), 0);
        assert!("Q0".parse::<PauliSum>().is_err());
        assert!("X".parse::<PauliSum>().is_err());
        assert!("X0 X0".parse::<PauliSum>().is_err());
        assert!("Z0 +".parse::<PauliSum>().is_err());
    }

    #[test]
    fn expectation_and_apply_match_dense() {
        let h: PauliSum = "0.7 X0 Y1 - 0.2 Z2 + 1.3 Y0 Z1 X2 + 0.4 X1 X2 - 0.9"
            .parse()
            .unwrap();
        let mut rng = crate::rng::Rng::seed_from_u64(1);
        let psi = StateVector::random(3, &mut rng);
        let dense = h.to_dense(3).unwrap();
        let v = nalgebra::DVector::from_column_slice(psi.amplitudes());
        let hv = &dense * &v;
        let exact = v.dotc(&hv).re;
        assert!((h.expectation(&psi).unwrap() - exact).abs() < 1e-12);
        let applied = h.apply(&psi).unwrap();
        for (a, b) in applied.iter().zip(hv.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
        // Hermitian
        assert!((&dense - dense.adjoint()).norm() < 1e-12);
    }

    #[test]
    fn lanczos_ground_state_matches_dense() {
        let h: PauliSum = "1.0 Z0 Z1 + 1.0 Z1 Z2 + 1.0 Z2 Z3 + 0.7 X0 + 0.7 X1 + 0.7 X2 + 0.7 X3"
            .parse()
            .unwrap();
        let (vals, _) = crate::linalg::eigh(&h.to_dense(4).unwrap());
        let e0 = h.ground_state_energy(4).unwrap();
        assert!((e0 - vals[0]).abs() < 1e-8);
    }

    #[test]
    fn product_of_commuting_sums() {
        let a: PauliSum = "Z0".parse().unwrap();
        let b: PauliSum = "Z1".parse().unwrap();
        let ab = &a * &b;
        assert_eq!(
            ab.terms()[0].1,
            PauliString::new(&[(0, Pauli::Z), (1, Pauli::Z)]).unwrap()
        );
        let xx = &"X0".parse::<PauliSum>().unwrap() * &"X0".parse::<PauliSum>().unwrap();
        assert_eq!(xx.constant(), 1.0);
    }
}
