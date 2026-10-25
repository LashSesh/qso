//! Linear combinations of Pauli strings with complex coefficients.
//!
//! A [`PauliSum`] is Hermitian by construction (real coefficients). Many
//! intermediate objects are not: the image of a fermionic ladder operator,
//! the commutator `[A, B]` of two Hamiltonians (anti-Hermitian), or the
//! generator `T − T†` of a coupled-cluster ansatz. [`QubitOperator`] holds
//! them, multiplies them with the Pauli algebra and converts a Hermitian
//! result back with [`QubitOperator::to_pauli_sum`].
//!
//! ```
//! use metatron_qso::quantum::{QubitOperator, PauliSum};
//!
//! let x: QubitOperator = (&"X0".parse::<PauliSum>().unwrap()).into();
//! let y: QubitOperator = (&"Y0".parse::<PauliSum>().unwrap()).into();
//! // [X, Y] = 2iZ
//! let c = x.commutator(&y);
//! assert_eq!(c.num_terms(), 1);
//! assert!((c.one_norm() - 2.0).abs() < 1e-12);
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Mul, Sub};

use num_complex::Complex64 as C64;

use super::pauli::{PauliString, PauliSum};
use crate::error::{QsoError, Result};

/// `Σ_k c_k P_k` with complex `c_k`; equal strings are merged.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QubitOperator {
    terms: BTreeMap<PauliString, C64>,
}

impl QubitOperator {
    /// The zero operator.
    pub fn new() -> Self {
        Self::default()
    }

    /// `c · I`.
    pub fn identity(c: C64) -> Self {
        let mut op = Self::new();
        op.add_term(c, PauliString::identity());
        op
    }

    /// Build from `(coefficient, string)` pairs.
    pub fn from_terms(terms: impl IntoIterator<Item = (C64, PauliString)>) -> Self {
        let mut op = Self::new();
        for (c, s) in terms {
            op.add_term(c, s);
        }
        op
    }

    /// Add `c · P`.
    pub fn add_term(&mut self, c: C64, string: PauliString) -> &mut Self {
        *self.terms.entry(string).or_default() += c;
        self
    }

    /// The terms, ordered by string.
    pub fn terms(&self) -> impl Iterator<Item = (C64, PauliString)> + '_ {
        self.terms.iter().map(|(s, &c)| (c, *s))
    }

    /// Coefficient of `string` (zero if absent).
    pub fn coefficient(&self, string: &PauliString) -> C64 {
        self.terms.get(string).copied().unwrap_or_default()
    }

    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Drop terms with `|c| ≤ tol`.
    pub fn simplify(&mut self, tol: f64) -> &mut Self {
        self.terms.retain(|_, c| c.norm() > tol);
        self
    }

    /// `Σ|c_k|`, an upper bound on the operator norm.
    pub fn one_norm(&self) -> f64 {
        self.terms.values().map(|c| c.norm()).sum()
    }

    /// Smallest register holding every string.
    pub fn min_qubits(&self) -> usize {
        self.terms
            .keys()
            .map(PauliString::min_qubits)
            .max()
            .unwrap_or(0)
    }

    /// `s · A`.
    pub fn scaled(&self, s: C64) -> Self {
        Self {
            terms: self.terms.iter().map(|(p, &c)| (*p, s * c)).collect(),
        }
    }

    /// `A†` (Pauli strings are Hermitian).
    pub fn adjoint(&self) -> Self {
        Self {
            terms: self.terms.iter().map(|(p, c)| (*p, c.conj())).collect(),
        }
    }

    /// `[A, B] = AB − BA`. Only anticommuting pairs contribute, each with
    /// `2·ab`.
    pub fn commutator(&self, other: &QubitOperator) -> QubitOperator {
        let mut out = QubitOperator::new();
        for (a, &ca) in &self.terms {
            for (b, &cb) in &other.terms {
                if !a.commutes_with(b) {
                    let (phase, s) = a.multiply(b);
                    out.add_term(2.0 * ca * cb * phase, s);
                }
            }
        }
        out
    }

    /// Whether `A = A†` up to `tol` in every coefficient.
    pub fn is_hermitian(&self, tol: f64) -> bool {
        self.terms.values().all(|c| c.im.abs() <= tol)
    }

    /// The Hermitian operator as a [`PauliSum`]: imaginary parts up to
    /// `tol` are dropped, larger ones are an error; terms with `|c| ≤ tol`
    /// are removed.
    pub fn to_pauli_sum(&self, tol: f64) -> Result<PauliSum> {
        if let Some((s, c)) = self.terms.iter().find(|(_, c)| c.im.abs() > tol) {
            return Err(QsoError::invalid(
                "operator",
                format!("not Hermitian: coefficient {c} of {s}"),
            ));
        }
        Ok(PauliSum::from_terms(
            self.terms
                .iter()
                .filter(|(_, c)| c.re.abs() > tol)
                .map(|(s, c)| (c.re, *s)),
        ))
    }
}

impl From<&PauliSum> for QubitOperator {
    fn from(h: &PauliSum) -> Self {
        Self::from_terms(h.terms().iter().map(|&(c, s)| (C64::new(c, 0.0), s)))
    }
}

impl From<PauliString> for QubitOperator {
    fn from(s: PauliString) -> Self {
        Self::from_terms([(C64::new(1.0, 0.0), s)])
    }
}

impl Add for &QubitOperator {
    type Output = QubitOperator;
    fn add(self, rhs: &QubitOperator) -> QubitOperator {
        let mut out = self.clone();
        for (s, &c) in &rhs.terms {
            out.add_term(c, *s);
        }
        out
    }
}

impl Sub for &QubitOperator {
    type Output = QubitOperator;
    fn sub(self, rhs: &QubitOperator) -> QubitOperator {
        let mut out = self.clone();
        for (s, &c) in &rhs.terms {
            out.add_term(-c, *s);
        }
        out
    }
}

impl Mul for &QubitOperator {
    type Output = QubitOperator;
    fn mul(self, rhs: &QubitOperator) -> QubitOperator {
        let mut out = QubitOperator::new();
        for (a, &ca) in &self.terms {
            for (b, &cb) in &rhs.terms {
                let (phase, s) = a.multiply(b);
                out.add_term(ca * cb * phase, s);
            }
        }
        out
    }
}

impl fmt::Display for QubitOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        for (k, (s, c)) in self.terms.iter().enumerate() {
            if k > 0 {
                write!(f, " + ")?;
            }
            write!(f, "({}{:+}i) {s}", c.re, c.im)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum::Pauli;

    fn op(s: &str) -> QubitOperator {
        (&s.parse::<PauliSum>().unwrap()).into()
    }

    #[test]
    fn products_and_commutators_follow_the_pauli_algebra() {
        let (x, y, z) = (op("X0"), op("Y0"), op("Z0"));
        let xy = &x * &y;
        assert_eq!(
            xy.coefficient(&PauliString::single(0, Pauli::Z)),
            C64::new(0.0, 1.0)
        );
        let c = x.commutator(&y);
        let expected = z.scaled(C64::new(0.0, 2.0));
        assert_eq!(c, expected);
        // Commuting strings give zero.
        assert!(op("Z0 Z1").commutator(&op("X0 X1")).is_empty());
        // Adjoint and Hermiticity.
        assert!(!c.is_hermitian(1e-12));
        assert!((&c + &c.adjoint()).simplify(1e-12).is_empty());
        let h = op("0.5 X0 Y1 - 2 Z0 + 1");
        assert_eq!(h.to_pauli_sum(1e-12).unwrap().num_terms(), 3);
        assert!(c.to_pauli_sum(1e-12).is_err());
        assert_eq!((&h - &h).simplify(0.0).num_terms(), 0);
    }
}
