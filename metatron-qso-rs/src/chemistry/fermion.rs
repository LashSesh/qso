//! Fermionic operators: sums of products of creation and annihilation
//! operators on numbered modes.
//!
//! A term is written as in OpenFermion, `"0.5 [2^ 0] + 0.5 [0^ 2]"`, where
//! `p^` creates and `p` annihilates a fermion in mode `p`. The operators
//! obey `{a_p, a_q†} = δ_pq`; [`FermionOperator::normal_ordered`] brings a
//! sum into the canonical form (creators left, indices descending), which
//! makes equal operators compare equal.
//!
//! ```
//! use metatron_qso::chemistry::FermionOperator;
//!
//! let a: FermionOperator = "[0]".parse().unwrap();
//! let b: FermionOperator = "[0^]".parse().unwrap();
//! // {a_0, a_0†} = 1
//! let anti = (&(&a * &b) + &(&b * &a)).normal_ordered();
//! assert_eq!(anti, "1 []".parse::<FermionOperator>().unwrap());
//! // a_1† a_0 = −a_0 a_1†
//! let hop: FermionOperator = "[1^ 0]".parse().unwrap();
//! assert_eq!(hop.normal_ordered(), "-1 [0 1^]".parse::<FermionOperator>().unwrap().normal_ordered());
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Mul, Sub};
use std::str::FromStr;

use num_complex::Complex64 as C64;

use crate::error::{QsoError, Result};

/// Largest mode index + 1 (Pauli strings hold 64 qubits).
pub const MAX_MODES: usize = 64;

/// One ladder operator: `a_mode†` if `dagger`, else `a_mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ladder {
    pub mode: usize,
    pub dagger: bool,
}

impl Ladder {
    /// `a_mode†`.
    pub fn create(mode: usize) -> Self {
        Self { mode, dagger: true }
    }

    /// `a_mode`.
    pub fn annihilate(mode: usize) -> Self {
        Self {
            mode,
            dagger: false,
        }
    }
}

/// `Σ_k c_k Π_j ℓ_kj` with ladder operators `ℓ` and complex `c_k`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FermionOperator {
    terms: BTreeMap<Vec<Ladder>, C64>,
}

impl FermionOperator {
    /// The zero operator.
    pub fn new() -> Self {
        Self::default()
    }

    /// `c · ℓ_1 ℓ_2 ⋯` (an empty product is the identity).
    pub fn term(c: impl Into<C64>, ops: &[Ladder]) -> Result<Self> {
        let mut f = Self::new();
        f.add_term(c, ops)?;
        Ok(f)
    }

    /// Add `c · ℓ_1 ℓ_2 ⋯`.
    pub fn add_term(&mut self, c: impl Into<C64>, ops: &[Ladder]) -> Result<&mut Self> {
        let c = c.into();
        if let Some(l) = ops.iter().find(|l| l.mode >= MAX_MODES) {
            return Err(QsoError::invalid(
                "mode",
                format!("mode {} exceeds the maximum {}", l.mode, MAX_MODES - 1),
            ));
        }
        if !(c.re.is_finite() && c.im.is_finite()) {
            return Err(QsoError::invalid("coefficient", "must be finite"));
        }
        *self.terms.entry(ops.to_vec()).or_default() += c;
        Ok(self)
    }

    /// `a_p† a_q`.
    pub fn hopping(p: usize, q: usize) -> Result<Self> {
        Self::term(1.0, &[Ladder::create(p), Ladder::annihilate(q)])
    }

    /// The number operator `a_p† a_p`.
    pub fn number(p: usize) -> Result<Self> {
        Self::hopping(p, p)
    }

    /// The terms as `(coefficient, product)`.
    pub fn terms(&self) -> impl Iterator<Item = (C64, &[Ladder])> + '_ {
        self.terms.iter().map(|(ops, &c)| (c, ops.as_slice()))
    }

    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Number of modes the operator acts on (largest index + 1).
    pub fn num_modes(&self) -> usize {
        self.terms
            .keys()
            .flat_map(|ops| ops.iter().map(|l| l.mode + 1))
            .max()
            .unwrap_or(0)
    }

    /// Drop terms with `|c| ≤ tol`.
    pub fn simplify(&mut self, tol: f64) -> &mut Self {
        self.terms.retain(|_, c| c.norm() > tol);
        self
    }

    /// `s · F`.
    pub fn scaled(&self, s: impl Into<C64>) -> Self {
        let s = s.into();
        Self {
            terms: self
                .terms
                .iter()
                .map(|(o, &c)| (o.clone(), s * c))
                .collect(),
        }
    }

    /// `F†`: reversed products, daggers flipped, coefficients conjugated.
    pub fn adjoint(&self) -> Self {
        let mut out = Self::new();
        for (ops, c) in &self.terms {
            let rev: Vec<Ladder> = ops
                .iter()
                .rev()
                .map(|l| Ladder {
                    mode: l.mode,
                    dagger: !l.dagger,
                })
                .collect();
            *out.terms.entry(rev).or_default() += c.conj();
        }
        out
    }

    /// The canonical form: every product ordered with creators first and
    /// indices descending within each kind, using the anticommutation
    /// relations; products with a repeated creator or annihilator vanish,
    /// and zero terms are removed.
    pub fn normal_ordered(&self) -> Self {
        let mut out = Self::new();
        let mut stack: Vec<(C64, Vec<Ladder>)> =
            self.terms.iter().map(|(o, &c)| (c, o.clone())).collect();
        while let Some((c, mut ops)) = stack.pop() {
            // Bubble sort; a swap of a_p a_p† (p equal) spawns δ-term.
            let mut sign = 1.0;
            let mut zero = false;
            'sort: loop {
                let mut swapped = false;
                for i in 0..ops.len().saturating_sub(1) {
                    let (l, r) = (ops[i], ops[i + 1]);
                    let before = |a: Ladder, b: Ladder| {
                        (a.dagger && !b.dagger) || (a.dagger == b.dagger && a.mode > b.mode)
                    };
                    if l == r {
                        zero = true;
                        break 'sort;
                    }
                    if before(l, r) {
                        continue;
                    }
                    // r must move left of l.
                    if !l.dagger && r.dagger && l.mode == r.mode {
                        // a_p a_p† = 1 − a_p† a_p
                        let mut rest = ops[..i].to_vec();
                        rest.extend_from_slice(&ops[i + 2..]);
                        stack.push((sign * c, rest));
                    }
                    ops.swap(i, i + 1);
                    sign = -sign;
                    swapped = true;
                }
                if !swapped {
                    break;
                }
            }
            if !zero {
                *out.terms.entry(ops).or_default() += sign * c;
            }
        }
        out.simplify(0.0);
        out
    }

    /// Whether `F = F†` up to `tol` after normal ordering.
    pub fn is_hermitian(&self, tol: f64) -> bool {
        let mut d = (self - &self.adjoint()).normal_ordered();
        d.simplify(tol).is_empty()
    }
}

impl Add for &FermionOperator {
    type Output = FermionOperator;
    fn add(self, rhs: &FermionOperator) -> FermionOperator {
        let mut out = self.clone();
        for (o, &c) in &rhs.terms {
            *out.terms.entry(o.clone()).or_default() += c;
        }
        out
    }
}

impl Sub for &FermionOperator {
    type Output = FermionOperator;
    fn sub(self, rhs: &FermionOperator) -> FermionOperator {
        self + &rhs.scaled(-1.0)
    }
}

impl Mul for &FermionOperator {
    type Output = FermionOperator;
    fn mul(self, rhs: &FermionOperator) -> FermionOperator {
        let mut out = FermionOperator::new();
        for (a, &ca) in &self.terms {
            for (b, &cb) in &rhs.terms {
                let mut ops = a.clone();
                ops.extend_from_slice(b);
                *out.terms.entry(ops).or_default() += ca * cb;
            }
        }
        out
    }
}

impl fmt::Display for FermionOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        for (k, (ops, c)) in self.terms.iter().enumerate() {
            if k > 0 {
                write!(f, " + ")?;
            }
            if c.im == 0.0 {
                write!(f, "{} [", c.re)?;
            } else {
                write!(f, "({}{:+}j) [", c.re, c.im)?;
            }
            for (i, l) in ops.iter().enumerate() {
                if i > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}{}", l.mode, if l.dagger { "^" } else { "" })?;
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}

fn parse_coefficient(s: &str) -> Result<C64> {
    let s = s.trim();
    let bad = || QsoError::invalid("fermion operator", format!("invalid coefficient '{s}'"));
    if s.is_empty() || s == "+" {
        return Ok(C64::new(1.0, 0.0));
    }
    if s == "-" {
        return Ok(C64::new(-1.0, 0.0));
    }
    let inner = s.trim_start_matches('+');
    let (neg, inner) = match inner.strip_prefix('-') {
        Some(rest) if rest.trim_start().starts_with('(') => (true, rest.trim()),
        _ => (false, inner.trim()),
    };
    let c = if let Some(body) = inner.strip_prefix('(').and_then(|b| b.strip_suffix(')')) {
        // (a+bj) or (a-bj) or (bj)
        let body = body.trim();
        let body = body.strip_suffix('j').ok_or_else(bad)?;
        let split = body
            .char_indices()
            .skip(1)
            .filter(|&(i, ch)| {
                (ch == '+' || ch == '-') && !matches!(body.as_bytes()[i - 1], b'e' | b'E')
            })
            .map(|(i, _)| i)
            .last();
        match split {
            Some(i) => C64::new(
                body[..i].trim().parse().map_err(|_| bad())?,
                body[i..].trim().parse().map_err(|_| bad())?,
            ),
            None => C64::new(0.0, body.trim().parse().map_err(|_| bad())?),
        }
    } else {
        C64::new(inner.replace(' ', "").parse().map_err(|_| bad())?, 0.0)
    };
    if !(c.re.is_finite() && c.im.is_finite()) {
        return Err(bad());
    }
    Ok(if neg { -c } else { c })
}

impl FromStr for FermionOperator {
    type Err = QsoError;

    /// Parse `"c [p^ q ...] + c' [...] ..."`; coefficients are real numbers
    /// or `(a+bj)`, and default to 1.
    fn from_str(s: &str) -> Result<Self> {
        let mut out = FermionOperator::new();
        let mut rest = s.trim();
        if rest.is_empty() {
            return Err(QsoError::invalid("fermion operator", "empty expression"));
        }
        while !rest.is_empty() {
            let open = rest.find('[').ok_or_else(|| {
                QsoError::invalid("fermion operator", format!("expected '[' in '{rest}'"))
            })?;
            let close = rest[open..]
                .find(']')
                .map(|i| open + i)
                .ok_or_else(|| QsoError::invalid("fermion operator", "missing ']'"))?;
            let mut coefficient = rest[..open].trim();
            // A leading '+' or '-' between terms belongs to the coefficient.
            if coefficient.ends_with('*') {
                coefficient = coefficient[..coefficient.len() - 1].trim();
            }
            let c = parse_coefficient(coefficient)?;
            let mut ops = Vec::new();
            for tok in rest[open + 1..close].split_whitespace() {
                let (num, dagger) = match tok.strip_suffix('^') {
                    Some(n) => (n, true),
                    None => (tok, false),
                };
                let mode: usize = num.parse().map_err(|_| {
                    QsoError::invalid("fermion operator", format!("invalid mode '{tok}'"))
                })?;
                ops.push(Ladder { mode, dagger });
            }
            out.add_term(c, &ops)?;
            rest = rest[close + 1..].trim_start();
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(s: &str) -> FermionOperator {
        s.parse().unwrap()
    }

    #[test]
    fn anticommutation_relations() {
        for p in 0..3 {
            for q in 0..3 {
                let a = f(&format!("[{p}]"));
                let b = f(&format!("[{q}^]"));
                let anti = (&(&a * &b) + &(&b * &a)).normal_ordered();
                if p == q {
                    assert_eq!(anti, f("[]"));
                } else {
                    assert!(anti.is_empty(), "{p} {q}: {anti}");
                }
                let aa = (&(&a * &f(&format!("[{q}]"))) + &(&f(&format!("[{q}]")) * &a))
                    .normal_ordered();
                assert!(aa.is_empty());
            }
        }
        // Repeated creators vanish.
        assert!(f("[1^ 1^]").normal_ordered().is_empty());
        // Canonical order: a_0† a_1† = −a_1† a_0†.
        assert_eq!(f("[0^ 1^]").normal_ordered(), f("-1 [1^ 0^]"));
    }

    #[test]
    fn parsing_display_and_adjoint() {
        let h = f("0.5 [2^ 0] + 0.5 [0^ 2] - [1^ 1] + (0+2j) [3^ 2^ 1 0]");
        assert_eq!(h.num_terms(), 4);
        assert_eq!(h.num_modes(), 4);
        assert!(!h.is_hermitian(1e-12));
        let herm = f("0.5 [2^ 0] + 0.5 [0^ 2] - [1^ 1]");
        assert!(herm.is_hermitian(1e-12));
        let round: FermionOperator = h.to_string().parse().unwrap();
        assert_eq!(round, h);
        assert_eq!(h.adjoint().adjoint(), h);
        assert_eq!(f("(1-2j) [0^]").adjoint(), f("(1+2j) [0]"));
        assert_eq!(f("-(1-2j) [0^]"), f("(-1+2j) [0^]"));
        assert_eq!(f("1e-3 [0^] - 2.5e+2 [1]").num_terms(), 2);
        for bad in ["", "[0^", "x [0]", "[a]", "[64^]", "nan [0]", "(1+2) [0]"] {
            assert!(bad.parse::<FermionOperator>().is_err(), "{bad}");
        }
    }
}
