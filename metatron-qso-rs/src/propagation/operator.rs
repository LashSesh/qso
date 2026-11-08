//! [`PauliOperator`]: real combinations of Pauli strings on any number of
//! qubits.

use std::cmp::Ordering;
use std::fmt;

use super::MAX_PROPAGATION_QUBITS;
use crate::error::{QsoError, Result};
use crate::noise::learning::SparsePauliLindblad;
use crate::quantum::pauli::parse_terms;
use crate::quantum::{Pauli, PauliString, PauliSum};

/// A real combination `Σ c_s P_s` of Hermitian Pauli strings on up to
/// [`MAX_PROPAGATION_QUBITS`] qubits: the observables and the Heisenberg
/// operators of Pauli propagation, and the rates of wide Pauli–Lindblad
/// models.
///
/// A string is stored as two bit masks of `⌈n/64⌉` words (`X` = x bit,
/// `Z` = z bit, `Y` = both, the Hermitian `Y` as in
/// [`PauliString`]). The operator is always canonical: equal strings are
/// merged (their coefficients added in the order they were given), terms
/// with coefficient 0 are removed, and the terms are sorted by their x
/// words and then their z words, word 0 (qubits 0–63) first. Up to 64
/// qubits this is the order of [`PauliSum`]. Coefficients are finite.
///
/// The text form is that of [`PauliSum`], `"0.5*Z0 Z1 + X120 - 1.05"`,
/// and [`Display`](fmt::Display) prints it so that
/// [`parse`](PauliOperator::parse) reads it back exactly.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PauliOperator {
    n_qubits: usize,
    /// Words per mask, `⌈n_qubits/64⌉`.
    words: usize,
    /// `2·words` per term: the x words, then the z words.
    keys: Vec<u64>,
    coefs: Vec<f64>,
}

/// Check a register size of Pauli propagation.
pub(crate) fn check_register(n_qubits: usize) -> Result<()> {
    if n_qubits == 0 || n_qubits > MAX_PROPAGATION_QUBITS {
        return Err(QsoError::TooManyQubits {
            what: "Pauli propagation",
            max: MAX_PROPAGATION_QUBITS,
            requested: n_qubits,
        });
    }
    Ok(())
}

/// The key (x words, then z words) of a string given by its factors, on
/// `n_qubits` qubits with `words` words per mask.
fn key_of(n_qubits: usize, words: usize, factors: &[(usize, Pauli)]) -> Result<Vec<u64>> {
    let mut key = vec![0u64; 2 * words];
    let mut used = vec![0u64; words];
    for &(q, p) in factors {
        if q >= n_qubits {
            return Err(QsoError::QubitOutOfRange { qubit: q, n_qubits });
        }
        let (w, bit) = (q / 64, 1u64 << (q % 64));
        if used[w] & bit != 0 {
            return Err(QsoError::DuplicateQubit {
                gate: "Pauli string",
                qubit: q,
            });
        }
        used[w] |= bit;
        match p {
            Pauli::I => {}
            Pauli::X => key[w] |= bit,
            Pauli::Z => key[words + w] |= bit,
            Pauli::Y => {
                key[w] |= bit;
                key[words + w] |= bit;
            }
        }
    }
    Ok(key)
}

fn non_finite() -> QsoError {
    QsoError::invalid(
        "coefficient",
        "coefficients must be finite (also after merging equal strings)",
    )
}

impl PauliOperator {
    /// The zero operator on `n_qubits` qubits (1 to
    /// [`MAX_PROPAGATION_QUBITS`]).
    pub fn new(n_qubits: usize) -> Result<Self> {
        check_register(n_qubits)?;
        Ok(PauliOperator {
            n_qubits,
            words: n_qubits.div_ceil(64),
            keys: Vec::new(),
            coefs: Vec::new(),
        })
    }

    /// Parse a sum in the grammar of [`PauliSum`]'s `FromStr`, such as
    /// `"0.5*Z0 Z1 + X120 - 1.05"`, on `n_qubits` qubits. Errors: `Parse`
    /// for malformed text, `QubitOutOfRange` for a qubit `≥ n_qubits`,
    /// `DuplicateQubit` for a qubit twice in one term, `InvalidArgument`
    /// when merged coefficients overflow.
    pub fn parse(n_qubits: usize, text: &str) -> Result<Self> {
        check_register(n_qubits)?;
        let words = n_qubits.div_ceil(64);
        let terms = parse_terms(text)?;
        let mut keys = Vec::with_capacity(terms.len() * 2 * words);
        let mut coefs = Vec::with_capacity(terms.len());
        for (c, factors) in &terms {
            keys.extend(key_of(n_qubits, words, factors)?);
            coefs.push(*c);
        }
        Self::from_rows(n_qubits, keys, coefs)
    }

    /// The sum `sum` on `n_qubits` qubits (every term must lie inside).
    pub fn from_sum(n_qubits: usize, sum: &PauliSum) -> Result<Self> {
        check_register(n_qubits)?;
        let w = n_qubits.div_ceil(64);
        let mut keys = Vec::with_capacity(sum.num_terms() * 2 * w);
        let mut coefs = Vec::with_capacity(sum.num_terms());
        for (c, s) in sum.terms() {
            let needed = s.min_qubits();
            if needed > n_qubits {
                return Err(QsoError::QubitOutOfRange {
                    qubit: needed - 1,
                    n_qubits,
                });
            }
            let mut key = vec![0u64; 2 * w];
            key[0] = s.x_mask();
            key[w] = s.z_mask();
            keys.extend(key);
            coefs.push(*c);
        }
        Self::from_rows(n_qubits, keys, coefs)
    }

    /// The rates of a sparse Pauli–Lindblad model (1.12, 1.31) as
    /// `Σ_k λ_k P_k` on a register of `n_qubits` qubits, model qubit `i`
    /// on register qubit `qubits[i]`. Generators with rate 0 are left
    /// out. The result is what [`Layer::pauli_lindblad`](super::Layer::pauli_lindblad)
    /// takes; models on several groups of qubits combine by adding their
    /// rates (they are Pauli channels, which commute).
    pub fn from_lindblad(
        model: &SparsePauliLindblad,
        qubits: &[usize],
        n_qubits: usize,
    ) -> Result<Self> {
        model.validate()?;
        check_register(n_qubits)?;
        if qubits.len() != model.n_qubits {
            return Err(QsoError::DimensionMismatch {
                expected: model.n_qubits,
                actual: qubits.len(),
            });
        }
        for (k, &q) in qubits.iter().enumerate() {
            if q >= n_qubits {
                return Err(QsoError::QubitOutOfRange { qubit: q, n_qubits });
            }
            if qubits[..k].contains(&q) {
                return Err(QsoError::DuplicateQubit {
                    gate: "qubit map",
                    qubit: q,
                });
            }
        }
        let w = n_qubits.div_ceil(64);
        let mut keys = Vec::with_capacity(model.generators.len() * 2 * w);
        let mut coefs = Vec::with_capacity(model.generators.len());
        for (g, &rate) in model.generators.iter().zip(&model.rates) {
            let mut key = vec![0u64; 2 * w];
            for (i, &q) in qubits.iter().enumerate() {
                let (wq, bit) = (q / 64, 1u64 << (q % 64));
                match g.get(i) {
                    Pauli::I => {}
                    Pauli::X => key[wq] |= bit,
                    Pauli::Z => key[w + wq] |= bit,
                    Pauli::Y => {
                        key[wq] |= bit;
                        key[w + wq] |= bit;
                    }
                }
            }
            keys.extend(key);
            coefs.push(rate);
        }
        Self::from_rows(n_qubits, keys, coefs)
    }

    /// The operator `Σ c·P` of `(c, factors)` terms on `n_qubits` qubits,
    /// each `P` given by `(qubit, Pauli)` factors (a qubit at most once;
    /// identity factors allowed): equal strings are merged in the order
    /// given and zero sums removed. Takes `O(T log T)` for `T` terms;
    /// build large operators so rather than term by term with
    /// [`add_term`](Self::add_term). Errors as [`parse`](Self::parse), and
    /// `InvalidArgument` for a coefficient that is not finite.
    pub fn from_terms<I, F>(n_qubits: usize, terms: I) -> Result<Self>
    where
        I: IntoIterator<Item = (f64, F)>,
        F: AsRef<[(usize, Pauli)]>,
    {
        check_register(n_qubits)?;
        let words = n_qubits.div_ceil(64);
        let mut keys = Vec::new();
        let mut coefs = Vec::new();
        for (c, factors) in terms {
            keys.extend(key_of(n_qubits, words, factors.as_ref())?);
            coefs.push(c);
        }
        Self::from_rows(n_qubits, keys, coefs)
    }

    /// Add `coefficient · P`, with `P` given by `(qubit, Pauli)` factors
    /// (a qubit at most once; identity factors allowed). The term is
    /// merged with an equal string, and removed if the sum is 0. A sorted
    /// insert: it takes time proportional to the operator's size, so
    /// [`from_terms`](Self::from_terms) builds many terms faster.
    pub fn add_term(&mut self, coefficient: f64, factors: &[(usize, Pauli)]) -> Result<&mut Self> {
        if !coefficient.is_finite() {
            return Err(non_finite());
        }
        let key = key_of(self.n_qubits, self.words, factors)?;
        match self.find(&key) {
            Ok(i) => {
                let c = self.coefs[i] + coefficient;
                if !c.is_finite() {
                    return Err(non_finite());
                }
                if c == 0.0 {
                    let r = 2 * self.words;
                    self.keys.drain(i * r..(i + 1) * r);
                    self.coefs.remove(i);
                } else {
                    self.coefs[i] = c;
                }
            }
            Err(i) => {
                if coefficient != 0.0 {
                    let r = 2 * self.words;
                    self.keys.splice(i * r..i * r, key);
                    self.coefs.insert(i, coefficient);
                }
            }
        }
        Ok(self)
    }

    /// The coefficient of the string given by `factors` (0 if absent).
    pub fn coefficient(&self, factors: &[(usize, Pauli)]) -> Result<f64> {
        let key = key_of(self.n_qubits, self.words, factors)?;
        Ok(self.find(&key).map_or(0.0, |i| self.coefs[i]))
    }

    /// Number of qubits of the register.
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of terms (strings with a non-zero coefficient).
    pub fn len(&self) -> usize {
        self.coefs.len()
    }

    /// Whether this is the zero operator.
    pub fn is_empty(&self) -> bool {
        self.coefs.is_empty()
    }

    /// `Σ |c_s|`, an upper bound on the operator norm.
    pub fn one_norm(&self) -> f64 {
        self.coefs.iter().map(|c| c.abs()).sum()
    }

    /// The largest number of non-identity factors of a term (0 for the
    /// zero operator and the identity).
    pub fn max_weight(&self) -> usize {
        (0..self.len())
            .map(|i| weight(self.key(i), self.words))
            .max()
            .unwrap_or(0)
    }

    /// The terms in canonical order: the coefficient and the factors of
    /// each string, by ascending qubit (identity factors left out).
    pub fn terms(&self) -> impl Iterator<Item = (f64, Vec<(usize, Pauli)>)> + '_ {
        (0..self.len()).map(move |i| (self.coefs[i], factors(self.key(i), self.words)))
    }

    /// The operator as a [`PauliSum`]; every term must lie on qubits
    /// `0..64` (else `QubitOutOfRange`).
    pub fn to_sum(&self) -> Result<PauliSum> {
        let mut terms = Vec::with_capacity(self.len());
        for (c, f) in self.terms() {
            terms.push((c, PauliString::new(&f)?));
        }
        Ok(PauliSum::from_terms(terms))
    }

    /// Words per mask, `⌈n/64⌉`.
    pub(crate) fn words(&self) -> usize {
        self.words
    }

    /// The keys (`2·words` per term: x words, then z words) and the
    /// coefficients, in canonical order.
    pub(crate) fn rows(&self) -> (&[u64], &[f64]) {
        (&self.keys, &self.coefs)
    }

    /// The operator with the given rows on `n_qubits` qubits: equal keys
    /// are merged in the order given, zero coefficients removed and the
    /// rows sorted. Keys must have `2·⌈n/64⌉` words and no bit at or
    /// beyond `n_qubits`.
    pub(crate) fn from_rows(n_qubits: usize, keys: Vec<u64>, coefs: Vec<f64>) -> Result<Self> {
        let mut op = Self::new(n_qubits)?;
        op.assign_rows(keys, coefs)?;
        Ok(op)
    }

    fn key(&self, i: usize) -> &[u64] {
        let r = 2 * self.words;
        &self.keys[i * r..(i + 1) * r]
    }

    /// Binary search for `key` among the sorted rows.
    fn find(&self, key: &[u64]) -> std::result::Result<usize, usize> {
        let (mut lo, mut hi) = (0, self.len());
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.key(mid).cmp(key) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Ok(mid),
            }
        }
        Err(lo)
    }

    /// Replace the rows by the canonical form of `keys`/`coefs`.
    fn assign_rows(&mut self, keys: Vec<u64>, coefs: Vec<f64>) -> Result<()> {
        let r = 2 * self.words;
        debug_assert_eq!(keys.len(), r * coefs.len());
        if coefs.iter().any(|c| !c.is_finite()) {
            return Err(non_finite());
        }
        let row = |i: usize| &keys[i * r..(i + 1) * r];
        let mut order: Vec<usize> = (0..coefs.len()).collect();
        // Stable: equal strings keep the order given, so they are added in it.
        order.sort_by(|&a, &b| row(a).cmp(row(b)));
        let mut out_keys = Vec::with_capacity(keys.len());
        let mut out_coefs = Vec::with_capacity(coefs.len());
        let mut i = 0;
        while i < order.len() {
            let mut c = 0.0;
            let mut j = i;
            while j < order.len() && row(order[j]) == row(order[i]) {
                c += coefs[order[j]];
                j += 1;
            }
            if !c.is_finite() {
                return Err(non_finite());
            }
            if c != 0.0 {
                out_keys.extend_from_slice(row(order[i]));
                out_coefs.push(c);
            }
            i = j;
        }
        self.keys = out_keys;
        self.coefs = out_coefs;
        Ok(())
    }
}

/// Number of non-identity factors of a key.
pub(crate) fn weight(key: &[u64], words: usize) -> usize {
    (0..words)
        .map(|w| (key[w] | key[words + w]).count_ones() as usize)
        .sum()
}

/// The factors of a key, by ascending qubit.
fn factors(key: &[u64], words: usize) -> Vec<(usize, Pauli)> {
    let mut out = Vec::new();
    for w in 0..words {
        let (x, z) = (key[w], key[words + w]);
        let mut bits = x | z;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let p = match ((x >> b) & 1, (z >> b) & 1) {
                (1, 0) => Pauli::X,
                (0, 1) => Pauli::Z,
                _ => Pauli::Y,
            };
            out.push((64 * w + b, p));
        }
    }
    out
}

impl fmt::Display for PauliOperator {
    /// The format of [`PauliSum`]: `"0.5 X0 Z1 - 1 Y70"`, the identity
    /// term without factors, `"0"` for the zero operator.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "0");
        }
        for (k, (c, factors)) in self.terms().enumerate() {
            if k == 0 {
                write!(f, "{c}")?;
            } else if c < 0.0 {
                write!(f, " - {}", -c)?;
            } else {
                write!(f, " + {c}")?;
            }
            for (q, p) in factors {
                write!(f, " {p:?}{q}")?;
            }
        }
        Ok(())
    }
}
