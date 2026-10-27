//! Molecular Hamiltonians from one- and two-electron integrals.
//!
//! [`MolecularIntegrals`] holds the integrals over `m` real spatial
//! orbitals in chemists' notation, as written by quantum-chemistry codes in
//! the FCIDUMP format (Knowles and Handy, Comput. Phys. Commun. 54, 75
//! (1989)):
//!
//! ```text
//! H = E_core + Σ_{pq,σ} h_pq a†_pσ a_qσ
//!     + ½ Σ_{pqrs,στ} (pq|rs) a†_pσ a†_rτ a_sτ a_qσ
//! ```
//!
//! Spin orbitals are in block order (`pα = p`, `pβ = m + p`). The
//! Hartree–Fock determinant fills the lowest `n_α` and `n_β` orbitals, which
//! is right for files written from a self-consistent-field calculation.
//! [`MolecularIntegrals::fci_energies`] diagonalises the Hamiltonian in the
//! space of all determinants with the same electron numbers (full
//! configuration interaction).
//!
//! ```
//! use metatron_qso::chemistry::{Mapping, MolecularIntegrals, QubitEncoding};
//!
//! let h2 = MolecularIntegrals::from_fcidump(include_str!(
//!     "../../examples/molecules/h2_sto3g.fcidump"
//! )).unwrap();
//! let h = h2.qubit_hamiltonian(QubitEncoding::new(Mapping::JordanWigner)).unwrap();
//! let fci = h2.fci_energies(1).unwrap()[0];
//! assert!((fci - (-1.137306)).abs() < 1e-6);
//! assert!((h.ground_state_energy(4).unwrap() - fci).abs() < 1e-9);
//! ```

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

use num_complex::Complex64 as C64;

use super::fermion::{FermionOperator, Ladder};
use super::mapping::QubitEncoding;
use crate::error::{QsoError, Result};
use crate::linalg::{KrylovOptions, RMatrix, eigh_real, lanczos_ground_state};
use crate::quantum::{Circuit, PauliString, PauliSum};

/// Largest number of spatial orbitals (64 spin orbitals).
pub const MAX_ORBITALS: usize = 32;
/// Largest determinant space diagonalised densely by
/// [`MolecularIntegrals::fci_energies`]; larger spaces use Lanczos for the
/// ground state only.
pub const DENSE_FCI_LIMIT: usize = 4096;
/// Largest determinant space [`MolecularIntegrals::fci_energies`] accepts.
pub const MAX_DETERMINANTS: usize = 1 << 18;

/// `C(n, k)`, saturating.
fn binomial(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut c: u128 = 1;
    for i in 0..k {
        c = c.saturating_mul((n - i) as u128) / (i + 1) as u128;
    }
    c
}

/// One- and two-electron integrals of a molecule with its electron count.
#[derive(Clone, Debug, PartialEq)]
pub struct MolecularIntegrals {
    n_orbitals: usize,
    n_alpha: usize,
    n_beta: usize,
    core_energy: f64,
    /// `h[p·m + q]`.
    one_body: Vec<f64>,
    /// `(pq|rs)` at `((p·m + q)·m + r)·m + s`.
    two_body: Vec<f64>,
}

fn idx2(m: usize, p: usize, q: usize) -> usize {
    p * m + q
}

fn idx4(m: usize, p: usize, q: usize, r: usize, s: usize) -> usize {
    ((p * m + q) * m + r) * m + s
}

impl MolecularIntegrals {
    /// Integrals over `n_orbitals` orbitals; `one_body` is `m × m` and
    /// `two_body` is `m⁴` (row-major, chemists' notation). Both must have
    /// the symmetries of real orbitals.
    pub fn new(
        n_orbitals: usize,
        n_alpha: usize,
        n_beta: usize,
        core_energy: f64,
        one_body: Vec<f64>,
        two_body: Vec<f64>,
    ) -> Result<Self> {
        let m = n_orbitals;
        if m == 0 || m > MAX_ORBITALS {
            return Err(QsoError::invalid(
                "n_orbitals",
                format!("need 1 to {MAX_ORBITALS} orbitals, got {m}"),
            ));
        }
        if n_alpha > m || n_beta > m {
            return Err(QsoError::invalid(
                "electrons",
                "more electrons than orbitals",
            ));
        }
        if one_body.len() != m * m || two_body.len() != m * m * m * m {
            return Err(QsoError::invalid(
                "integrals",
                "one_body needs m² and two_body m⁴ entries",
            ));
        }
        if !core_energy.is_finite() || one_body.iter().chain(&two_body).any(|v| !v.is_finite()) {
            return Err(QsoError::invalid("integrals", "values must be finite"));
        }
        let tol = 1e-10;
        for p in 0..m {
            for q in 0..m {
                if (one_body[idx2(m, p, q)] - one_body[idx2(m, q, p)]).abs() > tol {
                    return Err(QsoError::invalid("one_body", "must be symmetric"));
                }
                for r in 0..m {
                    for s in 0..m {
                        let v = two_body[idx4(m, p, q, r, s)];
                        for w in [
                            idx4(m, q, p, r, s),
                            idx4(m, p, q, s, r),
                            idx4(m, r, s, p, q),
                        ] {
                            if (v - two_body[w]).abs() > tol {
                                return Err(QsoError::invalid(
                                    "two_body",
                                    "must have the 8-fold symmetry of real orbitals",
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok(Self {
            n_orbitals,
            n_alpha,
            n_beta,
            core_energy,
            one_body,
            two_body,
        })
    }

    /// Read an FCIDUMP file (restricted orbitals; `NORB`, `NELEC` and
    /// optionally `MS2` in the header). Integrals may be listed with any of
    /// their symmetric index orders; orbital energies are ignored.
    pub fn from_fcidump(text: &str) -> Result<Self> {
        let bad = |msg: String| QsoError::invalid("FCIDUMP", msg);
        let upper = text.to_ascii_uppercase();
        let start = ["&FCI", "$FCI"]
            .iter()
            .filter_map(|t| upper.find(t))
            .min()
            .ok_or_else(|| bad("missing '&FCI' header".into()))?;
        let header_end = ["&END", "$END", "/"]
            .iter()
            .filter_map(|t| upper[start..].find(t).map(|i| (start + i, t.len())))
            .min()
            .ok_or_else(|| bad("missing '&END' or '/' after the header".into()))?;
        let header = &upper[start + 4..header_end.0];
        // Namelist items are separated by commas or blanks; `KEY = v` may
        // carry blanks around '='.
        let mut normalised = String::with_capacity(header.len());
        for part in header.split('=') {
            if !normalised.is_empty() {
                normalised.push('=');
            }
            normalised.push_str(part.trim());
        }
        let mut fields: HashMap<String, Vec<String>> = HashMap::new();
        let mut key = String::new();
        for token in normalised.split(|c: char| c == ',' || c.is_whitespace()) {
            if token.is_empty() {
                continue;
            }
            if let Some((k, v)) = token.split_once('=') {
                key = k.to_string();
                let entry = fields.entry(key.clone()).or_default();
                if !v.is_empty() {
                    entry.push(v.to_string());
                }
            } else if !key.is_empty() {
                fields
                    .entry(key.clone())
                    .or_default()
                    .push(token.to_string());
            }
        }
        let int = |name: &str, default: Option<i64>| -> Result<i64> {
            match fields.get(name).and_then(|v| v.first()) {
                Some(v) => v
                    .parse()
                    .map_err(|_| bad(format!("invalid {name} = '{v}'"))),
                None => default.ok_or_else(|| bad(format!("missing {name}"))),
            }
        };
        if int("IUHF", Some(0))? != 0 {
            return Err(bad("unrestricted (IUHF) files are not supported".into()));
        }
        let norb = int("NORB", None)?;
        let nelec = int("NELEC", None)?;
        let ms2 = int("MS2", Some(0))?;
        if norb < 1 || norb > MAX_ORBITALS as i64 {
            return Err(bad(format!(
                "NORB must lie in 1..={MAX_ORBITALS}, got {norb}"
            )));
        }
        if nelec < 0
            || nelec > 2 * norb
            || ms2.unsigned_abs() > nelec.unsigned_abs()
            || (nelec + ms2) % 2 != 0
        {
            return Err(bad(format!("inconsistent NELEC = {nelec}, MS2 = {ms2}")));
        }
        let m = norb as usize;
        let n_alpha = ((nelec + ms2) / 2) as usize;
        let n_beta = ((nelec - ms2) / 2) as usize;
        let mut one_body = vec![0.0; m * m];
        let mut two_body = vec![0.0; m * m * m * m];
        let mut core = 0.0;
        let body = &text[header_end.0 + header_end.1..];
        for line in body.lines() {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }
            if tokens.len() != 5 {
                return Err(bad(format!("expected 'value i j k l' in line '{line}'")));
            }
            let value: f64 = tokens[0]
                .replace(['D', 'd'], "e")
                .parse()
                .map_err(|_| bad(format!("invalid value '{}'", tokens[0])))?;
            if !value.is_finite() {
                return Err(bad(format!("non-finite value in line '{line}'")));
            }
            let mut ix = [0usize; 4];
            for (slot, t) in ix.iter_mut().zip(&tokens[1..]) {
                let v: usize = t.parse().map_err(|_| bad(format!("invalid index '{t}'")))?;
                if v > m {
                    return Err(bad(format!("index {v} exceeds NORB = {m}")));
                }
                *slot = v;
            }
            match ix {
                [0, 0, 0, 0] => core += value,
                [i, j, 0, 0] if i > 0 && j > 0 => {
                    one_body[idx2(m, i - 1, j - 1)] = value;
                    one_body[idx2(m, j - 1, i - 1)] = value;
                }
                [_, 0, 0, 0] => {} // orbital energy
                [i, j, k, l] if i > 0 && j > 0 && k > 0 && l > 0 => {
                    let (p, q, r, s) = (i - 1, j - 1, k - 1, l - 1);
                    for (a, b, c, d) in [
                        (p, q, r, s),
                        (q, p, r, s),
                        (p, q, s, r),
                        (q, p, s, r),
                        (r, s, p, q),
                        (s, r, p, q),
                        (r, s, q, p),
                        (s, r, q, p),
                    ] {
                        two_body[idx4(m, a, b, c, d)] = value;
                    }
                }
                _ => return Err(bad(format!("invalid index pattern in line '{line}'"))),
            }
        }
        Self::new(m, n_alpha, n_beta, core, one_body, two_body)
    }

    /// Write the integrals as an FCIDUMP file (unique entries only).
    pub fn to_fcidump(&self) -> String {
        let m = self.n_orbitals;
        let mut out = String::new();
        let _ = writeln!(
            out,
            " &FCI NORB={m},NELEC={},MS2={},\n  ORBSYM={},\n  ISYM=1,\n &END",
            self.n_alpha + self.n_beta,
            self.n_alpha as i64 - self.n_beta as i64,
            vec!["1"; m].join(",")
        );
        for p in 0..m {
            for q in 0..=p {
                for r in 0..m {
                    for s in 0..=r {
                        if p * (p + 1) / 2 + q < r * (r + 1) / 2 + s {
                            continue;
                        }
                        let v = self.two_body[idx4(m, p, q, r, s)];
                        if v != 0.0 {
                            let _ = writeln!(out, " {v:e} {} {} {} {}", p + 1, q + 1, r + 1, s + 1);
                        }
                    }
                }
            }
        }
        for p in 0..m {
            for q in 0..=p {
                let v = self.one_body[idx2(m, p, q)];
                if v != 0.0 {
                    let _ = writeln!(out, " {v:e} {} {} 0 0", p + 1, q + 1);
                }
            }
        }
        let _ = writeln!(out, " {:e} 0 0 0 0", self.core_energy);
        out
    }

    pub fn n_orbitals(&self) -> usize {
        self.n_orbitals
    }

    pub fn n_spin_orbitals(&self) -> usize {
        2 * self.n_orbitals
    }

    pub fn n_alpha(&self) -> usize {
        self.n_alpha
    }

    pub fn n_beta(&self) -> usize {
        self.n_beta
    }

    /// Nuclear repulsion plus any frozen-core energy.
    pub fn core_energy(&self) -> f64 {
        self.core_energy
    }

    /// `h_pq`.
    pub fn one_body(&self, p: usize, q: usize) -> f64 {
        self.one_body[idx2(self.n_orbitals, p, q)]
    }

    /// `(pq|rs)`.
    pub fn two_body(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        self.two_body[idx4(self.n_orbitals, p, q, r, s)]
    }

    /// The Hamiltonian as a fermionic operator on `2m` modes, the core
    /// energy as its constant term.
    pub fn fermion_hamiltonian(&self) -> FermionOperator {
        let m = self.n_orbitals;
        let mut terms: BTreeMap<Vec<Ladder>, f64> = BTreeMap::new();
        terms.insert(Vec::new(), self.core_energy);
        let spins = [0, m];
        for p in 0..m {
            for q in 0..m {
                let h = self.one_body(p, q);
                if h != 0.0 {
                    for &s in &spins {
                        *terms
                            .entry(vec![Ladder::create(p + s), Ladder::annihilate(q + s)])
                            .or_default() += h;
                    }
                }
            }
        }
        for p in 0..m {
            for q in 0..m {
                for r in 0..m {
                    for s in 0..m {
                        let v = self.two_body(p, q, r, s);
                        if v == 0.0 {
                            continue;
                        }
                        for &a in &spins {
                            for &b in &spins {
                                if a == b && p == r {
                                    continue; // a†_p a†_p = 0
                                }
                                *terms
                                    .entry(vec![
                                        Ladder::create(p + a),
                                        Ladder::create(r + b),
                                        Ladder::annihilate(s + b),
                                        Ladder::annihilate(q + a),
                                    ])
                                    .or_default() += 0.5 * v;
                            }
                        }
                    }
                }
            }
        }
        let mut f = FermionOperator::new();
        for (ops, c) in terms {
            if c != 0.0 {
                f.add_term(c, &ops).expect("modes are in range");
            }
        }
        f
    }

    /// The qubit Hamiltonian under `encoding` (constant term = core energy).
    pub fn qubit_hamiltonian(&self, encoding: QubitEncoding) -> Result<PauliSum> {
        encoding.map_hermitian(
            &self.fermion_hamiltonian(),
            self.n_spin_orbitals(),
            self.n_alpha,
            self.n_beta,
        )
    }

    /// Occupation bit mask of the Hartree–Fock determinant.
    pub fn hartree_fock_occupation(&self) -> u64 {
        let alpha = (1u64 << self.n_alpha) - 1;
        let beta = ((1u64 << self.n_beta) - 1) << self.n_orbitals;
        alpha | beta
    }

    /// Energy of the Hartree–Fock determinant,
    /// `E_core + Σ_i h_ii + ½ Σ_ij [(ii|jj) − δ_σiσj (ij|ji)]`.
    pub fn hartree_fock_energy(&self) -> f64 {
        let occ: Vec<(usize, usize)> = (0..self.n_alpha)
            .map(|p| (p, 0))
            .chain((0..self.n_beta).map(|p| (p, 1)))
            .collect();
        let mut e = self.core_energy;
        for &(i, _) in &occ {
            e += self.one_body(i, i);
        }
        for &(i, si) in &occ {
            for &(j, sj) in &occ {
                e += 0.5 * self.two_body(i, i, j, j);
                if si == sj {
                    e -= 0.5 * self.two_body(i, j, j, i);
                }
            }
        }
        e
    }

    /// Circuit preparing the Hartree–Fock determinant under `encoding`
    /// (X gates on the encoded occupation).
    pub fn hartree_fock_circuit(&self, encoding: QubitEncoding) -> Result<Circuit> {
        let n = self.n_spin_orbitals();
        let nq = encoding.n_qubits(n);
        // Validates the encoding for these electron numbers.
        encoding.map(&FermionOperator::new(), n, self.n_alpha, self.n_beta)?;
        let bits = encoding.encode(self.hartree_fock_occupation(), n)?;
        let mut c = Circuit::new(nq);
        for q in 0..nq {
            if (bits >> q) & 1 == 1 {
                c.x(q);
            }
        }
        Ok(c)
    }

    /// Occupation bit masks of all determinants with `n_α` and `n_β`
    /// electrons, in increasing order (at most [`MAX_DETERMINANTS`]).
    pub fn determinants(&self) -> Result<Vec<u64>> {
        let dim = binomial(self.n_orbitals, self.n_alpha)
            .saturating_mul(binomial(self.n_orbitals, self.n_beta));
        if dim > MAX_DETERMINANTS as u128 {
            return Err(QsoError::invalid(
                "molecule",
                format!("{dim} determinants exceed the limit of {MAX_DETERMINANTS}"),
            ));
        }
        let m = self.n_orbitals;
        let alpha = combinations(m, self.n_alpha);
        let beta = combinations(m, self.n_beta);
        let mut out = Vec::with_capacity(alpha.len() * beta.len());
        for &b in &beta {
            for &a in &alpha {
                out.push(a | (b << m));
            }
        }
        Ok(out)
    }

    /// The lowest `n_roots` full-CI energies (core energy included) in the
    /// space of determinants with `n_α` and `n_β` electrons. Spaces up to
    /// [`DENSE_FCI_LIMIT`] determinants are diagonalised densely; larger
    /// ones, up to [`MAX_DETERMINANTS`], give the ground state by Lanczos
    /// (`n_roots` must then be 1), as does a single root above 1024
    /// determinants. The matrix follows from the Slater–Condon rules.
    pub fn fci_energies(&self, n_roots: usize) -> Result<Vec<f64>> {
        let basis = self.determinants()?;
        let index: HashMap<u64, usize> = basis.iter().enumerate().map(|(i, &b)| (b, i)).collect();
        let columns = basis
            .iter()
            .map(|&d| self.slater_condon(d, &index))
            .collect();
        lowest_eigenvalues(columns, n_roots)
    }

    /// Spin-orbital integral `⟨pq|rs⟩ = (pr|qs) δ_{σp σr} δ_{σq σs}`
    /// (spin orbitals in block order).
    fn physicist(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let m = self.n_orbitals;
        if (p < m) != (r < m) || (q < m) != (s < m) {
            return 0.0;
        }
        self.two_body[idx4(m, p % m, r % m, q % m, s % m)]
    }

    /// `⟨pq||rs⟩ = ⟨pq|rs⟩ − ⟨pq|sr⟩`.
    fn antisymmetrised(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        self.physicist(p, q, r, s) - self.physicist(p, q, s, r)
    }

    fn spin_one_body(&self, p: usize, q: usize) -> f64 {
        let m = self.n_orbitals;
        if (p < m) != (q < m) {
            return 0.0;
        }
        self.one_body[idx2(m, p % m, q % m)]
    }

    /// Column `H|d⟩` in the determinant basis by the Slater–Condon rules.
    fn slater_condon(&self, d: u64, index: &HashMap<u64, usize>) -> Vec<(usize, f64)> {
        let n = self.n_spin_orbitals();
        let occ: Vec<usize> = (0..n).filter(|&p| d >> p & 1 == 1).collect();
        let virt: Vec<usize> = (0..n).filter(|&p| d >> p & 1 == 0).collect();
        let mut col = Vec::new();
        // Diagonal.
        let mut diag = self.core_energy;
        for (k, &i) in occ.iter().enumerate() {
            diag += self.spin_one_body(i, i);
            for &j in &occ[..k] {
                diag += self.antisymmetrised(i, j, i, j);
            }
        }
        col.push((index[&d], diag));
        // Singles i → a.
        for &i in &occ {
            for &a in &virt {
                if (i < self.n_orbitals) != (a < self.n_orbitals) {
                    continue;
                }
                let mut v = self.spin_one_body(a, i);
                for &j in &occ {
                    if j != i {
                        v += self.antisymmetrised(a, j, i, j);
                    }
                }
                if v != 0.0 {
                    let (phase, e) = create(annihilate((1.0, d), i), a);
                    col.push((index[&e], phase * v));
                }
            }
        }
        // Doubles i < j → a < b: ⟨ab||ij⟩ times the sign of a†_a a†_b a_j a_i.
        for (x, &i) in occ.iter().enumerate() {
            for &j in &occ[x + 1..] {
                for (y, &a) in virt.iter().enumerate() {
                    for &b in &virt[y + 1..] {
                        let v = self.antisymmetrised(a, b, i, j);
                        if v == 0.0 {
                            continue;
                        }
                        let state = annihilate(annihilate((1.0, d), i), j);
                        let (phase, e) = create(create(state, b), a);
                        col.push((index[&e], phase * v));
                    }
                }
            }
        }
        col.sort_by_key(|e| e.0);
        col
    }

    /// [`MolecularIntegrals::fci_energies`] computed from the qubit
    /// Hamiltonian of `encoding` in the span of the encoded determinants.
    /// All encodings give the same energies; comparing them checks the
    /// mappings.
    pub fn fci_energies_with(&self, encoding: QubitEncoding, n_roots: usize) -> Result<Vec<f64>> {
        let n = self.n_spin_orbitals();
        let basis = self
            .determinants()?
            .into_iter()
            .map(|d| encoding.encode(d, n))
            .collect::<Result<Vec<u64>>>()?;
        let h = self.qubit_hamiltonian(encoding)?;
        sector_energies(&h, &basis, n_roots)
    }
}

/// All `n`-bit masks with `k` bits set, in increasing order.
fn combinations(n: usize, k: usize) -> Vec<u64> {
    if k > n {
        return Vec::new();
    }
    if k == 0 {
        return vec![0];
    }
    let mut out = Vec::new();
    let mut v: u64 = (1 << k) - 1;
    let limit = 1u64 << n;
    while v < limit {
        out.push(v);
        // Gosper's hack.
        let c = v & v.wrapping_neg();
        let r = v + c;
        v = (((r ^ v) >> 2) / c) | r;
    }
    out
}

/// `a_p` on `phase·|d⟩` (`p` occupied).
fn annihilate((phase, d): (f64, u64), p: usize) -> (f64, u64) {
    let sign = if (d & ((1u64 << p) - 1)).count_ones() % 2 == 1 {
        -phase
    } else {
        phase
    };
    (sign, d & !(1u64 << p))
}

/// `a†_p` on `phase·|d⟩` (`p` empty).
fn create((phase, d): (f64, u64), p: usize) -> (f64, u64) {
    let sign = if (d & ((1u64 << p) - 1)).count_ones() % 2 == 1 {
        -phase
    } else {
        phase
    };
    (sign, d | (1u64 << p))
}

/// `P|b⟩ = phase |b'⟩`.
fn pauli_action(s: &PauliString, b: u64) -> (C64, u64) {
    let (x, z) = (s.x_mask(), s.z_mask());
    let y = (x & z).count_ones();
    let mut phase = match y % 4 {
        0 => C64::new(1.0, 0.0),
        1 => C64::new(0.0, 1.0),
        2 => C64::new(-1.0, 0.0),
        _ => C64::new(0.0, -1.0),
    };
    if (b & z).count_ones() % 2 == 1 {
        phase = -phase;
    }
    (phase, b ^ x)
}

/// The lowest `n_roots` eigenvalues of `h` restricted to the span of the
/// basis states `basis` (which `h` must leave invariant).
pub(crate) fn sector_energies(h: &PauliSum, basis: &[u64], n_roots: usize) -> Result<Vec<f64>> {
    let dim = basis.len();
    if n_roots == 0 || n_roots > dim {
        return Err(QsoError::invalid(
            "n_roots",
            format!("need 1 to {dim} roots, got {n_roots}"),
        ));
    }
    let index: HashMap<u64, usize> = basis.iter().enumerate().map(|(i, &b)| (b, i)).collect();
    // Columns as sparse lists.
    let mut columns: Vec<Vec<(usize, f64)>> = Vec::with_capacity(dim);
    for &b in basis {
        // Single Pauli terms may leave the sector; their sum may not.
        let mut image: HashMap<u64, C64> = HashMap::new();
        for &(c, s) in h.terms() {
            let (phase, b2) = pauli_action(&s, b);
            *image.entry(b2).or_default() += c * phase;
        }
        let mut col: Vec<(usize, f64)> = Vec::new();
        for (b2, v) in image {
            match index.get(&b2) {
                Some(&row) => {
                    if v.im.abs() > 1e-10 {
                        return Err(QsoError::invalid(
                            "hamiltonian",
                            "complex matrix elements in the determinant basis",
                        ));
                    }
                    if v.re != 0.0 {
                        col.push((row, v.re));
                    }
                }
                None if v.norm() > 1e-10 => {
                    return Err(QsoError::invalid(
                        "hamiltonian",
                        "does not conserve the electron numbers of the sector",
                    ));
                }
                None => {}
            }
        }
        col.sort_by_key(|e| e.0);
        columns.push(col);
    }
    lowest_eigenvalues(columns, n_roots)
}

/// The lowest `n_roots` eigenvalues of the real symmetric matrix with
/// sparse `columns`.
fn lowest_eigenvalues(columns: Vec<Vec<(usize, f64)>>, n_roots: usize) -> Result<Vec<f64>> {
    let dim = columns.len();
    if n_roots == 0 || n_roots > dim {
        return Err(QsoError::invalid(
            "n_roots",
            format!("need 1 to {dim} roots, got {n_roots}"),
        ));
    }
    // A dense eigensolver costs O(dim³): above 1024 determinants a single
    // root comes from Lanczos.
    if dim <= DENSE_FCI_LIMIT && (n_roots > 1 || dim <= 1024) {
        let mut m = RMatrix::zeros(dim, dim);
        for (c, col) in columns.iter().enumerate() {
            for &(r, v) in col {
                m[(r, c)] = v;
            }
        }
        let (values, _) = eigh_real(&m);
        return Ok(values[..n_roots].to_vec());
    }
    if n_roots > 1 {
        return Err(QsoError::invalid(
            "n_roots",
            format!("only the ground state is available above {DENSE_FCI_LIMIT} determinants"),
        ));
    }
    let apply = |x: &[C64], y: &mut [C64]| {
        y.iter_mut().for_each(|v| *v = C64::new(0.0, 0.0));
        for (c, col) in columns.iter().enumerate() {
            for &(r, v) in col {
                y[r] += v * x[c];
            }
        }
    };
    let opts = KrylovOptions {
        tol: 1e-10,
        ..KrylovOptions::default()
    };
    Ok(vec![
        lanczos_ground_state(dim, apply, None, &opts)?.eigenvalue,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chemistry::mapping::Mapping;

    pub(crate) fn molecule(name: &str) -> MolecularIntegrals {
        let text = match name {
            "h2" => include_str!("../../examples/molecules/h2_sto3g.fcidump"),
            "lih" => include_str!("../../examples/molecules/lih_sto3g.fcidump"),
            "beh2" => include_str!("../../examples/molecules/beh2_sto3g.fcidump"),
            _ => unreachable!(),
        };
        MolecularIntegrals::from_fcidump(text).unwrap()
    }

    #[test]
    fn combinations_are_complete() {
        assert_eq!(binomial(7, 3), 35);
        assert_eq!(binomial(64, 32), 1_832_624_140_942_590_534);
        let big = MolecularIntegrals::from_fcidump(" &FCI NORB=32,NELEC=32 &END\n 1.0 1 1 0 0\n")
            .unwrap();
        assert!(big.fci_energies(1).is_err());
        assert_eq!(
            combinations(4, 2),
            vec![0b0011, 0b0101, 0b0110, 0b1001, 0b1010, 0b1100]
        );
        assert_eq!(combinations(3, 0), vec![0]);
        assert_eq!(combinations(7, 3).len(), 35);
    }

    #[test]
    fn hartree_fock_energies_match_pyscf() {
        // examples/molecules/reference.json
        for (name, hf) in [
            ("h2", -1.116998996754004),
            ("lih", -7.8620238601271195),
            ("beh2", -15.560334935987296),
        ] {
            let mol = molecule(name);
            assert!((mol.hartree_fock_energy() - hf).abs() < 1e-9, "{name}");
            // The HF circuit prepares a state with that energy.
            for encoding in [
                QubitEncoding::new(Mapping::JordanWigner),
                QubitEncoding::new(Mapping::BravyiKitaev),
                QubitEncoding::parity_reduced(),
            ] {
                if name == "beh2" && encoding.mapping != Mapping::Parity {
                    continue; // 14 qubits: covered by the reduced encoding
                }
                let h = mol.qubit_hamiltonian(encoding).unwrap();
                let c = mol.hartree_fock_circuit(encoding).unwrap();
                let psi = crate::quantum::StateVector::from_circuit(&c, &[]).unwrap();
                assert!(
                    (h.expectation(&psi).unwrap() - hf).abs() < 1e-9,
                    "{name} {encoding:?}"
                );
            }
        }
    }

    #[test]
    fn fcidump_round_trip_and_errors() {
        let mol = molecule("lih");
        let again = MolecularIntegrals::from_fcidump(&mol.to_fcidump()).unwrap();
        assert_eq!(again.n_orbitals(), 6);
        assert_eq!((again.n_alpha(), again.n_beta()), (2, 2));
        for p in 0..6 {
            for q in 0..6 {
                assert!((again.one_body(p, q) - mol.one_body(p, q)).abs() < 1e-14);
                for r in 0..6 {
                    for s in 0..6 {
                        assert!(
                            (again.two_body(p, q, r, s) - mol.two_body(p, q, r, s)).abs() < 1e-14
                        );
                    }
                }
            }
        }
        let header = " &FCI NORB=2,NELEC=2,MS2=0,\n ORBSYM=1,1,\n ISYM=1,\n &END\n";
        assert!(
            MolecularIntegrals::from_fcidump(&format!("{header} 0.5D0 1 1 0 0\n 1.0 0 0 0 0\n"))
                .is_ok()
        );
        for bad in [
            "no header".to_string(),
            " &FCI NORB=2,NELEC=2 \n 1.0 0 0 0 0\n".to_string(),
            format!("{header} 1.0 3 1 0 0\n"),
            format!("{header} nan 1 1 0 0\n"),
            format!("{header} 1.0 1 1 1\n"),
            format!("{header} 1.0 1 0 1 0\n"),
            " &FCI NORB=2,NELEC=5,MS2=0 &END\n".to_string(),
            " &FCI NORB=2,NELEC=2,MS2=1 &END\n".to_string(),
            " &FCI NORB=40,NELEC=2 &END\n".to_string(),
            " &FCI NORB=2,NELEC=2,IUHF=1 &END\n".to_string(),
            " &FCI NORB=2,NELEC=2,MS2=-9223372036854775808 &END\n".to_string(),
            " &FCI NORB=2,NELEC=2 &END 1.0 1 1\n".to_string(),
        ] {
            assert!(MolecularIntegrals::from_fcidump(&bad).is_err(), "{bad}");
        }
        assert!(MolecularIntegrals::new(1, 1, 0, 0.0, vec![1.0], vec![0.5]).is_ok());
        assert!(
            MolecularIntegrals::new(2, 1, 0, 0.0, vec![1.0, 0.1, 0.2, 1.0], vec![0.0; 16]).is_err()
        );
    }
}
