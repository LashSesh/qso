//! Hartree–Fock self-consistent field.
//!
//! [`rhf`] and [`uhf`] solve the restricted (closed-shell) and unrestricted
//! Roothaan–Hall equations `F C = S C ε` on [`AoIntegrals`]. The basis is
//! orthogonalised canonically (`X = U s^{−1/2}`, dropping overlap
//! eigenvalues below [`ScfOptions::linear_dependence`]). The iterations
//! start from a superposition of atomic densities ([`Guess::Atoms`]: each
//! atom's spherically averaged Hartree–Fock density in the same basis, as
//! PySCF's `atom` guess) and are accelerated by Pulay's DIIS on the
//! commutator `F D S − S D F` (Pulay, J. Comput. Chem. 3, 556 (1982)).
//! The calculation converges when the
//! energy changes by less than [`ScfOptions::energy_tolerance`] and the
//! largest entry of the orbital gradient `F_ia` (occupied `i`, virtual
//! `a`) is below [`ScfOptions::gradient_tolerance`]; a
//! calculation that does not converge is an error. If DIIS does not
//! converge, a second run without DIIS and with a level shift of 0.5 Ha
//! ([`ScfOptions::level_shift`]) follows.
//!
//! Like every Hartree–Fock code, the iterations find a stationary point
//! near the guess, which need not be the lowest one. The core-Hamiltonian
//! guess ([`Guess::Core`]) lands above the ground state for N₂ in STO-3G
//! by 0.73 Ha; the atomic guess finds PySCF's solutions in the tests.
//!
//! [`ScfResult::molecular_integrals`] transforms the integrals to the
//! restricted orbitals, optionally freezing core orbitals and keeping an
//! active space, for the qubit Hamiltonians of [`super::molecule`].
//!
//! ```
//! use metatron_qso::chemistry::integrals::{AoIntegrals, Basis, Molecule};
//! use metatron_qso::chemistry::scf::{ScfOptions, rhf};
//!
//! let h2 = Molecule::from_atoms("H 0 0 0; H 0 0 0.735", 0, 1).unwrap();
//! let ints = AoIntegrals::compute(&h2, Basis::Sto3g).unwrap();
//! let hf = rhf(&h2, &ints, &ScfOptions::default()).unwrap();
//! assert!((hf.energy - (-1.116998996754)).abs() < 1e-9);
//! let mo = hf.molecular_integrals(&ints, 0, None).unwrap();
//! assert!((mo.fci_energies(1).unwrap()[0] - (-1.137306035753)).abs() < 1e-9);
//! ```

use serde::{Deserialize, Serialize};

use super::integrals::{AoIntegrals, Molecule};
use super::molecule::MolecularIntegrals;
use crate::error::{QsoError, Result};
use crate::linalg::{RMatrix, eigh_real};

/// Settings of [`rhf`] and [`uhf`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScfOptions {
    /// Largest number of iterations.
    pub max_iterations: usize,
    /// Largest change of the energy (hartree) at convergence.
    pub energy_tolerance: f64,
    /// Largest entry of the orbital gradient `F_ia` (occupied `i`,
    /// virtual `a`, in the orbitals of the density).
    pub gradient_tolerance: f64,
    /// Number of Fock matrices DIIS extrapolates over (0 switches it off).
    pub diis_space: usize,
    /// Overlap eigenvalues below this are dropped as linearly dependent.
    pub linear_dependence: f64,
    /// Starting density.
    #[serde(default)]
    pub guess: Guess,
    /// Level shift (hartree) added to the virtual orbitals while the
    /// orbital gradient is above 10⁻³; it damps oscillations without
    /// changing the converged solution. 0 switches it off.
    #[serde(default)]
    pub level_shift: f64,
}

/// Starting density of [`rhf`] and [`uhf`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Guess {
    /// Superposition of spherically averaged atomic Hartree–Fock densities.
    #[default]
    Atoms,
    /// Orbitals of the core Hamiltonian, both spins filled with the average
    /// occupation `½(n_α + n_β)`.
    Core,
}

impl Default for ScfOptions {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            energy_tolerance: 1e-10,
            gradient_tolerance: 1e-8,
            diis_space: 8,
            linear_dependence: 1e-8,
            guess: Guess::Atoms,
            level_shift: 0.0,
        }
    }
}

impl ScfOptions {
    fn validate(&self) -> Result<()> {
        if self.max_iterations == 0 {
            return Err(QsoError::invalid("max_iterations", "must be positive"));
        }
        for (name, v) in [
            ("energy_tolerance", self.energy_tolerance),
            ("gradient_tolerance", self.gradient_tolerance),
            ("linear_dependence", self.linear_dependence),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(QsoError::invalid(name, "must be positive and finite"));
            }
        }
        if !(self.level_shift.is_finite() && self.level_shift >= 0.0) {
            return Err(QsoError::invalid(
                "level_shift",
                "must be non-negative and finite",
            ));
        }
        Ok(())
    }
}

/// A converged Hartree–Fock solution.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfResult {
    /// Total energy including the nuclear repulsion (hartree).
    pub energy: f64,
    /// Nuclear repulsion energy.
    pub nuclear_repulsion: f64,
    /// Orbital energies, ascending: one set for RHF, `[α, β]` for UHF.
    pub orbital_energies: Vec<Vec<f64>>,
    /// Orbital coefficients (columns, in the atomic-orbital basis): one
    /// matrix for RHF, `[α, β]` for UHF.
    pub coefficients: Vec<RMatrix>,
    pub n_alpha: usize,
    pub n_beta: usize,
    /// Iterations used.
    pub iterations: usize,
    /// Overlap matrix of the basis.
    overlap: RMatrix,
    /// The nuclei and basis of the integrals used.
    source: AoIntegralsId,
}

/// What identifies the integrals of a calculation.
#[derive(Clone, Debug, PartialEq)]
struct AoIntegralsId {
    basis: super::integrals::Basis,
    harmonics: super::integrals::Harmonics,
    n_functions: usize,
    molecule: Molecule,
}

impl ScfResult {
    /// Whether the solution is restricted (one set of orbitals).
    pub fn is_restricted(&self) -> bool {
        self.coefficients.len() == 1
    }

    /// `⟨S²⟩` of the determinant: `S(S+1)` for RHF; for UHF
    /// `S_z(S_z+1) + n_β − Σ_ij |⟨α_i|β_j⟩|²` (spin contamination makes it
    /// larger than `S(S+1)`).
    pub fn s_squared(&self) -> f64 {
        let sz = (self.n_alpha as f64 - self.n_beta as f64) / 2.0;
        if self.is_restricted() {
            return sz * (sz + 1.0);
        }
        let mut value = sz * (sz + 1.0) + self.n_beta as f64;
        let (ca, cb) = (&self.coefficients[0], self.coefficients.last().unwrap());
        let ca = ca.columns(0, self.n_alpha);
        let cb = cb.columns(0, self.n_beta);
        let overlap = ca.transpose() * &self.overlap * cb;
        value -= overlap.iter().map(|v| v * v).sum::<f64>();
        value
    }

    /// The Hamiltonian in the restricted orbitals with the lowest `frozen`
    /// orbitals doubly occupied and folded into the core energy, keeping
    /// `active` orbitals (all others when `None`). The result's
    /// Hartree–Fock determinant has the SCF energy.
    pub fn molecular_integrals(
        &self,
        ints: &AoIntegrals,
        frozen: usize,
        active: Option<usize>,
    ) -> Result<MolecularIntegrals> {
        if !self.is_restricted() {
            return Err(QsoError::invalid(
                "scf",
                "molecular integrals need restricted (closed-shell) orbitals; open shells are not supported",
            ));
        }
        if !(ints.basis() == self.source.basis
            && ints.harmonics() == self.source.harmonics
            && ints.n_functions() == self.source.n_functions
            && ints.belongs_to(&self.source.molecule))
        {
            return Err(QsoError::invalid(
                "ints",
                "the integrals are not those of this Hartree–Fock calculation",
            ));
        }
        let c = &self.coefficients[0];
        let n = ints.n_functions();
        let m_all = c.ncols();
        if frozen > self.n_beta {
            return Err(QsoError::invalid(
                "frozen",
                format!(
                    "{frozen} frozen orbitals exceed the {} doubly occupied ones",
                    self.n_beta
                ),
            ));
        }
        let active = active.unwrap_or(m_all - frozen);
        if active == 0 || frozen + active > m_all {
            return Err(QsoError::invalid(
                "active",
                format!(
                    "{active} active orbitals after {frozen} frozen exceed the {m_all} orbitals"
                ),
            ));
        }
        if frozen + active < self.n_alpha {
            return Err(QsoError::invalid(
                "active",
                format!(
                    "{active} active orbitals after {frozen} frozen cannot hold the {} occupied ones",
                    self.n_alpha
                ),
            ));
        }
        if active > super::molecule::MAX_ORBITALS {
            return Err(QsoError::invalid(
                "active",
                format!(
                    "an active space of {active} orbitals exceeds the maximum {} (choose `active`)",
                    super::molecule::MAX_ORBITALS
                ),
            ));
        }
        // h and (pq|rs) over the frozen and active orbitals.
        let m = frozen + active;
        let cm = c.columns(0, m).into_owned();
        let h_ao = RMatrix::from_fn(n, n, |i, j| ints.core(i, j));
        let h = cm.transpose() * &h_ao * &cm;
        let eri = transform_eri(ints, &cm);
        let g = |p: usize, q: usize, r: usize, s: usize| eri[((p * m + q) * m + r) * m + s];
        let mut core = self.nuclear_repulsion;
        for i in 0..frozen {
            core += 2.0 * h[(i, i)];
            for j in 0..frozen {
                core += 2.0 * g(i, i, j, j) - g(i, j, j, i);
            }
        }
        let mut one = vec![0.0; active * active];
        for p in 0..active {
            for q in 0..active {
                let mut v = h[(frozen + p, frozen + q)];
                for i in 0..frozen {
                    v += 2.0 * g(frozen + p, frozen + q, i, i) - g(frozen + p, i, i, frozen + q);
                }
                one[p * active + q] = v;
            }
        }
        let mut two = vec![0.0; active.pow(4)];
        for p in 0..active {
            for q in 0..active {
                for r in 0..active {
                    for s in 0..active {
                        two[((p * active + q) * active + r) * active + s] =
                            g(frozen + p, frozen + q, frozen + r, frozen + s);
                    }
                }
            }
        }
        MolecularIntegrals::new(
            active,
            self.n_alpha - frozen,
            self.n_beta - frozen,
            core,
            one,
            two,
        )
    }
}

/// `(pq|rs)` over the columns of `c`, by four quarter transformations.
fn transform_eri(ints: &AoIntegrals, c: &RMatrix) -> Vec<f64> {
    let (n, m) = (ints.n_functions(), c.ncols());
    let ao = ints.eri_tensor();
    // Transform the last index, then rotate the tensor so the next index
    // is last: (ijk l) → (l ijk). Four steps return the original order.
    let mut cur = ao.to_vec();
    let mut dims = [n, n, n, n];
    for _ in 0..4 {
        let rest = dims[0] * dims[1] * dims[2];
        let last = dims[3];
        let mut next = vec![0.0; m * rest];
        for a in 0..rest {
            let row = &cur[a * last..(a + 1) * last];
            for p in 0..m {
                let mut v = 0.0;
                for (k, x) in row.iter().enumerate() {
                    v += x * c[(k, p)];
                }
                next[p * rest + a] = v;
            }
        }
        cur = next;
        dims = [m, dims[0], dims[1], dims[2]];
    }
    cur
}

fn validate_inputs(molecule: &Molecule, ints: &AoIntegrals) -> Result<()> {
    if !ints.belongs_to(molecule) {
        return Err(QsoError::invalid(
            "ints",
            "the integrals belong to another molecule or geometry",
        ));
    }
    let n = ints.n_functions();
    let (na, nb) = molecule.n_alpha_beta();
    if na > n || nb > n {
        return Err(QsoError::invalid(
            "molecule",
            format!("{na} α and {nb} β electrons do not fit into {n} basis functions"),
        ));
    }
    Ok(())
}

/// Canonical orthogonalisation `X = U s^{−1/2}`.
fn orthogonaliser(s: &RMatrix, threshold: f64) -> Result<RMatrix> {
    let (values, vectors) = eigh_real(s);
    let keep: Vec<usize> = (0..values.len())
        .filter(|&i| values[i] > threshold)
        .collect();
    if keep.is_empty() {
        return Err(QsoError::invalid("basis", "the overlap matrix is singular"));
    }
    Ok(RMatrix::from_fn(s.nrows(), keep.len(), |r, c| {
        vectors[(r, keep[c])] / values[keep[c]].sqrt()
    }))
}

/// Coulomb and exchange matrices of the density `d`.
fn coulomb_exchange(ints: &AoIntegrals, d: &RMatrix) -> (RMatrix, RMatrix) {
    let n = ints.n_functions();
    let eri = ints.eri_tensor();
    let mut j = RMatrix::zeros(n, n);
    let mut k = RMatrix::zeros(n, n);
    for p in 0..n {
        for q in 0..=p {
            let mut vj = 0.0;
            let mut vk = 0.0;
            for r in 0..n {
                let base_j = ((p * n + q) * n + r) * n;
                let base_k = ((p * n + r) * n + q) * n;
                for s in 0..n {
                    let drs = d[(r, s)];
                    vj += eri[base_j + s] * drs;
                    vk += eri[base_k + s] * drs;
                }
            }
            j[(p, q)] = vj;
            j[(q, p)] = vj;
            k[(p, q)] = vk;
            k[(q, p)] = vk;
        }
    }
    (j, k)
}

/// Orbitals and energies of `f` in the orthogonal basis `x`.
fn diagonalise(f: &RMatrix, x: &RMatrix) -> (Vec<f64>, RMatrix) {
    let fo = x.transpose() * f * x;
    let fo = (&fo + fo.transpose()) * 0.5;
    let (e, v) = eigh_real(&fo);
    (e, x * v)
}

fn density(c: &RMatrix, occupied: usize) -> RMatrix {
    let co = c.columns(0, occupied);
    co * co.transpose()
}

/// DIIS extrapolation over stored Fock matrices and error vectors.
struct Diis {
    space: usize,
    focks: Vec<Vec<RMatrix>>,
    errors: Vec<Vec<f64>>,
}

impl Diis {
    fn new(space: usize) -> Self {
        Self {
            space,
            focks: Vec::new(),
            errors: Vec::new(),
        }
    }

    fn extrapolate(&mut self, focks: Vec<RMatrix>, error: Vec<f64>) -> Vec<RMatrix> {
        if self.space == 0 {
            return focks;
        }
        self.focks.push(focks);
        self.errors.push(error);
        if self.focks.len() > self.space {
            self.focks.remove(0);
            self.errors.remove(0);
        }
        let k = self.focks.len();
        if k < 2 {
            return self.focks[k - 1].clone();
        }
        // [B −1; −1 0] [c; λ] = [0; −1].
        let mut b = RMatrix::zeros(k + 1, k + 1);
        for i in 0..k {
            for j in 0..=i {
                let v: f64 = self.errors[i]
                    .iter()
                    .zip(&self.errors[j])
                    .map(|(a, b)| a * b)
                    .sum();
                b[(i, j)] = v;
                b[(j, i)] = v;
            }
            b[(i, k)] = -1.0;
            b[(k, i)] = -1.0;
        }
        // Scale for conditioning.
        let scale = (0..k).map(|i| b[(i, i)]).fold(0.0, f64::max);
        if scale > 0.0 {
            for i in 0..k {
                for j in 0..k {
                    b[(i, j)] /= scale;
                }
            }
        }
        let mut rhs = nalgebra::DVector::zeros(k + 1);
        rhs[k] = -1.0;
        let Some(c) = b.clone().lu().solve(&rhs) else {
            // Singular: restart from the newest matrices.
            let last = self.focks.pop().unwrap();
            let err = self.errors.pop().unwrap();
            self.focks.clear();
            self.errors.clear();
            self.focks.push(last.clone());
            self.errors.push(err);
            return last;
        };
        let spins = self.focks[0].len();
        (0..spins)
            .map(|s| {
                let mut f = self.focks[0][s].clone() * c[0];
                for (i, fi) in self.focks.iter().enumerate().skip(1) {
                    f += &fi[s] * c[i];
                }
                f
            })
            .collect()
    }
}

/// `X^T (F D S − S D F) X`, flattened.
fn commutator(f: &RMatrix, d: &RMatrix, s: &RMatrix, x: &RMatrix) -> Vec<f64> {
    let fds = f * d * s;
    let e = &fds - fds.transpose();
    (x.transpose() * e * x).iter().copied().collect()
}

/// Restricted (closed-shell) Hartree–Fock. Needs a singlet.
pub fn rhf(molecule: &Molecule, ints: &AoIntegrals, options: &ScfOptions) -> Result<ScfResult> {
    options.validate()?;
    validate_inputs(molecule, ints)?;
    let (na, nb) = molecule.n_alpha_beta();
    if na != nb {
        return Err(QsoError::invalid(
            "multiplicity",
            "restricted Hartree–Fock needs a closed shell; use unrestricted Hartree–Fock",
        ));
    }
    with_fallback(molecule, ints, options, Kind::Restricted)
}

/// Unrestricted Hartree–Fock: separate α and β orbitals.
pub fn uhf(molecule: &Molecule, ints: &AoIntegrals, options: &ScfOptions) -> Result<ScfResult> {
    options.validate()?;
    validate_inputs(molecule, ints)?;
    with_fallback(molecule, ints, options, Kind::Unrestricted)
}

/// Restricted open-shell Hartree–Fock: one set of orbitals, the lowest
/// `n_β` doubly occupied and the next `n_α − n_β` singly (α). The
/// orbitals diagonalise Roothaan's effective Fock matrix with PySCF's
/// blocks: `F_c = ½(F_α + F_β)` within the closed, open and virtual
/// spaces and between closed and virtual orbitals, `F_β` between closed
/// and open, `F_α` between open and virtual. The energy does not depend on
/// this choice; the orbital energies do. For a closed shell it is RHF.
pub fn rohf(molecule: &Molecule, ints: &AoIntegrals, options: &ScfOptions) -> Result<ScfResult> {
    options.validate()?;
    validate_inputs(molecule, ints)?;
    with_fallback(molecule, ints, options, Kind::RestrictedOpen)
}

/// The kind of determinant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Restricted,
    Unrestricted,
    RestrictedOpen,
}

/// The SCF with `options`; if DIIS does not converge, once more without
/// DIIS and with a level shift of 0.5 Ha (slower, but it settles
/// oscillating cases such as CN in STO-3G).
fn with_fallback(
    molecule: &Molecule,
    ints: &AoIntegrals,
    options: &ScfOptions,
    kind: Kind,
) -> Result<ScfResult> {
    match scf(molecule, ints, options, kind) {
        Err(e @ QsoError::NoConvergence { .. })
            if options.diis_space > 0 && options.level_shift == 0.0 =>
        {
            let shifted = ScfOptions {
                diis_space: 0,
                level_shift: 0.5,
                ..options.clone()
            };
            scf(molecule, ints, &shifted, kind).map_err(|_| e)
        }
        other => other,
    }
}

fn scf(
    molecule: &Molecule,
    ints: &AoIntegrals,
    options: &ScfOptions,
    kind: Kind,
) -> Result<ScfResult> {
    let restricted = kind == Kind::Restricted;
    let open = kind == Kind::RestrictedOpen;
    let n = ints.n_functions();
    let (na, nb) = molecule.n_alpha_beta();
    let s = RMatrix::from_fn(n, n, |i, j| ints.overlap(i, j));
    let h = RMatrix::from_fn(n, n, |i, j| ints.core(i, j));
    let x = orthogonaliser(&s, options.linear_dependence)?;
    if na > x.ncols() {
        return Err(QsoError::invalid(
            "basis",
            format!(
                "{na} electrons of one spin exceed {} independent functions",
                x.ncols()
            ),
        ));
    }
    let e_nuc = molecule.nuclear_repulsion();
    // Sets of orbitals.
    let spins = if kind == Kind::Unrestricted { 2 } else { 1 };
    let occupied = [na, nb];
    // Both spins start from half the guessed total density.
    let start = match options.guess {
        Guess::Atoms => atomic_guess(ints)? * 0.5,
        Guess::Core => {
            let (_, c) = diagonalise(&h, &x);
            RMatrix::from_fn(n, c.ncols(), |r, k| {
                let occ = ((k < na) as u8 + (k < nb) as u8) as f64 / 2.0;
                c[(r, k)] * occ
            }) * c.transpose()
        }
    };
    let mut orbitals: Vec<(Vec<f64>, RMatrix)> = Vec::new();
    let mut diis = Diis::new(options.diis_space);
    let mut previous = f64::INFINITY;
    let mut last_residual = f64::INFINITY;
    for iteration in 1..=options.max_iterations {
        let d: Vec<RMatrix> = if iteration == 1 {
            vec![start.clone(), start.clone()]
        } else {
            (0..2)
                .map(|sp| density(&orbitals[sp.min(spins - 1)].1, occupied[sp]))
                .collect()
        };
        let focks = fock_matrices(ints, &h, &d, restricted);
        // E = ½ Σ_σ Tr[D_σ (H + F_σ)].
        let mut energy = e_nuc;
        for sp in 0..2 {
            let f = &focks[sp.min(focks.len() - 1)];
            energy += 0.5 * (d[sp].component_mul(&(&h + f))).sum();
        }
        if !energy.is_finite() {
            break;
        }
        if iteration == 1 {
            // The guess is no determinant: no gradient, no DIIS.
            orbitals = if open {
                vec![diagonalise(&((&focks[0] + &focks[1]) * 0.5), &x)]
            } else {
                focks.iter().map(|f| diagonalise(f, &x)).collect()
            };
            previous = energy;
            continue;
        }
        // Convergence: the orbital gradient F_ia in the orbitals of the
        // density. DIIS: the commutator F D S − S D F in the atomic-orbital
        // basis, which stays the same from one iteration to the next.
        let mut gradient = Vec::new();
        let mut error = Vec::new();
        // ROHF: the gradient between closed, open and virtual orbitals,
        // and the effective Fock matrix's commutators with both densities.
        let effective = open.then(|| roothaan_fock(&focks, &d, &s));
        if let Some(f) = &effective {
            let c = &orbitals[0].1;
            let fa = c.transpose() * &focks[0] * c;
            let fb = c.transpose() * &focks[1] * c;
            for i in 0..c.ncols() {
                for j in i + 1..c.ncols() {
                    let g = match (i < nb, i < na, j < nb, j < na) {
                        // closed–open: F_β; closed–virtual: F_α + F_β.
                        (true, _, false, true) => fb[(i, j)],
                        (true, _, false, false) => fa[(i, j)] + fb[(i, j)],
                        // open–virtual: F_α.
                        (false, true, false, false) => fa[(i, j)],
                        _ => continue,
                    };
                    gradient.push(g);
                }
            }
            for dm in &d {
                let fds = f * dm * &s;
                error.extend((&fds - fds.transpose()).iter().copied());
            }
        }
        for sp in 0..if open { 0 } else { spins } {
            let c = &orbitals[sp].1;
            let occ = occupied[sp];
            let fmo = c.transpose() * &focks[sp] * c;
            for i in 0..occ {
                for a in occ..c.ncols() {
                    gradient.push(fmo[(i, a)]);
                }
            }
            let fds = &focks[sp] * &d[sp] * &s;
            error.extend((&fds - fds.transpose()).iter().copied());
        }
        let residual = gradient.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        last_residual = residual;
        if (energy - previous).abs() < options.energy_tolerance
            && residual < options.gradient_tolerance
        {
            // Orbitals and energies of the converged Fock matrix itself.
            let last: Vec<(Vec<f64>, RMatrix)> = match &effective {
                Some(f) => vec![diagonalise(f, &x)],
                None => focks.iter().map(|f| diagonalise(f, &x)).collect(),
            };
            let orbital_energies = last.iter().map(|o| o.0.clone()).collect();
            let coefficients = last.into_iter().map(|o| o.1).collect();
            return Ok(ScfResult {
                energy,
                nuclear_repulsion: e_nuc,
                orbital_energies,
                coefficients,
                n_alpha: na,
                n_beta: nb,
                iterations: iteration,
                overlap: s,
                source: AoIntegralsId {
                    basis: ints.basis(),
                    harmonics: ints.harmonics(),
                    n_functions: n,
                    molecule: molecule.clone(),
                },
            });
        }
        previous = energy;
        let mut extrapolated = diis.extrapolate(effective.map_or(focks, |f| vec![f]), error);
        if options.level_shift > 0.0 && residual > 1e-3 {
            // F + σ (S − S D S) raises the virtual orbitals by σ (ROHF:
            // those empty in both spins).
            for (sp, f) in extrapolated.iter_mut().enumerate() {
                let sds = &s * &d[sp] * &s;
                *f += (&s - sds) * options.level_shift;
            }
        }
        orbitals = extrapolated.iter().map(|f| diagonalise(f, &x)).collect();
    }
    Err(QsoError::NoConvergence {
        method: match kind {
            Kind::Restricted => "RHF",
            Kind::Unrestricted => "UHF",
            Kind::RestrictedOpen => "ROHF",
        },
        iterations: options.max_iterations,
        residual: last_residual,
    })
}

/// Roothaan's effective Fock matrix of a restricted open shell (PySCF's
/// blocks, see [`rohf`]) from `[F_α, F_β]` and `[D_α, D_β]`.
fn roothaan_fock(f: &[RMatrix], d: &[RMatrix], s: &RMatrix) -> RMatrix {
    let n = s.nrows();
    let fc = (&f[0] + &f[1]) * 0.5;
    let pc = &d[1] * s;
    let po = (&d[0] - &d[1]) * s;
    let pv = RMatrix::identity(n, n) - &d[0] * s;
    let mut g = (pc.transpose() * &fc * &pc) * 0.5
        + (po.transpose() * &fc * &po) * 0.5
        + (pv.transpose() * &fc * &pv) * 0.5
        + po.transpose() * &f[1] * &pc
        + po.transpose() * &f[0] * &pv
        + pv.transpose() * &fc * &pc;
    g = &g + g.transpose();
    g
}

/// Fock matrices `H + J(D_α + D_β) − K(D_σ)`: one for restricted
/// densities (equal spins), `[α, β]` otherwise.
fn fock_matrices(ints: &AoIntegrals, h: &RMatrix, d: &[RMatrix], restricted: bool) -> Vec<RMatrix> {
    if restricted {
        let (j, k) = coulomb_exchange(ints, &d[0]);
        vec![h + j * 2.0 - k]
    } else {
        let (ja, ka) = coulomb_exchange(ints, &d[0]);
        let (jb, kb) = coulomb_exchange(ints, &d[1]);
        let j = ja + jb;
        vec![h + &j - ka, h + j - kb]
    }
}

/// Superposition of atomic densities: each atom's spherically averaged
/// restricted Hartree–Fock density (fractional occupations shared equally
/// over degenerate orbitals), in the blocks of its basis functions.
fn atomic_guess(ints: &AoIntegrals) -> Result<RMatrix> {
    let n = ints.n_functions();
    let atoms = ints.atoms();
    let charges = ints.charges();
    let mut d = RMatrix::zeros(n, n);
    let mut cache: Vec<Option<RMatrix>> = vec![None; super::basis_data::ELEMENTS + 1];
    let mut start = 0;
    while start < n {
        let atom = atoms[start];
        let mut end = start;
        while end < n && atoms[end] == atom {
            end += 1;
        }
        let z = charges[atom] as usize;
        if cache[z].is_none() {
            cache[z] = Some(atomic_density(z as u32, ints.basis(), ints.harmonics())?);
        }
        let block = cache[z].as_ref().expect("cached");
        debug_assert_eq!(block.nrows(), end - start);
        d.view_mut((start, start), (end - start, end - start))
            .copy_from(block);
        start = end;
    }
    Ok(d)
}

/// Total density of an atom with spherically averaged occupations.
fn atomic_density(
    z: u32,
    basis: super::integrals::Basis,
    harmonics: super::integrals::Harmonics,
) -> Result<RMatrix> {
    let multiplicity = if z % 2 == 1 { 2 } else { 1 };
    let atom = Molecule::new(vec![z], vec![[0.0; 3]], 0, multiplicity)?;
    let ints = AoIntegrals::compute_with(&atom, basis, harmonics)?;
    let n = ints.n_functions();
    let s = RMatrix::from_fn(n, n, |i, j| ints.overlap(i, j));
    let h = RMatrix::from_fn(n, n, |i, j| ints.core(i, j));
    let x = orthogonaliser(&s, 1e-8)?;
    let (mut eps, mut c) = diagonalise(&h, &x);
    let mut diis = Diis::new(8);
    let mut d = RMatrix::zeros(n, n);
    let mut previous = f64::INFINITY;
    for _ in 0..100 {
        let occ = fractional_occupations(&eps, z as f64);
        d = RMatrix::from_fn(n, c.ncols(), |r, k| c[(r, k)] * occ[k]) * c.transpose();
        let half = &d * 0.5;
        let (j, k) = coulomb_exchange(&ints, &half);
        let f = &h + j * 2.0 - k;
        let energy = 0.5 * (half.component_mul(&(&h + &f))).sum() * 2.0;
        let error = commutator(&f, &half, &s, &x);
        let residual = error.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if (energy - previous).abs() < 1e-10 && residual < 1e-6 {
            break;
        }
        previous = energy;
        let f = diis.extrapolate(vec![f], error).remove(0);
        (eps, c) = diagonalise(&f, &x);
    }
    Ok(d)
}

/// Occupations (0 to 2) of orbitals with energies `eps` (ascending) for
/// `electrons`, shared equally within groups of degenerate orbitals.
fn fractional_occupations(eps: &[f64], electrons: f64) -> Vec<f64> {
    let mut occ = vec![0.0; eps.len()];
    let mut left = electrons;
    let mut k = 0;
    while k < eps.len() && left > 0.0 {
        let mut end = k + 1;
        while end < eps.len() && (eps[end] - eps[k]).abs() < 1e-6 * eps[k].abs().max(1.0) {
            end += 1;
        }
        let size = (end - k) as f64;
        let put = left.min(2.0 * size);
        for o in &mut occ[k..end] {
            *o = put / size;
        }
        left -= put;
        k = end;
    }
    occ
}

#[cfg(test)]
mod tests;
