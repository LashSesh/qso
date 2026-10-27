//! Hartree–Fock self-consistent field.
//!
//! [`rhf`] and [`uhf`] solve the restricted (closed-shell) and unrestricted
//! Roothaan–Hall equations `F C = S C ε` on [`AoIntegrals`]. The basis is
//! orthogonalised canonically (`X = U s^{−1/2}`, dropping overlap
//! eigenvalues below [`ScfOptions::linear_dependence`]); the iterations
//! start from the core Hamiltonian (its orbitals filled with the average
//! occupation `½(n_α + n_β)` for both spins) and are accelerated by Pulay's DIIS on
//! the commutator `F D S − S D F` (Pulay, J. Comput. Chem. 3, 556 (1982)).
//! The calculation converges when the energy changes by less than
//! [`ScfOptions::energy_tolerance`] and the largest commutator entry is
//! below [`ScfOptions::gradient_tolerance`]; a calculation that does not
//! converge is an error.
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
    /// Largest entry of the orbital gradient `X^T (F D S − S D F) X`.
    pub gradient_tolerance: f64,
    /// Number of Fock matrices DIIS extrapolates over (0 switches it off).
    pub diis_space: usize,
    /// Overlap eigenvalues below this are dropped as linearly dependent.
    pub linear_dependence: f64,
}

impl Default for ScfOptions {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            energy_tolerance: 1e-10,
            gradient_tolerance: 1e-8,
            diis_space: 8,
            linear_dependence: 1e-8,
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
                "molecular integrals need restricted orbitals (use rhf)",
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
        if active == 0 || frozen + active > m_all || frozen + active < self.n_alpha {
            return Err(QsoError::invalid(
                "active",
                format!(
                    "{active} active orbitals after {frozen} frozen must hold the {} occupied ones within {m_all}",
                    self.n_alpha
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

/// Orbital gradient `X^T (F D S − S D F) X`, flattened.
fn gradient(f: &RMatrix, d: &RMatrix, s: &RMatrix, x: &RMatrix) -> Vec<f64> {
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
            "restricted Hartree–Fock needs a closed shell (use uhf)",
        ));
    }
    scf(molecule, ints, options, true)
}

/// Unrestricted Hartree–Fock: separate α and β orbitals.
pub fn uhf(molecule: &Molecule, ints: &AoIntegrals, options: &ScfOptions) -> Result<ScfResult> {
    options.validate()?;
    validate_inputs(molecule, ints)?;
    scf(molecule, ints, options, false)
}

fn scf(
    molecule: &Molecule,
    ints: &AoIntegrals,
    options: &ScfOptions,
    restricted: bool,
) -> Result<ScfResult> {
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
    let spins = if restricted { 1 } else { 2 };
    let occupied = [na, nb];
    let (e0, c0) = diagonalise(&h, &x);
    let mut orbitals = vec![(e0, c0); spins];
    let mut diis = Diis::new(options.diis_space);
    let mut previous = f64::INFINITY;
    let mut last_residual = f64::INFINITY;
    for iteration in 1..=options.max_iterations {
        // Densities per spin. The first iteration gives both spins the
        // core orbitals with the average occupation ½(n_α + n_β) (as
        // PySCF does): starting from the aufbau occupation per spin can
        // trap UHF in an excited state (OH in 6-31G does).
        let d: Vec<RMatrix> = if iteration == 1 {
            let c = &orbitals[0].1;
            let average = RMatrix::from_fn(n, c.ncols(), |r, k| {
                let occ = ((k < na) as u8 + (k < nb) as u8) as f64 / 2.0;
                c[(r, k)] * occ
            }) * c.transpose();
            vec![average.clone(), average]
        } else {
            (0..2)
                .map(|sp| density(&orbitals[sp.min(spins - 1)].1, occupied[sp]))
                .collect()
        };
        let total = &d[0] + &d[1];
        let (j, k0) = coulomb_exchange(ints, &d[0]);
        let ks: Vec<RMatrix> = if restricted {
            vec![k0]
        } else {
            vec![k0, coulomb_exchange(ints, &d[1]).1]
        };
        let j = if restricted {
            j * 2.0
        } else {
            j + coulomb_exchange(ints, &d[1]).0
        };
        let focks: Vec<RMatrix> = ks.iter().map(|k| &h + &j - k).collect();
        // E = ½ Σ_σ Tr[D_σ (H + F_σ)].
        let mut energy = e_nuc;
        for sp in 0..2 {
            let f = &focks[sp.min(spins - 1)];
            energy += 0.5 * (d[sp].component_mul(&(&h + f))).sum();
        }
        let mut error = Vec::new();
        for sp in 0..spins {
            let dsp = if restricted {
                &total * 0.5
            } else {
                d[sp].clone()
            };
            error.extend(gradient(&focks[sp], &dsp, &s, &x));
        }
        let residual = error.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        last_residual = residual;
        if !energy.is_finite() {
            break;
        }
        if (energy - previous).abs() < options.energy_tolerance
            && residual < options.gradient_tolerance
        {
            let orbital_energies = orbitals.iter().map(|o| o.0.clone()).collect();
            let coefficients = orbitals.into_iter().map(|o| o.1).collect();
            return Ok(ScfResult {
                energy,
                nuclear_repulsion: e_nuc,
                orbital_energies,
                coefficients,
                n_alpha: na,
                n_beta: nb,
                iterations: iteration,
                overlap: s,
            });
        }
        previous = energy;
        let extrapolated = diis.extrapolate(focks, error);
        orbitals = extrapolated.iter().map(|f| diagonalise(f, &x)).collect();
    }
    Err(QsoError::NoConvergence {
        method: if restricted { "RHF" } else { "UHF" },
        iterations: options.max_iterations,
        residual: last_residual,
    })
}

#[cfg(test)]
mod tests;
