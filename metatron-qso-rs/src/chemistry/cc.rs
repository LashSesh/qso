//! Correlation beyond Hartree–Fock: MP2, CCSD and CCSD(T).
//!
//! The references of 1.62 for molecules beyond the reach of full CI, in
//! the orbitals of a converged [`ScfResult`], with the lowest `frozen`
//! orbitals kept doubly occupied (a frozen core):
//!
//! * [`mp2`]: second-order Møller–Plesset energies in canonical RHF or UHF
//!   orbitals, with their same-spin and opposite-spin parts.
//! * [`ccsd`]: coupled-cluster singles and doubles (Purvis and Bartlett,
//!   J. Chem. Phys. 76, 1910 (1982)). A closed shell in restricted orbitals
//!   is solved in spatial orbitals; any other reference (UHF, ROHF) in spin
//!   orbitals with the equations of Stanton, Gauss, Watts and Bartlett
//!   (J. Chem. Phys. 94, 4334 (1991)), which take the whole Fock matrix and
//!   so hold for the non-canonical orbitals of ROHF too. The amplitudes are
//!   iterated by Jacobi steps from the MP2 amplitudes, accelerated by DIIS
//!   on the amplitudes.
//! * [`ccsd_t`]: CCSD and the perturbative triples of Raghavachari,
//!   Trucks, Pople and Head-Gordon (Chem. Phys. Lett. 157, 479 (1989)), for
//!   canonical references: in spatial orbitals for closed shells (the form
//!   of Rendell, Lee and Komornicki, Chem. Phys. Lett. 178, 462 (1991)), in
//!   spin orbitals for UHF.
//!
//! The integrals are transformed to the molecular orbitals in full (four
//! quarter transformations, each a matrix product), so a closed shell in
//! 64 basis functions holds 64⁴ of them; spin orbitals hold their whole
//! antisymmetrised tensor, up to [`MAX_SPIN_ORBITALS`].
//!
//! ```
//! use metatron_qso::chemistry::cc::{CcOptions, ccsd_t, mp2};
//! use metatron_qso::chemistry::integrals::{AoIntegrals, Basis, Molecule};
//! use metatron_qso::chemistry::scf::{ScfOptions, rhf};
//!
//! let water = Molecule::from_atoms("O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", 0, 1)
//!     .unwrap();
//! let ints = AoIntegrals::compute(&water, Basis::Sto3g).unwrap();
//! let hf = rhf(&water, &ints, &ScfOptions::default()).unwrap();
//! let pt = mp2(&hf, &ints, 0).unwrap();
//! assert!((pt.correlation - -0.0355456516).abs() < 1e-8);
//! let cc = ccsd_t(&hf, &ints, 0, &CcOptions::default()).unwrap();
//! assert!((cc.ccsd.correlation - -0.049438563).abs() < 1e-8);
//! ```

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use super::integrals::AoIntegrals;
use super::scf::ScfResult;
use crate::error::{QsoError, Result};
use crate::linalg::RMatrix;

/// Most spin orbitals (correlated, both spins) of a spin-orbital
/// calculation: their antisymmetrised integrals take `n⁴` doubles (800 MB
/// at the limit).
pub const MAX_SPIN_ORBITALS: usize = 100;

/// Settings of [`ccsd`] and [`ccsd_t`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CcOptions {
    /// Largest number of iterations.
    pub max_iterations: usize,
    /// Largest change of the correlation energy (hartree) at convergence.
    pub energy_tolerance: f64,
    /// Largest norm of the change of the amplitudes at convergence.
    pub amplitude_tolerance: f64,
    /// Amplitude vectors DIIS extrapolates over (0 switches it off).
    pub diis_space: usize,
}

impl Default for CcOptions {
    fn default() -> Self {
        CcOptions {
            max_iterations: 200,
            energy_tolerance: 1e-10,
            amplitude_tolerance: 1e-8,
            diis_space: 8,
        }
    }
}

impl CcOptions {
    fn validate(&self) -> Result<()> {
        if self.max_iterations == 0 {
            return Err(QsoError::invalid("max_iterations", "must be positive"));
        }
        for (name, v) in [
            ("energy_tolerance", self.energy_tolerance),
            ("amplitude_tolerance", self.amplitude_tolerance),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(QsoError::invalid(name, "must be positive and finite"));
            }
        }
        Ok(())
    }
}

/// Second-order Møller–Plesset energies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Mp2 {
    /// Correlation energy (hartree).
    pub correlation: f64,
    /// Hartree–Fock plus correlation.
    pub energy: f64,
    /// Its part from pairs of electrons of equal spin.
    pub same_spin: f64,
    /// Its part from pairs of opposite spin.
    pub opposite_spin: f64,
}

/// A converged CCSD calculation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Ccsd {
    /// Correlation energy (hartree).
    pub correlation: f64,
    /// Hartree–Fock plus correlation.
    pub energy: f64,
    /// Iterations used.
    pub iterations: usize,
    /// The T1 diagnostic of Lee and Taylor, `√(Σ (t_i^a)² / N)` over the
    /// spatial single amplitudes of a closed shell with `N` correlated
    /// electrons; in spin orbitals `√(Σ (t_i^a)² / 2N)`, the same for a
    /// closed shell.
    pub t1_diagnostic: f64,
    /// Whether the equations were solved in spin orbitals (open shells),
    /// not spatial ones.
    pub spin_orbitals: bool,
}

/// A converged CCSD(T) calculation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CcsdT {
    pub ccsd: Ccsd,
    /// The perturbative triples correction (hartree).
    pub triples: f64,
    /// Hartree–Fock plus CCSD plus triples.
    pub energy: f64,
}

// ---- orbitals, integrals and Fock matrices --------------------------------

/// `(pq|rs)` over the columns of `c[0]` (p), `c[1]` (q), `c[2]` (r) and
/// `c[3]` (s), row-major, by four quarter transformations, each a matrix
/// product: the last index is transformed and moved to the front.
fn transform(ints: &AoIntegrals, c: [&RMatrix; 4]) -> Vec<f64> {
    let n = ints.n_functions();
    let mut cur = ints.eri_tensor().to_vec();
    let mut dims = [n, n, n, n];
    for k in (0..4).rev() {
        let m = c[k].ncols();
        let rest = dims[0] * dims[1] * dims[2];
        let last = dims[3];
        // The row-major (rest × last) tensor is the column-major (last ×
        // rest) matrix W; Wᵀ C is column-major (rest × m): index p·rest + a.
        let w = DMatrix::from_column_slice(last, rest, &cur);
        cur = w.tr_mul(c[k]).as_slice().to_vec();
        dims = [m, dims[0], dims[1], dims[2]];
    }
    cur
}

/// `h + J(d_total) − K(d_spin)` in the atomic orbitals.
fn fock_ao(ints: &AoIntegrals, d_total: &RMatrix, d_spin: &RMatrix) -> RMatrix {
    let n = ints.n_functions();
    let eri = ints.eri_tensor();
    let mut f = RMatrix::from_fn(n, n, |i, j| ints.core(i, j));
    for mu in 0..n {
        for nu in 0..n {
            let mut v = 0.0;
            for la in 0..n {
                for si in 0..n {
                    let base = ((mu * n + nu) * n + la) * n + si;
                    v += eri[base] * d_total[(la, si)];
                    v -= eri[((mu * n + la) * n + nu) * n + si] * d_spin[(la, si)];
                }
            }
            f[(mu, nu)] += v;
        }
    }
    f
}

/// The orbitals a correlated calculation works in: per spin the active
/// orbitals (all but the `frozen` lowest), their occupied ones first, and
/// their Fock matrix.
struct Reference {
    /// Active orbitals of each spin (as many for both).
    m: usize,
    /// Active occupied orbitals of each spin.
    occ: [usize; 2],
    coeff: [RMatrix; 2],
    fock: [RMatrix; 2],
    /// One set of orbitals for both spins (RHF, ROHF).
    restricted: bool,
    /// Hartree–Fock energy.
    energy: f64,
}

impl Reference {
    fn new(hf: &ScfResult, ints: &AoIntegrals, frozen: usize) -> Result<Self> {
        hf.check_integrals(ints)?;
        if frozen > hf.n_beta {
            return Err(QsoError::invalid(
                "frozen",
                format!(
                    "{frozen} frozen orbitals exceed the {} doubly occupied ones",
                    hf.n_beta
                ),
            ));
        }
        let restricted = hf.is_restricted();
        let c = |s: usize| &hf.coefficients[if restricted { 0 } else { s }];
        let all = c(0).ncols();
        let density = |s: usize, n: usize| {
            let occ = c(s).columns(0, n);
            occ * occ.transpose()
        };
        let (da, db) = (density(0, hf.n_alpha), density(1, hf.n_beta));
        let total = &da + &db;
        let m = all - frozen;
        let coeff = [0, 1].map(|s| c(s).columns(frozen, m).into_owned());
        let fock = [(0, &da), (1, &db)].map(|(s, d)| {
            let f = fock_ao(ints, &total, d);
            coeff[s].transpose() * f * &coeff[s]
        });
        Ok(Reference {
            m,
            occ: [hf.n_alpha - frozen, hf.n_beta - frozen],
            coeff,
            fock,
            restricted,
            energy: hf.energy,
        })
    }

    fn closed_shell(&self) -> bool {
        self.restricted && self.occ[0] == self.occ[1]
    }

    /// Whether the occupied–virtual and off-diagonal blocks of the Fock
    /// matrices vanish (canonical Hartree–Fock orbitals).
    fn canonical(&self) -> bool {
        self.fock.iter().all(|f| {
            let scale = f.iter().fold(1.0f64, |a, &x| a.max(x.abs()));
            (0..self.m).all(|p| (0..self.m).all(|q| p == q || f[(p, q)].abs() < 1e-6 * scale))
        })
    }
}

// ---- MP2 -------------------------------------------------------------------

/// The MP2 energy of a canonical RHF or UHF reference (RHF and ROHF of a
/// closed shell are the same), with the lowest `frozen` orbitals frozen.
pub fn mp2(hf: &ScfResult, ints: &AoIntegrals, frozen: usize) -> Result<Mp2> {
    let r = Reference::new(hf, ints, frozen)?;
    if r.restricted && !r.closed_shell() {
        return Err(QsoError::invalid(
            "scf",
            "MP2 takes canonical RHF or UHF orbitals; ROHF of an open shell is not canonical",
        ));
    }
    let m = r.m;
    let e = |s: usize, p: usize| r.fock[s][(p, p)];
    // Spin pairs: (α, α), (β, β), (α, β); a closed shell has α = β.
    let pairs: &[(usize, usize)] = if r.closed_shell() {
        &[(0, 0), (0, 1)]
    } else {
        &[(0, 0), (1, 1), (0, 1)]
    };
    let (mut same, mut opposite) = (0.0, 0.0);
    for &(s1, s2) in pairs {
        let (o1, o2) = (r.occ[s1], r.occ[s2]);
        let (v1, v2) = (m - o1, m - o2);
        if o1 == 0 || o2 == 0 || v1 == 0 || v2 == 0 {
            continue;
        }
        let (c1, c2) = (&r.coeff[s1], &r.coeff[s2]);
        let occ1 = c1.columns(0, o1).into_owned();
        let vir1 = c1.columns(o1, v1).into_owned();
        let occ2 = c2.columns(0, o2).into_owned();
        let vir2 = c2.columns(o2, v2).into_owned();
        // (ia|jb), i a of spin s1 and j b of spin s2.
        let g = transform(ints, [&occ1, &vir1, &occ2, &vir2]);
        let at = |i: usize, a: usize, j: usize, b: usize| g[((i * v1 + a) * o2 + j) * v2 + b];
        let mut sum = 0.0;
        for i in 0..o1 {
            for a in 0..v1 {
                for j in 0..o2 {
                    for b in 0..v2 {
                        let d = e(s1, i) + e(s2, j) - e(s1, o1 + a) - e(s2, o2 + b);
                        let x = at(i, a, j, b);
                        if s1 == s2 {
                            // ½ Σ (ia|jb)[(ia|jb) − (ib|ja)] / D.
                            sum += 0.5 * x * (x - at(i, b, j, a)) / d;
                        } else {
                            sum += x * x / d;
                        }
                    }
                }
            }
        }
        if s1 == s2 {
            // A closed shell's β pairs equal its α pairs.
            same += if r.closed_shell() { 2.0 * sum } else { sum };
        } else {
            opposite += sum;
        }
    }
    Ok(Mp2 {
        correlation: same + opposite,
        energy: r.energy + same + opposite,
        same_spin: same,
        opposite_spin: opposite,
    })
}

// ---- DIIS ------------------------------------------------------------------

/// Pulay's DIIS on amplitude vectors, the change of each step as its error.
struct Diis {
    space: usize,
    vectors: Vec<Vec<f64>>,
    errors: Vec<Vec<f64>>,
}

impl Diis {
    fn new(space: usize) -> Self {
        Diis {
            space,
            vectors: Vec::new(),
            errors: Vec::new(),
        }
    }

    /// The extrapolated amplitudes after the step to `t` with change
    /// `error`.
    fn extrapolate(&mut self, t: Vec<f64>, error: Vec<f64>) -> Vec<f64> {
        if self.space < 2 {
            return t;
        }
        if self.vectors.len() == self.space {
            self.vectors.remove(0);
            self.errors.remove(0);
        }
        self.vectors.push(t);
        self.errors.push(error);
        let k = self.vectors.len();
        if k < 2 {
            return self.vectors[0].clone();
        }
        let mut b = DMatrix::<f64>::zeros(k + 1, k + 1);
        for i in 0..k {
            for j in 0..=i {
                let v: f64 = self.errors[i]
                    .iter()
                    .zip(&self.errors[j])
                    .map(|(x, y)| x * y)
                    .sum();
                b[(i, j)] = v;
                b[(j, i)] = v;
            }
            b[(i, k)] = -1.0;
            b[(k, i)] = -1.0;
        }
        // Scale for conditioning.
        let scale = (0..k).map(|i| b[(i, i)]).fold(0.0f64, f64::max);
        if scale > 0.0 {
            for i in 0..k {
                for j in 0..k {
                    b[(i, j)] /= scale;
                }
            }
        }
        let mut rhs = nalgebra::DVector::<f64>::zeros(k + 1);
        rhs[k] = -1.0;
        let Some(c) = b.lu().solve(&rhs) else {
            return self.vectors[k - 1].clone();
        };
        if c.iter().any(|x| !x.is_finite()) {
            return self.vectors[k - 1].clone();
        }
        let mut out = vec![0.0; self.vectors[0].len()];
        for (i, v) in self.vectors.iter().enumerate() {
            for (o, x) in out.iter_mut().zip(v) {
                *o += c[i] * x;
            }
        }
        out
    }
}

/// Iterate `step` (amplitudes → new amplitudes and their energy) from
/// `start` to convergence.
fn solve(
    start: Vec<f64>,
    options: &CcOptions,
    mut step: impl FnMut(&[f64]) -> (Vec<f64>, f64),
) -> Result<(Vec<f64>, f64, usize)> {
    options.validate()?;
    let mut t = start;
    let mut energy = f64::NAN;
    let mut diis = Diis::new(options.diis_space);
    for iteration in 1..=options.max_iterations {
        let (next, e) = step(&t);
        if !e.is_finite() || next.iter().any(|x| !x.is_finite()) {
            return Err(QsoError::invalid(
                "ccsd",
                format!("the amplitudes diverged at iteration {iteration}"),
            ));
        }
        let change: Vec<f64> = next.iter().zip(&t).map(|(a, b)| a - b).collect();
        let norm = change.iter().map(|x| x * x).sum::<f64>().sqrt();
        let converged =
            (e - energy).abs() < options.energy_tolerance && norm < options.amplitude_tolerance;
        energy = e;
        if converged {
            return Ok((next, energy, iteration));
        }
        t = diis.extrapolate(next, change);
    }
    Err(QsoError::invalid(
        "ccsd",
        format!(
            "the amplitudes did not converge in {} iterations",
            options.max_iterations
        ),
    ))
}

// ---- closed shells in spatial orbitals --------------------------------------

/// The closed-shell equations in spatial orbitals: `o` occupied and `v`
/// virtual orbitals, `(pq|rs)` over all of them, the Fock matrix.
struct Closed {
    o: usize,
    v: usize,
    g: Vec<f64>,
    f: RMatrix,
}

impl Closed {
    fn new(r: &Reference, ints: &AoIntegrals) -> Self {
        let c = &r.coeff[0];
        Closed {
            o: r.occ[0],
            v: r.m - r.occ[0],
            g: transform(ints, [c, c, c, c]),
            f: r.fock[0].clone(),
        }
    }

    /// `(pq|rs)`, occupied orbitals first.
    #[inline]
    fn g(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let m = self.o + self.v;
        self.g[((p * m + q) * m + r) * m + s]
    }

    fn eo(&self, i: usize) -> f64 {
        self.f[(i, i)]
    }

    fn ev(&self, a: usize) -> f64 {
        self.f[(self.o + a, self.o + a)]
    }

    fn fov(&self, i: usize, a: usize) -> f64 {
        self.f[(i, self.o + a)]
    }

    fn t2i(&self, i: usize, j: usize, a: usize, b: usize) -> usize {
        ((i * self.o + j) * self.v + a) * self.v + b
    }

    /// MP2 amplitudes: `t1 = 0`, `t2 = (ia|jb)/D`.
    fn start(&self) -> Vec<f64> {
        let (o, v) = (self.o, self.v);
        let mut t = vec![0.0; o * v + o * o * v * v];
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let d = self.eo(i) + self.eo(j) - self.ev(a) - self.ev(b);
                        t[o * v + self.t2i(i, j, a, b)] = self.g(i, o + a, j, o + b) / d;
                    }
                }
            }
        }
        t
    }

    fn energy(&self, t1: &[f64], t2: &[f64]) -> f64 {
        let (o, v) = (self.o, self.v);
        let mut e = 0.0;
        for i in 0..o {
            for a in 0..v {
                e += 2.0 * self.fov(i, a) * t1[i * v + a];
            }
        }
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let tau = t2[self.t2i(i, j, a, b)] + t1[i * v + a] * t1[j * v + b];
                        e += tau * (2.0 * self.g(i, o + a, j, o + b) - self.g(i, o + b, j, o + a));
                    }
                }
            }
        }
        e
    }

    /// One Jacobi step of the closed-shell CCSD equations (the
    /// intermediates of Koch et al. as PySCF's `rccsd` writes them).
    fn step(&self, t: &[f64]) -> (Vec<f64>, f64) {
        let (o, v) = (self.o, self.v);
        let (t1, t2) = t.split_at(o * v);
        let t1 = |i: usize, a: usize| t1[i * v + a];
        let t2 = |i: usize, j: usize, a: usize, b: usize| t2[self.t2i(i, j, a, b)];
        let ovov = |i, a, j, b| self.g(i, o + a, j, o + b);
        let oovv = |i, j, a, b| self.g(i, j, o + a, o + b);
        let ovvo = |i, a, b, j| self.g(i, o + a, o + b, j);
        let ovoo = |i, a, j, k| self.g(i, o + a, j, k);
        let oooo = |i, j, k, l| self.g(i, j, k, l);
        let ovvv = |i, a, b, c| self.g(i, o + a, o + b, o + c);
        let vvvv = |a, b, c, d| self.g(o + a, o + b, o + c, o + d);
        let fov = |i, a| self.fov(i, a);
        // Foo, Fvv, Fov.
        let mut f_oo = vec![0.0; o * o];
        for k in 0..o {
            for i in 0..o {
                let mut x = self.f[(k, i)];
                for l in 0..o {
                    for c in 0..v {
                        for d in 0..v {
                            let w = 2.0 * ovov(k, c, l, d) - ovov(k, d, l, c);
                            x += w * (t2(i, l, c, d) + t1(i, c) * t1(l, d));
                        }
                    }
                }
                f_oo[k * o + i] = x;
            }
        }
        let mut fvv = vec![0.0; v * v];
        for a in 0..v {
            for c in 0..v {
                let mut x = self.f[(o + a, o + c)];
                for k in 0..o {
                    for l in 0..o {
                        for d in 0..v {
                            let w = 2.0 * ovov(k, c, l, d) - ovov(k, d, l, c);
                            x -= w * (t2(k, l, a, d) + t1(k, a) * t1(l, d));
                        }
                    }
                }
                fvv[a * v + c] = x;
            }
        }
        let mut fovi = vec![0.0; o * v];
        for k in 0..o {
            for c in 0..v {
                let mut x = fov(k, c);
                for l in 0..o {
                    for d in 0..v {
                        x += (2.0 * ovov(k, c, l, d) - ovov(k, d, l, c)) * t1(l, d);
                    }
                }
                fovi[k * v + c] = x;
            }
        }
        let foo_ = |k: usize, i: usize| f_oo[k * o + i] - if k == i { self.eo(i) } else { 0.0 };
        let fvv_ = |a: usize, c: usize| fvv[a * v + c] - if a == c { self.ev(a) } else { 0.0 };
        let fov_ = |k: usize, c: usize| fovi[k * v + c];
        // T1.
        let mut r1 = vec![0.0; o * v];
        for i in 0..o {
            for a in 0..v {
                let mut x = fov(i, a);
                for k in 0..o {
                    for c in 0..v {
                        x -= 2.0 * fov(k, c) * t1(k, a) * t1(i, c);
                        x += 2.0 * fov_(k, c) * t2(k, i, c, a);
                        x -= fov_(k, c) * t2(i, k, c, a);
                        x += fov_(k, c) * t1(i, c) * t1(k, a);
                        x += 2.0 * ovvo(k, c, a, i) * t1(k, c);
                        x -= oovv(k, i, a, c) * t1(k, c);
                    }
                }
                for c in 0..v {
                    x += fvv_(a, c) * t1(i, c);
                }
                for k in 0..o {
                    x -= foo_(k, i) * t1(k, a);
                }
                for k in 0..o {
                    for c in 0..v {
                        for d in 0..v {
                            let w = 2.0 * ovvv(k, d, a, c) - ovvv(k, c, a, d);
                            x += w * (t2(i, k, c, d) + t1(k, d) * t1(i, c));
                        }
                    }
                }
                for k in 0..o {
                    for l in 0..o {
                        for c in 0..v {
                            let w = 2.0 * ovoo(l, c, k, i) - ovoo(k, c, l, i);
                            x -= w * (t2(k, l, a, c) + t1(l, c) * t1(k, a));
                        }
                    }
                }
                r1[i * v + a] = x;
            }
        }
        // Loo, Lvv.
        let mut loo = vec![0.0; o * o];
        for k in 0..o {
            for i in 0..o {
                let mut x = foo_(k, i);
                for c in 0..v {
                    x += fov(k, c) * t1(i, c);
                }
                for l in 0..o {
                    for c in 0..v {
                        x += (2.0 * ovoo(l, c, k, i) - ovoo(k, c, l, i)) * t1(l, c);
                    }
                }
                loo[k * o + i] = x;
            }
        }
        let mut lvv = vec![0.0; v * v];
        for a in 0..v {
            for c in 0..v {
                let mut x = fvv_(a, c);
                for k in 0..o {
                    x -= fov(k, c) * t1(k, a);
                    for d in 0..v {
                        x += (2.0 * ovvv(k, d, a, c) - ovvv(k, c, a, d)) * t1(k, d);
                    }
                }
                lvv[a * v + c] = x;
            }
        }
        // Woooo[k, l, i, j].
        let mut woooo = vec![0.0; o * o * o * o];
        for k in 0..o {
            for l in 0..o {
                for i in 0..o {
                    for j in 0..o {
                        let mut x = oooo(k, i, l, j);
                        for c in 0..v {
                            x += ovoo(l, c, k, i) * t1(j, c) + ovoo(k, c, l, j) * t1(i, c);
                            for d in 0..v {
                                x += ovov(k, c, l, d) * (t2(i, j, c, d) + t1(i, c) * t1(j, d));
                            }
                        }
                        woooo[((k * o + l) * o + i) * o + j] = x;
                    }
                }
            }
        }
        // Wvvvv[a, b, c, d] as the (ab × cd) matrix.
        let mut wvvvv = DMatrix::<f64>::zeros(v * v, v * v);
        for a in 0..v {
            for b in 0..v {
                for c in 0..v {
                    for d in 0..v {
                        let mut x = vvvv(a, c, b, d);
                        for k in 0..o {
                            x -= ovvv(k, d, a, c) * t1(k, b) + ovvv(k, c, b, d) * t1(k, a);
                        }
                        wvvvv[(a * v + b, c * v + d)] = x;
                    }
                }
            }
        }
        // Wvoov[a, k, i, c], Wvovo[a, k, c, i].
        let mut wvoov = vec![0.0; v * o * o * v];
        let mut wvovo = vec![0.0; v * o * v * o];
        for a in 0..v {
            for k in 0..o {
                for i in 0..o {
                    for c in 0..v {
                        let mut x = ovvo(k, c, a, i);
                        let mut y = oovv(k, i, a, c);
                        for d in 0..v {
                            x += ovvv(k, c, a, d) * t1(i, d);
                            y += ovvv(k, d, a, c) * t1(i, d);
                        }
                        for l in 0..o {
                            x -= ovoo(k, c, l, i) * t1(l, a);
                            y -= ovoo(l, c, k, i) * t1(l, a);
                            for d in 0..v {
                                x -= 0.5 * ovov(l, d, k, c) * t2(i, l, d, a);
                                x -= 0.5 * ovov(l, c, k, d) * t2(i, l, a, d);
                                x -= ovov(l, d, k, c) * t1(i, d) * t1(l, a);
                                x += ovov(l, d, k, c) * t2(i, l, a, d);
                                y -= 0.5 * ovov(l, c, k, d) * t2(i, l, d, a);
                                y -= ovov(l, c, k, d) * t1(i, d) * t1(l, a);
                            }
                        }
                        wvoov[((a * o + k) * o + i) * v + c] = x;
                        wvovo[((a * o + k) * v + c) * o + i] = y;
                    }
                }
            }
        }
        let wvoov = |a: usize, k: usize, i: usize, c: usize| wvoov[((a * o + k) * o + i) * v + c];
        let wvovo = |a: usize, k: usize, c: usize, i: usize| wvovo[((a * o + k) * v + c) * o + i];
        // T2: terms that are added with their (i ↔ j, a ↔ b) partner.
        let mut half = vec![0.0; o * o * v * v];
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let mut x = 0.0;
                        for c in 0..v {
                            // (ia|cb) − Σ_k (ki|bc) t_k^a, times t_j^c.
                            let mut y = ovvv(i, a, c, b);
                            for k in 0..o {
                                y -= oovv(k, i, b, c) * t1(k, a);
                            }
                            x += y * t1(j, c);
                            x += lvv[a * v + c] * t2(i, j, c, b);
                        }
                        for k in 0..o {
                            // (ia|jk) + Σ_c (kc|ai) t_j^c, times t_k^b.
                            let mut y = ovoo(i, a, j, k);
                            for c in 0..v {
                                y += ovvo(k, c, a, i) * t1(j, c);
                            }
                            x -= y * t1(k, b);
                            x -= loo[k * o + i] * t2(k, j, a, b);
                            for c in 0..v {
                                x += 2.0 * wvoov(a, k, i, c) * t2(k, j, c, b);
                                x -= wvovo(a, k, c, i) * t2(k, j, c, b);
                                x -= wvoov(a, k, i, c) * t2(k, j, b, c);
                                x -= wvovo(b, k, c, i) * t2(k, j, a, c);
                            }
                        }
                        half[self.t2i(i, j, a, b)] = x;
                    }
                }
            }
        }
        // τ as the (ij × cd) matrix, and Σ_cd τ_ij^cd W_abcd.
        let mut tau = DMatrix::<f64>::zeros(o * o, v * v);
        for i in 0..o {
            for j in 0..o {
                for c in 0..v {
                    for d in 0..v {
                        tau[(i * o + j, c * v + d)] = t2(i, j, c, d) + t1(i, c) * t1(j, d);
                    }
                }
            }
        }
        let ladder = &tau * wvvvv.transpose();
        let mut r2 = vec![0.0; o * o * v * v];
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let mut x = ovov(i, a, j, b)
                            + half[self.t2i(i, j, a, b)]
                            + half[self.t2i(j, i, b, a)]
                            + ladder[(i * o + j, a * v + b)];
                        for k in 0..o {
                            for l in 0..o {
                                x += woooo[((k * o + l) * o + i) * o + j]
                                    * tau[(k * o + l, a * v + b)];
                            }
                        }
                        r2[self.t2i(i, j, a, b)] = x;
                    }
                }
            }
        }
        let mut next = vec![0.0; t.len()];
        for i in 0..o {
            for a in 0..v {
                next[i * v + a] = r1[i * v + a] / (self.eo(i) - self.ev(a));
            }
        }
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let d = self.eo(i) + self.eo(j) - self.ev(a) - self.ev(b);
                        next[o * v + self.t2i(i, j, a, b)] = r2[self.t2i(i, j, a, b)] / d;
                    }
                }
            }
        }
        let e = self.energy(&next[..o * v], &next[o * v..]);
        (next, e)
    }

    /// The perturbative triples of a canonical closed shell, in the form of
    /// Rendell, Lee and Komornicki: `W_ijk^abc = P[Σ_d (bd|ck) t_ij^ad −
    /// Σ_l (ck|jl) t_il^ab]` over the six permutations of the pairs (ia),
    /// (jb), (kc); `V = W + (bj|ck) t_i^a + (ai|ck) t_j^b + (ai|bj) t_k^c`;
    /// `E = Σ (4W_abc + W_bca + W_cab)(V_abc − V_cba)/(3D)`.
    fn triples(&self, t: &[f64]) -> f64 {
        let (o, v) = (self.o, self.v);
        let (t1, t2) = t.split_at(o * v);
        let t2 = |i: usize, j: usize, a: usize, b: usize| t2[self.t2i(i, j, a, b)];
        // X_k[d, (b, c)] = (bd|ck), and t_ij as (a × d) matrices.
        let x: Vec<DMatrix<f64>> = (0..o)
            .map(|k| DMatrix::from_fn(v, v * v, |d, bc| self.g(o + bc / v, o + d, o + bc % v, k)))
            .collect();
        let tij: Vec<DMatrix<f64>> = (0..o * o)
            .map(|ij| DMatrix::from_fn(v, v, |a, d| t2(ij / o, ij % o, a, d)))
            .collect();
        let mut energy = 0.0;
        let idx = |a: usize, b: usize, c: usize| (a * v + b) * v + c;
        for i in 0..o {
            for j in 0..o {
                for k in 0..o {
                    // W over the six permutations of (ia), (jb), (kc): the
                    // permutation (i' a', j' b', k' c') adds
                    // Σ_d (b'd|c'k') t_i'j'^a'd − Σ_l (c'k'|j'l) t_i'l^a'b'.
                    let mut w = vec![0.0; v * v * v];
                    let perms: [([usize; 3], [usize; 3]); 6] = [
                        ([i, j, k], [0, 1, 2]),
                        ([i, k, j], [0, 2, 1]),
                        ([j, i, k], [1, 0, 2]),
                        ([j, k, i], [1, 2, 0]),
                        ([k, i, j], [2, 0, 1]),
                        ([k, j, i], [2, 1, 0]),
                    ];
                    for (occ, pos) in perms {
                        let [pi, pj, pk] = occ;
                        // Σ_d t_pipj[a', d] (b'd|c'pk) for all a', b', c'.
                        let prod = &tij[pi * o + pj] * &x[pk];
                        for a in 0..v {
                            for b in 0..v {
                                for c in 0..v {
                                    let abc = [a, b, c];
                                    let (a1, b1, c1) = (abc[pos[0]], abc[pos[1]], abc[pos[2]]);
                                    let mut val = prod[(a1, b1 * v + c1)];
                                    for l in 0..o {
                                        val -= self.g(o + c1, pk, pj, l) * t2(pi, l, a1, b1);
                                    }
                                    w[idx(a, b, c)] += val;
                                }
                            }
                        }
                    }
                    let ev = self.eo(i) + self.eo(j) + self.eo(k);
                    for a in 0..v {
                        for b in 0..v {
                            for c in 0..v {
                                let wabc = w[idx(a, b, c)];
                                let vv = |a: usize, b: usize, c: usize| {
                                    w[idx(a, b, c)]
                                        + self.g(o + b, j, o + c, k) * t1[i * v + a]
                                        + self.g(o + a, i, o + c, k) * t1[j * v + b]
                                        + self.g(o + a, i, o + b, j) * t1[k * v + c]
                                };
                                let d = ev - self.ev(a) - self.ev(b) - self.ev(c);
                                let y = 4.0 * wabc + w[idx(b, c, a)] + w[idx(c, a, b)];
                                energy += y * (vv(a, b, c) - vv(c, b, a)) / (3.0 * d);
                            }
                        }
                    }
                }
            }
        }
        energy
    }
}

// ---- spin orbitals -----------------------------------------------------------

/// The spin-orbital equations: occupied spin orbitals first (α then β),
/// then the virtual ones (α then β), their antisymmetrised integrals
/// `⟨pq||rs⟩` and Fock matrix.
struct Spin {
    o: usize,
    v: usize,
    /// `⟨pq||rs⟩` over all spin orbitals.
    w: Vec<f64>,
    f: RMatrix,
    /// `⟨ab||ef⟩` as the (ab × ef) matrix, `⟨mn||ef⟩` as (mn × ef) and
    /// `⟨am||ef⟩` as (ef × am).
    vvvv: DMatrix<f64>,
    oovv: DMatrix<f64>,
    vovv: DMatrix<f64>,
}

impl Spin {
    fn new(r: &Reference, ints: &AoIntegrals) -> Result<Self> {
        let m = r.m;
        let n = 2 * m;
        if n > MAX_SPIN_ORBITALS {
            return Err(QsoError::invalid(
                "basis",
                format!(
                    "{n} correlated spin orbitals exceed the {MAX_SPIN_ORBITALS} of a spin-orbital calculation"
                ),
            ));
        }
        let [oa, ob] = r.occ;
        // Spin orbital → (spin, active spatial orbital).
        let mut map = Vec::with_capacity(n);
        map.extend((0..oa).map(|p| (0usize, p)));
        map.extend((0..ob).map(|p| (1usize, p)));
        map.extend((oa..m).map(|p| (0usize, p)));
        map.extend((ob..m).map(|p| (1usize, p)));
        let (ca, cb) = (&r.coeff[0], &r.coeff[1]);
        let aa = transform(ints, [ca, ca, ca, ca]);
        let (ab, bb) = if r.restricted {
            (aa.clone(), aa.clone())
        } else {
            (
                transform(ints, [ca, ca, cb, cb]),
                transform(ints, [cb, cb, cb, cb]),
            )
        };
        // (pq|rs) of spin orbitals.
        let chem = |p: usize, q: usize, r: usize, s: usize| -> f64 {
            let ((sp, p), (sq, q), (sr, r), (ss, s)) = (map[p], map[q], map[r], map[s]);
            if sp != sq || sr != ss {
                return 0.0;
            }
            let at = |t: &[f64], p, q, r, s| t[((p * m + q) * m + r) * m + s];
            match (sp, sr) {
                (0, 0) => at(&aa, p, q, r, s),
                (1, 1) => at(&bb, p, q, r, s),
                (0, 1) => at(&ab, p, q, r, s),
                _ => at(&ab, r, s, p, q),
            }
        };
        let mut w = vec![0.0; n * n * n * n];
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        w[((p * n + q) * n + r) * n + s] = chem(p, r, q, s) - chem(p, s, q, r);
                    }
                }
            }
        }
        let f = RMatrix::from_fn(n, n, |p, q| {
            let ((sp, p), (sq, q)) = (map[p], map[q]);
            if sp == sq { r.fock[sp][(p, q)] } else { 0.0 }
        });
        let (o, v) = (oa + ob, n - oa - ob);
        let at = |p: usize, q: usize, r: usize, s: usize| w[((p * n + q) * n + r) * n + s];
        let vvvv = DMatrix::from_fn(v * v, v * v, |ab, ef| {
            at(o + ab / v, o + ab % v, o + ef / v, o + ef % v)
        });
        let oovv = DMatrix::from_fn(o * o, v * v, |mn, ef| {
            at(mn / o, mn % o, o + ef / v, o + ef % v)
        });
        let vovv = DMatrix::from_fn(v * v, v * o, |ef, am| {
            at(o + am / o, am % o, o + ef / v, o + ef % v)
        });
        Ok(Spin {
            o,
            v,
            w,
            f,
            vvvv,
            oovv,
            vovv,
        })
    }

    /// `⟨pq||rs⟩`, occupied spin orbitals first.
    #[inline]
    fn w(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let n = self.o + self.v;
        self.w[((p * n + q) * n + r) * n + s]
    }

    fn t2i(&self, i: usize, j: usize, a: usize, b: usize) -> usize {
        ((i * self.o + j) * self.v + a) * self.v + b
    }

    fn dia(&self, i: usize, a: usize) -> f64 {
        self.f[(i, i)] - self.f[(self.o + a, self.o + a)]
    }

    fn start(&self) -> Vec<f64> {
        let (o, v) = (self.o, self.v);
        let mut t = vec![0.0; o * v + o * o * v * v];
        for i in 0..o {
            for a in 0..v {
                t[i * v + a] = self.f[(i, o + a)] / self.dia(i, a);
            }
        }
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let d = self.dia(i, a) + self.dia(j, b);
                        t[o * v + self.t2i(i, j, a, b)] = self.w(i, j, o + a, o + b) / d;
                    }
                }
            }
        }
        t
    }

    fn energy(&self, t1: &[f64], t2: &[f64]) -> f64 {
        let (o, v) = (self.o, self.v);
        let mut e = 0.0;
        for i in 0..o {
            for a in 0..v {
                e += self.f[(i, o + a)] * t1[i * v + a];
            }
        }
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let w = self.w(i, j, o + a, o + b);
                        e += 0.25 * w * t2[self.t2i(i, j, a, b)]
                            + 0.5 * w * t1[i * v + a] * t1[j * v + b];
                    }
                }
            }
        }
        e
    }

    /// One Jacobi step of the spin-orbital CCSD equations of Stanton,
    /// Gauss, Watts and Bartlett.
    fn step(&self, t: &[f64]) -> (Vec<f64>, f64) {
        let (o, v) = (self.o, self.v);
        let (t1s, t2s) = t.split_at(o * v);
        let t1 = |i: usize, a: usize| t1s[i * v + a];
        let t2 = |i: usize, j: usize, a: usize, b: usize| t2s[self.t2i(i, j, a, b)];
        let w = |p, q, r, s| self.w(p, q, r, s);
        let f = |p: usize, q: usize| self.f[(p, q)];
        // Occupied p, virtual o + a.
        let tau_t = |i, j, a, b| t2(i, j, a, b) + 0.5 * (t1(i, a) * t1(j, b) - t1(i, b) * t1(j, a));
        let tau = |i, j, a, b| t2(i, j, a, b) + t1(i, a) * t1(j, b) - t1(i, b) * t1(j, a);
        // F_ae, F_mi, F_me.
        let mut fae = vec![0.0; v * v];
        for a in 0..v {
            for e in 0..v {
                let mut x = if a == e { 0.0 } else { f(o + a, o + e) };
                for m in 0..o {
                    x -= 0.5 * f(m, o + e) * t1(m, a);
                    for ff in 0..v {
                        x += t1(m, ff) * w(m, o + a, o + ff, o + e);
                        for n in 0..o {
                            x -= 0.5 * tau_t(m, n, a, ff) * w(m, n, o + e, o + ff);
                        }
                    }
                }
                fae[a * v + e] = x;
            }
        }
        let mut fmi = vec![0.0; o * o];
        for m in 0..o {
            for i in 0..o {
                let mut x = if m == i { 0.0 } else { f(m, i) };
                for e in 0..v {
                    x += 0.5 * t1(i, e) * f(m, o + e);
                    for n in 0..o {
                        x += t1(n, e) * w(m, n, i, o + e);
                        for ff in 0..v {
                            x += 0.5 * tau_t(i, n, e, ff) * w(m, n, o + e, o + ff);
                        }
                    }
                }
                fmi[m * o + i] = x;
            }
        }
        let mut fme = vec![0.0; o * v];
        for m in 0..o {
            for e in 0..v {
                let mut x = f(m, o + e);
                for n in 0..o {
                    for ff in 0..v {
                        x += t1(n, ff) * w(m, n, o + e, o + ff);
                    }
                }
                fme[m * v + e] = x;
            }
        }
        let fae = |a: usize, e: usize| fae[a * v + e];
        let fmi = |m: usize, i: usize| fmi[m * o + i];
        let fme = |m: usize, e: usize| fme[m * v + e];
        // τ as the (ij × ef) matrix, and X[mn, ij] = Σ_ef ⟨mn||ef⟩ τ_ij^ef.
        let tau_m = DMatrix::from_fn(o * o, v * v, |ij, ef| tau(ij / o, ij % o, ef / v, ef % v));
        let xmn = &self.oovv * tau_m.transpose();
        // W_mnij as the (mn × ij) matrix: ⟨mn||ij⟩ + P(ij) t_j^e ⟨mn||ie⟩
        // + ¼ X.
        let mut wmnij = 0.25 * &xmn;
        for m in 0..o {
            for n in 0..o {
                for i in 0..o {
                    for j in 0..o {
                        let mut x = w(m, n, i, j);
                        for e in 0..v {
                            x += t1(j, e) * w(m, n, i, o + e) - t1(i, e) * w(m, n, j, o + e);
                        }
                        wmnij[(m * o + n, i * o + j)] += x;
                    }
                }
            }
        }
        // W_mbej as the (me × bj) matrix: ⟨mb||ej⟩ + t_j^f ⟨mb||ef⟩ −
        // t_n^b ⟨mn||ej⟩ − Σ_nf (½ t_jn^fb + t_j^f t_n^b) ⟨mn||ef⟩.
        let x = DMatrix::from_fn(o * v, o * v, |jb, nf| {
            let (j, b, n, ff) = (jb / v, jb % v, nf / v, nf % v);
            0.5 * t2(j, n, ff, b) + t1(j, ff) * t1(n, b)
        });
        let y = DMatrix::from_fn(o * v, o * v, |nf, me| {
            w(me / v, nf / v, o + me % v, o + nf % v)
        });
        let z = &x * &y;
        let mut wmbej = DMatrix::<f64>::zeros(o * v, v * o);
        for m in 0..o {
            for b in 0..v {
                for e in 0..v {
                    for j in 0..o {
                        let mut x = w(m, o + b, o + e, j) - z[(j * v + b, m * v + e)];
                        for ff in 0..v {
                            x += t1(j, ff) * w(m, o + b, o + e, o + ff);
                        }
                        for n in 0..o {
                            x -= t1(n, b) * w(m, n, o + e, j);
                        }
                        wmbej[(m * v + e, b * o + j)] = x;
                    }
                }
            }
        }
        // T1.
        let mut next = vec![0.0; t.len()];
        for i in 0..o {
            for a in 0..v {
                let mut x = f(i, o + a);
                for e in 0..v {
                    x += t1(i, e) * fae(a, e);
                }
                for m in 0..o {
                    x -= t1(m, a) * fmi(m, i);
                    for e in 0..v {
                        x += t2(i, m, a, e) * fme(m, e);
                        for ff in 0..v {
                            x -= 0.5 * t2(i, m, e, ff) * w(m, o + a, o + e, o + ff);
                        }
                        for n in 0..o {
                            x -= 0.5 * t2(m, n, a, e) * w(n, m, o + e, i);
                        }
                    }
                }
                for n in 0..o {
                    for ff in 0..v {
                        x -= t1(n, ff) * w(n, o + a, i, o + ff);
                    }
                }
                next[i * v + a] = x / self.dia(i, a);
            }
        }
        // T2: the ladders Σ_ef τ_ij^ef W_abef, with W_abef = ⟨ab||ef⟩ −
        // P(ab) t_m^b ⟨am||ef⟩ + ¼ Σ_mn τ_mn^ab ⟨mn||ef⟩ never built:
        // τ ⟨ab||ef⟩ᵀ − P(ab) Σ_m t_m^b Z[ij, am] + ¼ Xᵀ τ, with Z[ij, am] =
        // Σ_ef τ_ij^ef ⟨am||ef⟩; and Σ_mn τ_mn^ab W_mnij.
        let z = &tau_m * &self.vovv;
        let mut ladder = &tau_m * self.vvvv.transpose() + 0.25 * xmn.transpose() * &tau_m;
        for ij in 0..o * o {
            for a in 0..v {
                for b in 0..v {
                    let mut x = 0.0;
                    for m in 0..o {
                        x -= t1(m, b) * z[(ij, a * o + m)] - t1(m, a) * z[(ij, b * o + m)];
                    }
                    ladder[(ij, a * v + b)] += x;
                }
            }
        }
        let ring = wmnij.transpose() * &tau_m;
        // Σ_me (t_im^ae W_mbej − t_i^e t_m^a ⟨mb||ej⟩) as the (ia × bj)
        // matrix.
        let tme = DMatrix::from_fn(o * v, o * v, |ia, me| t2(ia / v, me / v, ia % v, me % v));
        let s1 = DMatrix::from_fn(o * v, o * v, |ia, me| {
            t1(ia / v, me % v) * t1(me / v, ia % v)
        });
        let vmat = DMatrix::from_fn(o * v, v * o, |me, bj| {
            w(me / v, o + bj / o, o + me % v, bj % o)
        });
        let both = &tme * &wmbej - &s1 * &vmat;
        // Σ_e t_ij^ae G_be with G_be = F_be − ½ t_m^b F_me: (ija × e)(e × b).
        let g = DMatrix::from_fn(v, v, |e, b| {
            let mut x = fae(b, e);
            for m in 0..o {
                x -= 0.5 * t1(m, b) * fme(m, e);
            }
            x
        });
        let t2m = DMatrix::from_fn(o * o * v, v, |ija, e| t2s[ija * v + e]);
        let pab_t = &t2m * &g;
        // Σ_m t_im^ab H_mj with H_mj = F_mj + ½ t_j^e F_me: (iab × m)(m × j).
        let h = DMatrix::from_fn(o, o, |m, j| {
            let mut x = fmi(m, j);
            for e in 0..v {
                x += 0.5 * t1(j, e) * fme(m, e);
            }
            x
        });
        let t2i = DMatrix::from_fn(o * v * v, o, |iab, m| {
            let (i, ab) = (iab / (v * v), iab % (v * v));
            t2s[(i * o + m) * v * v + ab]
        });
        let pij_t = &t2i * &h;
        let mut pab = vec![0.0; o * o * v * v];
        let mut pij = vec![0.0; o * o * v * v];
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let k = self.t2i(i, j, a, b);
                        let mut xab = pab_t[((i * o + j) * v + a, b)];
                        let mut xij = -pij_t[((i * v + a) * v + b, j)];
                        for m in 0..o {
                            xab -= t1(m, a) * w(m, o + b, i, j);
                        }
                        for e in 0..v {
                            xij += t1(i, e) * w(o + a, o + b, o + e, j);
                        }
                        pab[k] = xab;
                        pij[k] = xij;
                    }
                }
            }
        }
        let both_at = |i: usize, j: usize, a: usize, b: usize| both[(i * v + a, b * o + j)];
        for i in 0..o {
            for j in 0..o {
                for a in 0..v {
                    for b in 0..v {
                        let k = self.t2i(i, j, a, b);
                        let x = w(i, j, o + a, o + b) + pab[k] - pab[self.t2i(i, j, b, a)] + pij[k]
                            - pij[self.t2i(j, i, a, b)]
                            + both_at(i, j, a, b)
                            - both_at(j, i, a, b)
                            - both_at(i, j, b, a)
                            + both_at(j, i, b, a)
                            + 0.5 * ladder[(i * o + j, a * v + b)]
                            + 0.5 * ring[(i * o + j, a * v + b)];
                        let d = self.dia(i, a) + self.dia(j, b);
                        next[o * v + k] = x / d;
                    }
                }
            }
        }
        let e = self.energy(&next[..o * v], &next[o * v..]);
        (next, e)
    }

    /// The perturbative triples of a canonical reference in spin orbitals:
    /// `E = Σ_{i<j<k, a<b<c} t(c) D (t(c) + t(d))` with
    /// `D t(d) = P(i/jk) P(a/bc) t_i^a ⟨jk||bc⟩` and
    /// `D t(c) = P(i/jk) P(a/bc) [Σ_e t_jk^ae ⟨ei||bc⟩ − Σ_m t_im^bc ⟨ma||jk⟩]`.
    fn triples(&self, t: &[f64]) -> f64 {
        let (o, v) = (self.o, self.v);
        let (t1s, t2s) = t.split_at(o * v);
        let t1 = |i: usize, a: usize| t1s[i * v + a];
        let t2 = |i: usize, j: usize, a: usize, b: usize| t2s[self.t2i(i, j, a, b)];
        // For each occupied p: E_p[e, (b, c)] = ⟨ep||bc⟩ and T_p[m, (b, c)]
        // = t_pm^bc; for each pair (q, r): T_qr[a, e] = t_qr^ae and
        // M_qr[a, m] = ⟨ma||qr⟩.
        let ep: Vec<DMatrix<f64>> = (0..o)
            .map(|p| DMatrix::from_fn(v, v * v, |e, bc| self.w(o + e, p, o + bc / v, o + bc % v)))
            .collect();
        let tp: Vec<DMatrix<f64>> = (0..o)
            .map(|p| DMatrix::from_fn(o, v * v, |m, bc| t2(p, m, bc / v, bc % v)))
            .collect();
        let tqr = |q: usize, r: usize| DMatrix::from_fn(v, v, |a, e| t2(q, r, a, e));
        let mqr = |q: usize, r: usize| DMatrix::from_fn(v, o, |a, m| self.w(m, o + a, q, r));
        // G_pqr[a, (b, c)] = Σ_e t_qr^ae ⟨ep||bc⟩ − Σ_m t_pm^bc ⟨ma||qr⟩.
        let g = |p: usize, q: usize, r: usize| &tqr(q, r) * &ep[p] - &mqr(q, r) * &tp[p];
        let mut energy = 0.0;
        for i in 0..o {
            for j in i + 1..o {
                for k in j + 1..o {
                    // P(i/jk) of the connected part.
                    let acc = g(i, j, k) - g(j, i, k) - g(k, j, i);
                    let at = |a: usize, b: usize, c: usize| acc[(a, b * v + c)];
                    let disc = |i: usize, j: usize, k: usize, a: usize, b: usize, c: usize| {
                        t1(i, a) * self.w(j, k, o + b, o + c)
                    };
                    let eo = self.f[(i, i)] + self.f[(j, j)] + self.f[(k, k)];
                    for a in 0..v {
                        for b in a + 1..v {
                            for c in b + 1..v {
                                let d = eo
                                    - self.f[(o + a, o + a)]
                                    - self.f[(o + b, o + b)]
                                    - self.f[(o + c, o + c)];
                                let tc = (at(a, b, c) - at(b, a, c) - at(c, b, a)) / d;
                                let pa = |i, j, k| {
                                    disc(i, j, k, a, b, c)
                                        - disc(i, j, k, b, a, c)
                                        - disc(i, j, k, c, b, a)
                                };
                                let td = (pa(i, j, k) - pa(j, i, k) - pa(k, j, i)) / d;
                                energy += tc * d * (tc + td);
                            }
                        }
                    }
                }
            }
        }
        energy
    }

    fn t1_diagnostic(&self, t: &[f64]) -> f64 {
        let norm: f64 = t[..self.o * self.v].iter().map(|x| x * x).sum();
        (norm / (2 * self.o).max(1) as f64).sqrt()
    }
}

/// How a CCSD calculation is solved.
enum Solved {
    Closed(Closed, Vec<f64>),
    Spin(Spin, Vec<f64>),
}

fn run_ccsd(
    hf: &ScfResult,
    ints: &AoIntegrals,
    frozen: usize,
    options: &CcOptions,
    spin_orbitals: bool,
) -> Result<(Ccsd, Solved)> {
    options.validate()?;
    let r = Reference::new(hf, ints, frozen)?;
    if r.closed_shell() && !spin_orbitals {
        let c = Closed::new(&r, ints);
        let (o, v) = (c.o, c.v);
        let (t, e, iterations) = if o == 0 || v == 0 {
            (vec![0.0; o * v + o * o * v * v], 0.0, 0)
        } else {
            solve(c.start(), options, |t| c.step(t))?
        };
        let norm: f64 = t[..o * v].iter().map(|x| x * x).sum();
        let result = Ccsd {
            correlation: e,
            energy: r.energy + e,
            iterations,
            t1_diagnostic: (norm / (2 * o).max(1) as f64).sqrt(),
            spin_orbitals: false,
        };
        return Ok((result, Solved::Closed(c, t)));
    }
    let s = Spin::new(&r, ints)?;
    let (o, v) = (s.o, s.v);
    let (t, e, iterations) = if o == 0 || v == 0 {
        (vec![0.0; o * v + o * o * v * v], 0.0, 0)
    } else {
        solve(s.start(), options, |t| s.step(t))?
    };
    let result = Ccsd {
        correlation: e,
        energy: r.energy + e,
        iterations,
        t1_diagnostic: s.t1_diagnostic(&t),
        spin_orbitals: true,
    };
    Ok((result, Solved::Spin(s, t)))
}

/// CCSD in the orbitals of `hf` with the lowest `frozen` orbitals frozen:
/// in spatial orbitals for a closed shell of RHF (or ROHF), in spin
/// orbitals otherwise (see the module docs).
pub fn ccsd(
    hf: &ScfResult,
    ints: &AoIntegrals,
    frozen: usize,
    options: &CcOptions,
) -> Result<Ccsd> {
    Ok(run_ccsd(hf, ints, frozen, options, false)?.0)
}

/// CCSD and its perturbative triples, for canonical references: a closed
/// shell of RHF (spatial orbitals) or UHF (spin orbitals). ROHF of an open
/// shell is not canonical and is refused.
pub fn ccsd_t(
    hf: &ScfResult,
    ints: &AoIntegrals,
    frozen: usize,
    options: &CcOptions,
) -> Result<CcsdT> {
    let r = Reference::new(hf, ints, frozen)?;
    if (r.restricted && !r.closed_shell()) || !r.canonical() {
        return Err(QsoError::invalid(
            "scf",
            "(T) takes canonical orbitals: RHF of a closed shell or UHF",
        ));
    }
    let (ccsd, solved) = run_ccsd(hf, ints, frozen, options, false)?;
    let triples = match &solved {
        Solved::Closed(c, t) => c.triples(t),
        Solved::Spin(s, t) => s.triples(t),
    };
    Ok(CcsdT {
        energy: ccsd.energy + triples,
        ccsd,
        triples,
    })
}

#[cfg(test)]
mod tests;
