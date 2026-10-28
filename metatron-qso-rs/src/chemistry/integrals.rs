//! Molecules and their integrals over contracted Cartesian Gaussians.
//!
//! A [`Molecule`] lists nuclei (charge and position in bohr), the total
//! charge and the spin multiplicity. [`AoIntegrals::compute`] evaluates the
//! overlap, kinetic, nuclear-attraction and electron-repulsion integrals of
//! a [`Basis`] by the McMurchie–Davidson scheme (Helgaker, Jørgensen and
//! Olsen, *Molecular Electronic-Structure Theory*, ch. 9): products of
//! Gaussians are expanded in Hermite Gaussians, whose Coulomb integrals
//! follow from the Boys function.
//!
//! ```
//! use metatron_qso::chemistry::integrals::{AoIntegrals, Basis, Molecule};
//!
//! let h2 = Molecule::from_atoms("H 0 0 0; H 0 0 0.735", 0, 1).unwrap();
//! let ints = AoIntegrals::compute(&h2, Basis::Sto3g).unwrap();
//! assert_eq!(ints.n_functions(), 2);
//! assert!((ints.overlap(0, 0) - 1.0).abs() < 1e-12);
//! ```

use serde::{Deserialize, Serialize};

use super::basis_data::{CC_PVDZ, ELEMENTS, SIX_31G, SIX_31G_STAR, STO_3G, ShellData};
use crate::error::{QsoError, Result};
use crate::parallel;

/// Bohr per ångström (CODATA 2018). PySCF 2.14 uses the CODATA 2010 bohr,
/// 0.52917721092 Å, which moves geometries given in ångström by 3·10⁻¹¹
/// relative (energies by about 10⁻¹⁰ Ha).
pub const BOHR_PER_ANGSTROM: f64 = 1.0 / 0.529_177_210_903;

/// Most basis functions: the electron-repulsion tensor has `n⁴` entries.
pub const MAX_BASIS_FUNCTIONS: usize = 64;
/// Most Cartesian components behind spherical basis functions (a d shell
/// has six for its five functions); their integrals are computed first.
pub const MAX_CARTESIAN_FUNCTIONS: usize = 80;
/// Largest angular momentum of a shell (g functions).
pub const MAX_SHELL_L: u32 = 4;

const SYMBOLS: [&str; 18] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar",
];

/// A molecule: nuclei, total charge and spin multiplicity `2S + 1`
/// (checked on construction and deserialisation).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "MoleculeData")]
pub struct Molecule {
    charges: Vec<u32>,
    positions: Vec<[f64; 3]>,
    charge: i32,
    multiplicity: u32,
}

#[derive(Deserialize)]
struct MoleculeData {
    charges: Vec<u32>,
    positions: Vec<[f64; 3]>,
    charge: i32,
    multiplicity: u32,
}

impl TryFrom<MoleculeData> for Molecule {
    type Error = QsoError;

    fn try_from(d: MoleculeData) -> Result<Self> {
        Molecule::new(d.charges, d.positions, d.charge, d.multiplicity)
    }
}

impl Molecule {
    /// Atomic numbers.
    pub fn charges(&self) -> &[u32] {
        &self.charges
    }

    /// Positions in bohr.
    pub fn positions(&self) -> &[[f64; 3]] {
        &self.positions
    }

    /// Total charge.
    pub fn charge(&self) -> i32 {
        self.charge
    }

    /// Spin multiplicity `2S + 1`.
    pub fn multiplicity(&self) -> u32 {
        self.multiplicity
    }

    /// A molecule from `"H 0 0 0; H 0 0 0.735"` (element and position in
    /// ångström per atom, separated by `;` or new lines), as in PySCF.
    pub fn from_atoms(text: &str, charge: i32, multiplicity: u32) -> Result<Self> {
        let mut charges = Vec::new();
        let mut positions = Vec::new();
        for item in text
            .split([';', '\n'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let fields: Vec<&str> = item.split_whitespace().collect();
            if fields.len() != 4 {
                return Err(QsoError::invalid(
                    "atoms",
                    format!("'{item}' is not 'element x y z'"),
                ));
            }
            let z = SYMBOLS
                .iter()
                .position(|s| s.eq_ignore_ascii_case(fields[0]))
                .ok_or_else(|| {
                    QsoError::invalid(
                        "atoms",
                        format!("element '{}' is not one of H to Ar", fields[0]),
                    )
                })?;
            let mut r = [0.0; 3];
            for (k, f) in fields[1..].iter().enumerate() {
                let v: f64 = f
                    .parse()
                    .map_err(|_| QsoError::invalid("atoms", format!("'{f}' is not a number")))?;
                r[k] = v * BOHR_PER_ANGSTROM;
            }
            charges.push(z as u32 + 1);
            positions.push(r);
        }
        Self::new(charges, positions, charge, multiplicity)
    }

    /// A molecule from atomic numbers (1 to 18) and positions in bohr.
    pub fn new(
        charges: Vec<u32>,
        positions: Vec<[f64; 3]>,
        charge: i32,
        multiplicity: u32,
    ) -> Result<Self> {
        if charges.is_empty() || charges.len() != positions.len() {
            return Err(QsoError::invalid(
                "atoms",
                "at least one atom, with one position per atom",
            ));
        }
        if charges.iter().any(|&z| !(1..=ELEMENTS as u32).contains(&z)) {
            return Err(QsoError::invalid(
                "atoms",
                "atomic numbers must lie in 1..=18 (H to Ar)",
            ));
        }
        if positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err(QsoError::invalid("atoms", "positions must be finite"));
        }
        for i in 0..positions.len() {
            for j in 0..i {
                if distance(positions[i], positions[j]) < 1e-6 {
                    return Err(QsoError::invalid(
                        "atoms",
                        format!("atoms {j} and {i} are at the same position"),
                    ));
                }
            }
        }
        let electrons = charges.iter().map(|&z| z as i64).sum::<i64>() - charge as i64;
        if electrons < 0 {
            return Err(QsoError::invalid(
                "charge",
                "more positive charge than electrons",
            ));
        }
        if multiplicity == 0
            || multiplicity as i64 - 1 > electrons
            || (electrons - (multiplicity as i64 - 1)) % 2 != 0
        {
            return Err(QsoError::invalid(
                "multiplicity",
                format!("multiplicity {multiplicity} is impossible with {electrons} electrons"),
            ));
        }
        Ok(Self {
            charges,
            positions,
            charge,
            multiplicity,
        })
    }

    /// Number of electrons.
    pub fn n_electrons(&self) -> usize {
        (self.charges.iter().map(|&z| z as i64).sum::<i64>() - self.charge as i64) as usize
    }

    /// `(n_alpha, n_beta)`.
    pub fn n_alpha_beta(&self) -> (usize, usize) {
        let n = self.n_electrons();
        let unpaired = self.multiplicity as usize - 1;
        ((n + unpaired) / 2, (n - unpaired) / 2)
    }

    /// Nuclear repulsion energy `Σ_{i<j} Z_i Z_j / r_ij` (hartree).
    pub fn nuclear_repulsion(&self) -> f64 {
        let mut e = 0.0;
        for i in 0..self.charges.len() {
            for j in 0..i {
                e += (self.charges[i] * self.charges[j]) as f64
                    / distance(self.positions[i], self.positions[j]);
            }
        }
        e
    }
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// A basis set (H to Ar).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Basis {
    /// STO-3G (minimal).
    Sto3g,
    /// 6-31G (split valence).
    SixThirtyOneG,
    /// 6-31G* (6-31G(d)): 6-31G with d functions on Li to Ar.
    SixThirtyOneGStar,
    /// cc-pVDZ (correlation consistent, double zeta): p functions on H
    /// and He, d functions on Li to Ar.
    CcPvdz,
}

impl Basis {
    /// `"sto-3g"`, `"6-31g"`, `"6-31g*"` (or `"6-31g(d)"`) or `"cc-pvdz"`
    /// (case-insensitive).
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().replace(['_', ' '], "-").as_str() {
            "sto-3g" | "sto3g" => Ok(Basis::Sto3g),
            "6-31g" | "631g" => Ok(Basis::SixThirtyOneG),
            "6-31g*" | "631g*" | "6-31g(d)" => Ok(Basis::SixThirtyOneGStar),
            "cc-pvdz" | "ccpvdz" => Ok(Basis::CcPvdz),
            _ => Err(QsoError::invalid(
                "basis",
                format!("unknown basis '{name}' (expected sto-3g, 6-31g, 6-31g* or cc-pvdz)"),
            )),
        }
    }

    /// The name `from_name` reads.
    pub fn name(self) -> &'static str {
        match self {
            Basis::Sto3g => "sto-3g",
            Basis::SixThirtyOneG => "6-31g",
            Basis::SixThirtyOneGStar => "6-31g*",
            Basis::CcPvdz => "cc-pvdz",
        }
    }

    fn shells(self, z: u32) -> &'static [ShellData] {
        let z = z as usize - 1;
        match self {
            Basis::Sto3g => STO_3G[z],
            Basis::SixThirtyOneG => SIX_31G[z],
            Basis::SixThirtyOneGStar => SIX_31G_STAR[z],
            Basis::CcPvdz => CC_PVDZ[z],
        }
    }
}

/// The angular functions of shells with `l ≥ 2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Harmonics {
    /// Real solid harmonics, `2l + 1` per shell, ordered `m = −l, …, l` (p
    /// shells `x, y, z`), as PySCF's default and as cc-pVDZ is defined.
    #[default]
    Spherical,
    /// Cartesian monomials `x^a y^b z^c`, `(l + 1)(l + 2)/2` per shell, in
    /// PySCF's order (`xx, xy, xz, yy, yz, zz`), each normalised to one
    /// (PySCF normalises the `d` and higher components differently).
    Cartesian,
}

/// The exponents `(a, b, c)` of the Cartesian components of a shell, in
/// PySCF's order.
fn cartesian_powers(l: u32) -> Vec<[u32; 3]> {
    let mut out = Vec::new();
    for a in (0..=l).rev() {
        for b in (0..=l - a).rev() {
            out.push([a, b, l - a - b]);
        }
    }
    out
}

fn binomial(n: i64, k: i64) -> f64 {
    if k < 0 || k > n {
        return 0.0;
    }
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// The real solid harmonic `S_lm` (unnormalised) as coefficients of the
/// Cartesian monomials of [`cartesian_powers`] (Helgaker, Jørgensen and
/// Olsen, eq. 6.4.47; the signs of PySCF's functions).
fn solid_harmonic(l: u32, m: i32) -> Vec<f64> {
    let powers = cartesian_powers(l);
    let mut c = vec![0.0; powers.len()];
    let (l, am) = (l as i64, m.unsigned_abs() as i64);
    // v runs over integers (m ≥ 0) or half-integers (m < 0): w = 2v.
    let w0 = if m < 0 { 1 } else { 0 };
    for t in 0..=(l - am) / 2 {
        for u in 0..=t {
            let mut w = w0;
            while w <= am {
                let sign = if (t + (w - w0) / 2) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                };
                let value = sign
                    * 0.25f64.powi(t as i32)
                    * binomial(l, t)
                    * binomial(l - t, am + t)
                    * binomial(t, u)
                    * binomial(am, w);
                let x = 2 * t + am - 2 * u - w;
                let y = 2 * u + w;
                let z = l - 2 * t - am;
                let k = powers
                    .iter()
                    .position(|p| p == &[x as u32, y as u32, z as u32])
                    .expect("a monomial of degree l");
                c[k] += value;
                w += 2;
            }
        }
    }
    c
}

/// One contracted Cartesian Gaussian `x^l y^m z^n Σ_k c_k e^{−α_k r²}`.
#[derive(Clone, Debug)]
struct Function {
    center: [f64; 3],
    powers: [u32; 3],
    exponents: Vec<f64>,
    /// Coefficients including the normalisation of the primitives and of
    /// the contraction.
    coefficients: Vec<f64>,
}

fn double_factorial(n: i64) -> f64 {
    if n <= 0 {
        1.0
    } else {
        (1..=n).rev().step_by(2).map(|k| k as f64).product()
    }
}

/// Norm of the primitive `x^l y^m z^n e^{−α r²}`.
fn primitive_norm(alpha: f64, p: [u32; 3]) -> f64 {
    let l = (p[0] + p[1] + p[2]) as i32;
    let num = (2.0 * alpha / std::f64::consts::PI).powf(0.75) * (4.0 * alpha).powf(l as f64 / 2.0);
    let den = (double_factorial(2 * p[0] as i64 - 1)
        * double_factorial(2 * p[1] as i64 - 1)
        * double_factorial(2 * p[2] as i64 - 1))
    .sqrt();
    num / den
}

impl Function {
    fn new(center: [f64; 3], powers: [u32; 3], prims: &[(f64, f64)]) -> Self {
        let exponents: Vec<f64> = prims.iter().map(|p| p.0).collect();
        let mut coefficients: Vec<f64> = prims
            .iter()
            .map(|&(a, c)| c * primitive_norm(a, powers))
            .collect();
        // Normalise the contraction: ⟨φ|φ⟩ = 1.
        let mut s = 0.0;
        for i in 0..exponents.len() {
            for j in 0..exponents.len() {
                s += coefficients[i]
                    * coefficients[j]
                    * overlap_primitive(exponents[i], powers, center, exponents[j], powers, center);
            }
        }
        let scale = s.sqrt().recip();
        coefficients.iter_mut().for_each(|c| *c *= scale);
        Self {
            center,
            powers,
            exponents,
            coefficients,
        }
    }
}

/// Hermite expansion coefficient `E^{ij}_t` of a 1-D Gaussian product
/// (exponents `a`, `b`, separation `qx = A − B`).
fn hermite_e(i: i32, j: i32, t: i32, qx: f64, a: f64, b: f64) -> f64 {
    let p = a + b;
    let q = a * b / p;
    if t < 0 || t > i + j || i < 0 || j < 0 {
        0.0
    } else if i == 0 && j == 0 && t == 0 {
        (-q * qx * qx).exp()
    } else if j == 0 {
        hermite_e(i - 1, j, t - 1, qx, a, b) / (2.0 * p)
            - q * qx / a * hermite_e(i - 1, j, t, qx, a, b)
            + (t + 1) as f64 * hermite_e(i - 1, j, t + 1, qx, a, b)
    } else {
        hermite_e(i, j - 1, t - 1, qx, a, b) / (2.0 * p)
            + q * qx / b * hermite_e(i, j - 1, t, qx, a, b)
            + (t + 1) as f64 * hermite_e(i, j - 1, t + 1, qx, a, b)
    }
}

fn overlap_primitive(
    a: f64,
    pa: [u32; 3],
    ca: [f64; 3],
    b: f64,
    pb: [u32; 3],
    cb: [f64; 3],
) -> f64 {
    let p = a + b;
    (0..3)
        .map(|k| hermite_e(pa[k] as i32, pb[k] as i32, 0, ca[k] - cb[k], a, b))
        .product::<f64>()
        * (std::f64::consts::PI / p).powf(1.5)
}

fn kinetic_primitive(
    a: f64,
    pa: [u32; 3],
    ca: [f64; 3],
    b: f64,
    pb: [u32; 3],
    cb: [f64; 3],
) -> f64 {
    let s = |q: [i32; 3]| {
        if q.iter().any(|&v| v < 0) {
            return 0.0;
        }
        overlap_primitive(a, pa, ca, b, [q[0] as u32, q[1] as u32, q[2] as u32], cb)
    };
    let l = [pb[0] as i32, pb[1] as i32, pb[2] as i32];
    let term0 = b * (2.0 * (l[0] + l[1] + l[2]) as f64 + 3.0) * s(l);
    let mut term1 = 0.0;
    let mut term2 = 0.0;
    for k in 0..3 {
        let mut up = l;
        up[k] += 2;
        let mut down = l;
        down[k] -= 2;
        term1 += s(up);
        term2 += (l[k] * (l[k] - 1)) as f64 * s(down);
    }
    term0 - 2.0 * b * b * term1 - 0.5 * term2
}

/// Boys function `F_n(T) = ∫₀¹ t^{2n} e^{−T t²} dt` for `n = 0..f.len()`.
fn boys(f: &mut [f64], t: f64) {
    let nmax = f.len() - 1;
    if t < 1e-14 {
        for (n, v) in f.iter_mut().enumerate() {
            *v = 1.0 / (2 * n + 1) as f64;
        }
        return;
    }
    if t > 60.0 {
        // Asymptotic: F_n = (2n−1)!! / 2^{n+1} √(π / T^{2n+1}); the
        // neglected terms are below e^{−60}.
        let mut v = 0.5 * (std::f64::consts::PI / t).sqrt();
        for (n, fv) in f.iter_mut().enumerate() {
            *fv = v;
            v *= (2 * n + 1) as f64 / (2.0 * t);
        }
        return;
    }
    // Series for F_nmax, then downward recursion (stable).
    let n = nmax as f64;
    let mut term = 1.0 / (2.0 * n + 1.0);
    let mut sum = term;
    let mut k = 0.0;
    while term > 1e-17 * sum {
        k += 1.0;
        term *= 2.0 * t / (2.0 * n + 2.0 * k + 1.0);
        sum += term;
    }
    let e = (-t).exp();
    f[nmax] = e * sum;
    for m in (0..nmax).rev() {
        f[m] = (2.0 * t * f[m + 1] + e) / (2 * m + 1) as f64;
    }
}

/// Entries of `E_t` per direction: `t ≤ 2 MAX_SHELL_L`.
const EDIM: usize = 2 * MAX_SHELL_L as usize + 1;

/// Hermite Coulomb integrals `R_{tuv}` for `t + u + v ≤ l`, at exponent
/// `p` and separation `pc`, in `out` at `(t·(l+1) + u)·(l+1) + v`
/// (`work` holds the intermediate orders).
fn hermite_r(l: usize, p: f64, pc: [f64; 3], out: &mut Vec<f64>, work: &mut Vec<f64>) {
    let d = l + 1;
    let at = |n: usize, t: usize, u: usize, v: usize| ((n * d + t) * d + u) * d + v;
    let r2 = pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2];
    let mut f = [0.0; 4 * MAX_SHELL_L as usize + 1];
    boys(&mut f[..=l], p * r2);
    work.clear();
    work.resize(d * d * d * d, 0.0);
    let r = work;
    let mut factor = 1.0;
    for (n, &fv) in f[..=l].iter().enumerate() {
        r[at(n, 0, 0, 0)] = factor * fv;
        factor *= -2.0 * p;
    }
    for total in 1..=l {
        for n in 0..=l - total {
            for t in 0..=total {
                for u in 0..=total - t {
                    let v = total - t - u;
                    let value = if t > 0 {
                        let prev = if t >= 2 {
                            (t - 1) as f64 * r[at(n + 1, t - 2, u, v)]
                        } else {
                            0.0
                        };
                        prev + pc[0] * r[at(n + 1, t - 1, u, v)]
                    } else if u > 0 {
                        let prev = if u >= 2 {
                            (u - 1) as f64 * r[at(n + 1, t, u - 2, v)]
                        } else {
                            0.0
                        };
                        prev + pc[1] * r[at(n + 1, t, u - 1, v)]
                    } else {
                        let prev = if v >= 2 {
                            (v - 1) as f64 * r[at(n + 1, t, u, v - 2)]
                        } else {
                            0.0
                        };
                        prev + pc[2] * r[at(n + 1, t, u, v - 1)]
                    };
                    r[at(n, t, u, v)] = value;
                }
            }
        }
    }
    out.clear();
    out.extend_from_slice(&r[..d * d * d]);
}

/// A product of two primitives: exponent, centre, contraction
/// coefficients and Hermite expansion `E_t` per direction.
#[derive(Clone, Debug)]
struct PrimPair {
    p: f64,
    center: [f64; 3],
    coef: f64,
    e: [[f64; EDIM]; 3],
}

/// The primitive products of two contracted functions.
#[derive(Clone, Debug)]
struct Pair {
    /// Angular momentum per direction (`E_t` for `t ≤ l[k]`).
    l: [usize; 3],
    prims: Vec<PrimPair>,
}

impl Pair {
    #[allow(clippy::needless_range_loop)]
    fn new(fi: &Function, fj: &Function) -> Self {
        let l = [0, 1, 2].map(|k| (fi.powers[k] + fj.powers[k]) as usize);
        let mut prims = Vec::with_capacity(fi.exponents.len() * fj.exponents.len());
        for (&a, &ca) in fi.exponents.iter().zip(&fi.coefficients) {
            for (&b, &cb) in fj.exponents.iter().zip(&fj.coefficients) {
                let p = a + b;
                let center = [0, 1, 2].map(|k| (a * fi.center[k] + b * fj.center[k]) / p);
                let mut e = [[0.0; EDIM]; 3];
                for k in 0..3 {
                    for t in 0..=l[k] {
                        e[k][t] = hermite_e(
                            fi.powers[k] as i32,
                            fj.powers[k] as i32,
                            t as i32,
                            fi.center[k] - fj.center[k],
                            a,
                            b,
                        );
                    }
                }
                prims.push(PrimPair {
                    p,
                    center,
                    coef: ca * cb,
                    e,
                });
            }
        }
        Self { l, prims }
    }

    fn total_l(&self) -> usize {
        self.l.iter().sum()
    }
}

/// `⟨i| 1/|r − C| |j⟩` over contracted functions.
#[allow(clippy::needless_range_loop)]
fn nuclear_pair(pair: &Pair, c: [f64; 3]) -> f64 {
    let l = pair.total_l();
    let d = l + 1;
    let (mut r, mut work) = (Vec::new(), Vec::new());
    let mut v = 0.0;
    for pp in &pair.prims {
        hermite_r(
            l,
            pp.p,
            [
                pp.center[0] - c[0],
                pp.center[1] - c[1],
                pp.center[2] - c[2],
            ],
            &mut r,
            &mut work,
        );
        let mut s = 0.0;
        for t in 0..=pair.l[0] {
            for u in 0..=pair.l[1] {
                for w in 0..=pair.l[2] {
                    s += pp.e[0][t] * pp.e[1][u] * pp.e[2][w] * r[(t * d + u) * d + w];
                }
            }
        }
        v += pp.coef * 2.0 * std::f64::consts::PI / pp.p * s;
    }
    v
}

/// `(ij|kl)` over contracted functions, from the pairs `ij` and `kl`.
#[allow(clippy::needless_range_loop)]
fn eri_pairs(ab: &Pair, cd: &Pair, r: &mut Vec<f64>, work: &mut Vec<f64>) -> f64 {
    let l = ab.total_l() + cd.total_l();
    let d = l + 1;
    let mut v = 0.0;
    for x in &ab.prims {
        for y in &cd.prims {
            let (p, q) = (x.p, y.p);
            let alpha = p * q / (p + q);
            hermite_r(
                l,
                alpha,
                [
                    x.center[0] - y.center[0],
                    x.center[1] - y.center[1],
                    x.center[2] - y.center[2],
                ],
                r,
                work,
            );
            let mut s = 0.0;
            for t in 0..=ab.l[0] {
                for u in 0..=ab.l[1] {
                    for w in 0..=ab.l[2] {
                        let left = x.e[0][t] * x.e[1][u] * x.e[2][w];
                        if left == 0.0 {
                            continue;
                        }
                        let mut inner = 0.0;
                        for tau in 0..=cd.l[0] {
                            for nu in 0..=cd.l[1] {
                                for phi in 0..=cd.l[2] {
                                    let sign = if (tau + nu + phi) % 2 == 1 { -1.0 } else { 1.0 };
                                    inner += sign
                                        * y.e[0][tau]
                                        * y.e[1][nu]
                                        * y.e[2][phi]
                                        * r[((t + tau) * d + u + nu) * d + w + phi];
                                }
                            }
                        }
                        s += left * inner;
                    }
                }
            }
            v += x.coef * y.coef * 2.0 * std::f64::consts::PI.powf(2.5) / (p * q * (p + q).sqrt())
                * s;
        }
    }
    v
}

/// Overlap, kinetic, nuclear-attraction and electron-repulsion integrals
/// over Cartesian functions (`n × n` and `n⁴`, row-major).
type Integrals = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn cartesian_integrals(molecule: &Molecule, functions: &[Function]) -> Integrals {
    let n = functions.len();
    // Primitive products of every pair i ≥ j, in the order of the
    // quartets below.
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();
    let pair_data: Vec<Pair> = pairs
        .iter()
        .map(|&(i, j)| Pair::new(&functions[i], &functions[j]))
        .collect();
    let mut overlap = vec![0.0; n * n];
    let mut kinetic = vec![0.0; n * n];
    let mut nuclear = vec![0.0; n * n];
    for (x, &(i, j)) in pairs.iter().enumerate() {
        let (fi, fj) = (&functions[i], &functions[j]);
        let mut sv = 0.0;
        let mut tv = 0.0;
        for a in 0..fi.exponents.len() {
            for b in 0..fj.exponents.len() {
                let c = fi.coefficients[a] * fj.coefficients[b];
                let (ea, eb) = (fi.exponents[a], fj.exponents[b]);
                sv += c * overlap_primitive(ea, fi.powers, fi.center, eb, fj.powers, fj.center);
                tv += c * kinetic_primitive(ea, fi.powers, fi.center, eb, fj.powers, fj.center);
            }
        }
        let vv: f64 = molecule
            .charges
            .iter()
            .zip(&molecule.positions)
            .map(|(&z, &c)| -(z as f64) * nuclear_pair(&pair_data[x], c))
            .sum();
        for (m, v) in [(&mut overlap, sv), (&mut kinetic, tv), (&mut nuclear, vv)] {
            m[i * n + j] = v;
            m[j * n + i] = v;
        }
    }
    // Unique quartets i ≥ j, k ≥ l, ij ≥ kl, in parallel.
    let quartets: Vec<(usize, usize)> = (0..pairs.len())
        .flat_map(|x| (0..=x).map(move |y| (x, y)))
        .collect();
    let chunk = 64;
    let values: Vec<f64> = parallel::map_tasks(quartets.len().div_ceil(chunk), |c| {
        let (mut r, mut work) = (Vec::new(), Vec::new());
        (c * chunk..((c + 1) * chunk).min(quartets.len()))
            .map(|k| {
                eri_pairs(
                    &pair_data[quartets[k].0],
                    &pair_data[quartets[k].1],
                    &mut r,
                    &mut work,
                )
            })
            .collect::<Vec<f64>>()
    })
    .into_iter()
    .flatten()
    .collect();
    let mut eri = vec![0.0; n * n * n * n];
    for (&(x, y), &v) in quartets.iter().zip(&values) {
        let ((i, j), (k, l)) = (pairs[x], pairs[y]);
        for (p, q, r, s) in [
            (i, j, k, l),
            (j, i, k, l),
            (i, j, l, k),
            (j, i, l, k),
            (k, l, i, j),
            (l, k, i, j),
            (k, l, j, i),
            (l, k, j, i),
        ] {
            eri[((p * n + q) * n + r) * n + s] = v;
        }
    }
    (overlap, kinetic, nuclear, eri)
}

/// Integrals over the atomic orbitals of a molecule.
#[derive(Clone, Debug, PartialEq)]
pub struct AoIntegrals {
    n: usize,
    overlap: Vec<f64>,
    kinetic: Vec<f64>,
    nuclear: Vec<f64>,
    /// `(ij|kl)` in chemists' notation, `((i·n + j)·n + k)·n + l`.
    eri: Vec<f64>,
    /// Atom of every basis function.
    atoms: Vec<usize>,
    /// Angular momentum of every basis function.
    angular: Vec<u32>,
    /// The basis and the nuclei the integrals belong to.
    basis: Basis,
    harmonics: Harmonics,
    charges: Vec<u32>,
    positions: Vec<[f64; 3]>,
}

/// `out[L, i, j, k] = Σ_l a[i, j, k, l] T[l, L]` for a sparse `T` given by
/// its columns: transforms the last index and moves it to the front.
fn quarter_transform(a: &[f64], dims: [usize; 4], columns: &[Vec<(usize, f64)>]) -> Vec<f64> {
    let [d0, d1, d2, d3] = dims;
    let inner = d0 * d1 * d2;
    let mut out = vec![0.0; columns.len() * inner];
    for (big, col) in columns.iter().enumerate() {
        let dst = &mut out[big * inner..(big + 1) * inner];
        for &(l, c) in col {
            for (x, v) in dst.iter_mut().enumerate() {
                *v += c * a[x * d3 + l];
            }
        }
    }
    out
}

impl AoIntegrals {
    /// The integrals of `molecule` in `basis`, with spherical `d` and
    /// higher shells ([`AoIntegrals::compute_with`]).
    pub fn compute(molecule: &Molecule, basis: Basis) -> Result<Self> {
        Self::compute_with(molecule, basis, Harmonics::Spherical)
    }

    /// The integrals of `molecule` in `basis`. Functions are ordered by
    /// atom, then by shell as the basis lists them, and within a shell as
    /// [`Harmonics`] says (`p` functions as `x, y, z`): the order of PySCF.
    pub fn compute_with(molecule: &Molecule, basis: Basis, harmonics: Harmonics) -> Result<Self> {
        Self::from_shells(molecule, basis, harmonics, &|z| basis.shells(z).to_vec())
    }

    /// The integrals with the shells `shells(z)` on every atom of atomic
    /// number `z` (`basis` only labels them).
    pub(crate) fn from_shells(
        molecule: &Molecule,
        basis: Basis,
        harmonics: Harmonics,
        shells_of: &dyn Fn(u32) -> Vec<ShellData>,
    ) -> Result<Self> {
        // Cartesian components of every shell: (first component, l, atom).
        let mut functions = Vec::new();
        let mut shells = Vec::new();
        for (atom, (&z, &pos)) in molecule.charges.iter().zip(&molecule.positions).enumerate() {
            for (l, prims) in shells_of(z) {
                let l = l as u32;
                debug_assert!(l <= MAX_SHELL_L);
                shells.push((functions.len(), l, atom));
                for p in cartesian_powers(l) {
                    functions.push(Function::new(pos, p, prims));
                }
            }
        }
        let spherical = |l: u32| harmonics == Harmonics::Spherical && l >= 2;
        let n: usize = shells
            .iter()
            .map(|&(_, l, _)| {
                if spherical(l) {
                    2 * l as usize + 1
                } else {
                    ((l + 1) * (l + 2) / 2) as usize
                }
            })
            .sum();
        let nc = functions.len();
        if n > MAX_BASIS_FUNCTIONS {
            return Err(QsoError::invalid(
                "molecule",
                format!("{n} basis functions exceed the maximum {MAX_BASIS_FUNCTIONS}"),
            ));
        }
        if nc > MAX_CARTESIAN_FUNCTIONS {
            return Err(QsoError::invalid(
                "molecule",
                format!("{nc} Cartesian components exceed the maximum {MAX_CARTESIAN_FUNCTIONS}"),
            ));
        }
        let (overlap, kinetic, nuclear, eri) = cartesian_integrals(molecule, &functions);
        // Every final function as a combination of Cartesian components.
        let mut columns: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        let mut atoms = Vec::with_capacity(n);
        let mut angular = Vec::with_capacity(n);
        for &(first, l, atom) in &shells {
            if spherical(l) {
                let k = ((l + 1) * (l + 2) / 2) as usize;
                // The monomial x^a y^b z^c is the unit-normalised component
                // times √((2a−1)!!(2b−1)!!(2c−1)!!), up to a common factor.
                let monomial: Vec<f64> = cartesian_powers(l)
                    .iter()
                    .map(|p| {
                        p.iter()
                            .map(|&e| double_factorial(2 * e as i64 - 1))
                            .product::<f64>()
                            .sqrt()
                    })
                    .collect();
                for m in -(l as i32)..=l as i32 {
                    let c: Vec<f64> = solid_harmonic(l, m)
                        .iter()
                        .zip(&monomial)
                        .map(|(c, f)| c * f)
                        .collect();
                    let mut norm = 0.0;
                    for a in 0..k {
                        for b in 0..k {
                            norm += c[a] * c[b] * overlap[(first + a) * nc + first + b];
                        }
                    }
                    let scale = norm.sqrt().recip();
                    columns.push(
                        c.iter()
                            .enumerate()
                            .filter(|(_, v)| **v != 0.0)
                            .map(|(a, v)| (first + a, v * scale))
                            .collect(),
                    );
                    atoms.push(atom);
                    angular.push(l);
                }
            } else {
                for a in 0..((l + 1) * (l + 2) / 2) as usize {
                    columns.push(vec![(first + a, 1.0)]);
                    atoms.push(atom);
                    angular.push(l);
                }
            }
        }
        let identity = n == nc;
        let two = |m: &[f64]| -> Vec<f64> {
            if identity {
                return m.to_vec();
            }
            let mut out = vec![0.0; n * n];
            for (i, ci) in columns.iter().enumerate() {
                for (j, cj) in columns.iter().enumerate() {
                    let mut v = 0.0;
                    for &(a, x) in ci {
                        for &(b, y) in cj {
                            v += x * y * m[a * nc + b];
                        }
                    }
                    out[i * n + j] = v;
                }
            }
            out
        };
        let (overlap, kinetic, nuclear) = (two(&overlap), two(&kinetic), two(&nuclear));
        let eri = if identity {
            eri
        } else {
            let a = quarter_transform(&eri, [nc, nc, nc, nc], &columns);
            let a = quarter_transform(&a, [n, nc, nc, nc], &columns);
            let a = quarter_transform(&a, [n, n, nc, nc], &columns);
            quarter_transform(&a, [n, n, n, nc], &columns)
        };
        Ok(Self {
            n,
            overlap,
            kinetic,
            nuclear,
            eri,
            atoms,
            angular,
            basis,
            harmonics,
            charges: molecule.charges.clone(),
            positions: molecule.positions.clone(),
        })
    }

    /// Spherical or Cartesian `d` and higher shells.
    pub fn harmonics(&self) -> Harmonics {
        self.harmonics
    }

    /// The basis set.
    pub fn basis(&self) -> Basis {
        self.basis
    }

    /// Atomic numbers of the nuclei.
    pub fn charges(&self) -> &[u32] {
        &self.charges
    }

    /// Whether these are the integrals of `molecule` (same nuclei at the
    /// same positions).
    pub fn belongs_to(&self, molecule: &Molecule) -> bool {
        self.charges == molecule.charges && self.positions == molecule.positions
    }

    pub fn n_functions(&self) -> usize {
        self.n
    }

    pub fn overlap(&self, i: usize, j: usize) -> f64 {
        self.overlap[i * self.n + j]
    }

    pub fn kinetic(&self, i: usize, j: usize) -> f64 {
        self.kinetic[i * self.n + j]
    }

    /// Nuclear attraction `⟨i| −Σ_A Z_A/|r − R_A| |j⟩`.
    pub fn nuclear(&self, i: usize, j: usize) -> f64 {
        self.nuclear[i * self.n + j]
    }

    /// Core Hamiltonian `T + V`.
    pub fn core(&self, i: usize, j: usize) -> f64 {
        self.kinetic(i, j) + self.nuclear(i, j)
    }

    /// `(ij|kl)` in chemists' notation.
    pub fn eri(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        self.eri[((i * self.n + j) * self.n + k) * self.n + l]
    }

    /// Atom of every basis function.
    pub fn atoms(&self) -> &[usize] {
        &self.atoms
    }

    /// Angular momentum of every basis function.
    pub fn angular_momenta(&self) -> &[u32] {
        &self.angular
    }

    /// All `(ij|kl)`, at `((i·n + j)·n + k)·n + l`.
    pub fn eri_tensor(&self) -> &[f64] {
        &self.eri
    }
}

#[cfg(test)]
mod tests;
