//! Molecules and their integrals over contracted Gaussians (spherical or
//! Cartesian shells up to g).
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
pub(crate) const MAX_CARTESIAN_FUNCTIONS: usize = 80;
/// Largest angular momentum of a shell the integrals take (g functions;
/// the bundled bases stop at d).
pub(crate) const MAX_SHELL_L: u32 = 4;

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
    if t > 60.0 + 4.0 * nmax as f64 {
        // Asymptotic: F_n = (2n−1)!! / 2^{n+1} √(π / T^{2n+1}); the
        // neglected terms are below e^{−T} T^{n} relative to it, under
        // 10⁻²⁰ from this threshold on (n ≤ 16).
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

/// The primitive products of two shells (first component and angular
/// momentum each), with the Hermite expansion `E^{ij}_t` of every pair of
/// exponents `i ≤ l_a`, `j ≤ l_b` per direction.
struct ShellPair {
    /// First function, angular momentum and number of functions of each
    /// group of shells.
    a: (usize, usize, usize),
    b: (usize, usize, usize),
    prims: Vec<ShellPrim>,
}

struct ShellPrim {
    p: f64,
    center: [f64; 3],
    /// Primitive indices in the two shells.
    pa: usize,
    pb: usize,
    /// `e[((k·(l_a+1) + i)·(l_b+1) + j)·(l_a+l_b+1) + t]`.
    e: Vec<f64>,
}

impl ShellPair {
    fn new(functions: &[Function], a: (usize, usize, usize), b: (usize, usize, usize)) -> Self {
        let (fa, fb) = (&functions[a.0], &functions[b.0]);
        let (la, lb) = (a.1, b.1);
        let mut prims = Vec::with_capacity(fa.exponents.len() * fb.exponents.len());
        for (pa, &x) in fa.exponents.iter().enumerate() {
            for (pb, &y) in fb.exponents.iter().enumerate() {
                let p = x + y;
                let center = [0, 1, 2].map(|k| (x * fa.center[k] + y * fb.center[k]) / p);
                let mut e = vec![0.0; 3 * (la + 1) * (lb + 1) * (la + lb + 1)];
                for k in 0..3 {
                    for i in 0..=la {
                        for j in 0..=lb {
                            for t in 0..=i + j {
                                e[((k * (la + 1) + i) * (lb + 1) + j) * (la + lb + 1) + t] =
                                    hermite_e(
                                        i as i32,
                                        j as i32,
                                        t as i32,
                                        fa.center[k] - fb.center[k],
                                        x,
                                        y,
                                    );
                            }
                        }
                    }
                }
                prims.push(ShellPrim {
                    p,
                    center,
                    pa,
                    pb,
                    e,
                });
            }
        }
        ShellPair { a, b, prims }
    }

    fn components(&self) -> (usize, usize) {
        (self.a.2, self.b.2)
    }
}

/// `(ab|cd)` for every component of four shells, at
/// `((a·n_b + b)·n_c + c)·n_d + d`: per pair of primitive products one
/// table `R_{tuv}`, contracted first with the right pair's expansion for
/// each of its components, then with the left one's.
fn eri_shells(
    functions: &[Function],
    ab: &ShellPair,
    cd: &ShellPair,
    r: &mut Vec<f64>,
    work: &mut Vec<f64>,
    w: &mut Vec<f64>,
) -> Vec<f64> {
    let (na, nb) = ab.components();
    let (nc, nd) = cd.components();
    let ncd = nc * nd;
    let (la, lb, lc, ld) = (ab.a.1, ab.b.1, cd.a.1, cd.b.1);
    let (lab, lcd) = (la + lb, lc + ld);
    let l = lab + lcd;
    let d = l + 1;
    let dw = lab + 1;
    let block = dw * dw * dw;
    let mut out = vec![0.0; na * nb * ncd];
    let at = |pair_l: (usize, usize), e: &[f64], k: usize, i: usize, j: usize, t: usize| {
        e[((k * (pair_l.0 + 1) + i) * (pair_l.1 + 1) + j) * (pair_l.0 + pair_l.1 + 1) + t]
    };
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
            let pref = 2.0 * std::f64::consts::PI.powf(2.5) / (p * q * (p + q).sqrt());
            // W[cd][t][u][v] = c_cd Σ (−1)^{τ+ν+φ} E_τ E_ν E_φ R_{t+τ, u+ν, v+φ}.
            w.clear();
            w.resize(ncd * block, 0.0);
            let mut any = false;
            for kc in 0..nc {
                let fc = &functions[cd.a.0 + kc];
                for kd in 0..nd {
                    let fd = &functions[cd.b.0 + kd];
                    let coef = fc.coefficients[y.pa] * fd.coefficients[y.pb];
                    if coef == 0.0 {
                        continue;
                    }
                    any = true;
                    let (c, dd) = (fc.powers, fd.powers);
                    let m = [0, 1, 2].map(|k| (c[k] + dd[k]) as usize);
                    let wb = &mut w[(kc * nd + kd) * block..(kc * nd + kd + 1) * block];
                    for t in 0..=lab {
                        for u in 0..=lab - t {
                            for v in 0..=lab - t - u {
                                let mut sum = 0.0;
                                for tau in 0..=m[0] {
                                    let ex =
                                        at((lc, ld), &y.e, 0, c[0] as usize, dd[0] as usize, tau);
                                    if ex == 0.0 {
                                        continue;
                                    }
                                    for nu in 0..=m[1] {
                                        let ey = ex
                                            * at(
                                                (lc, ld),
                                                &y.e,
                                                1,
                                                c[1] as usize,
                                                dd[1] as usize,
                                                nu,
                                            );
                                        for phi in 0..=m[2] {
                                            let ez = ey
                                                * at(
                                                    (lc, ld),
                                                    &y.e,
                                                    2,
                                                    c[2] as usize,
                                                    dd[2] as usize,
                                                    phi,
                                                );
                                            let sign =
                                                if (tau + nu + phi) % 2 == 1 { -1.0 } else { 1.0 };
                                            sum += sign
                                                * ez
                                                * r[((t + tau) * d + u + nu) * d + v + phi];
                                        }
                                    }
                                }
                                wb[(t * dw + u) * dw + v] = coef * sum;
                            }
                        }
                    }
                }
            }
            if !any {
                continue;
            }
            for ka in 0..na {
                let fa = &functions[ab.a.0 + ka];
                for kb in 0..nb {
                    let fb = &functions[ab.b.0 + kb];
                    let coef = pref * fa.coefficients[x.pa] * fb.coefficients[x.pb];
                    if coef == 0.0 {
                        continue;
                    }
                    let (a, b) = (fa.powers, fb.powers);
                    let m = [0, 1, 2].map(|k| (a[k] + b[k]) as usize);
                    let row = &mut out[(ka * nb + kb) * ncd..(ka * nb + kb + 1) * ncd];
                    for t in 0..=m[0] {
                        let ex = at((la, lb), &x.e, 0, a[0] as usize, b[0] as usize, t);
                        if ex == 0.0 {
                            continue;
                        }
                        for u in 0..=m[1] {
                            let ey = ex * at((la, lb), &x.e, 1, a[1] as usize, b[1] as usize, u);
                            for v in 0..=m[2] {
                                let ez = coef
                                    * ey
                                    * at((la, lb), &x.e, 2, a[2] as usize, b[2] as usize, v);
                                if ez == 0.0 {
                                    continue;
                                }
                                let k = (t * dw + u) * dw + v;
                                for (cdk, o) in row.iter_mut().enumerate() {
                                    *o += ez * w[cdk * block + k];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Schwarz factors `Q_ab = √max |(ab|ab)|` of every pair of groups of
/// shells, the maximum over their components.
fn schwarz_factors(functions: &[Function], pairs: &[ShellPair]) -> Vec<f64> {
    parallel::map_tasks(pairs.len(), |x| {
        let (mut r, mut work, mut w) = (Vec::new(), Vec::new(), Vec::new());
        let ab = &pairs[x];
        let block = eri_shells(functions, ab, ab, &mut r, &mut work, &mut w);
        let (na, nb) = ab.components();
        let mut max: f64 = 0.0;
        for ka in 0..na {
            for kb in 0..nb {
                let k = ka * nb + kb;
                max = max.max(block[k * na * nb + k].abs());
            }
        }
        max.sqrt()
    })
}

/// All `(ij|kl)` over the Cartesian functions, from the unique quartets of
/// groups of shells (`shells`: first function, angular momentum and number
/// of functions; the columns of a general contraction share their
/// primitives and form one group), in parallel. A quartet with
/// `Q_ab·Q_cd < threshold` is skipped (its integrals set to zero): by the
/// Cauchy–Schwarz inequality `|(ab|cd)| ≤ √((ab|ab)(cd|cd))`, each of its
/// integrals lies below the threshold. Returns the integrals and the
/// number of quartets skipped and computed.
fn repulsion(
    functions: &[Function],
    shells: &[(usize, usize, usize)],
    threshold: f64,
) -> (Vec<f64>, usize, usize) {
    let n = functions.len();
    let pairs: Vec<ShellPair> = (0..shells.len())
        .flat_map(|i| (0..=i).map(move |j| (i, j)))
        .map(|(i, j)| ShellPair::new(functions, shells[i], shells[j]))
        .collect();
    let q = if threshold > 0.0 {
        schwarz_factors(functions, &pairs)
    } else {
        vec![f64::INFINITY; pairs.len()]
    };
    let all = pairs.len() * (pairs.len() + 1) / 2;
    let quartets: Vec<(usize, usize)> = (0..pairs.len())
        .flat_map(|x| (0..=x).map(move |y| (x, y)))
        .filter(|&(x, y)| {
            // A NaN factor keeps the quartet.
            let bound = q[x] * q[y];
            bound.is_nan() || bound >= threshold
        })
        .collect();
    let chunk = 16;
    let blocks: Vec<Vec<f64>> = parallel::map_tasks(quartets.len().div_ceil(chunk), |c| {
        let (mut r, mut work, mut w) = (Vec::new(), Vec::new(), Vec::new());
        (c * chunk..((c + 1) * chunk).min(quartets.len()))
            .map(|k| {
                let (x, y) = quartets[k];
                eri_shells(functions, &pairs[x], &pairs[y], &mut r, &mut work, &mut w)
            })
            .collect::<Vec<_>>()
    })
    .into_iter()
    .flatten()
    .collect();
    let mut eri = vec![0.0; n * n * n * n];
    for (&(x, y), block) in quartets.iter().zip(&blocks) {
        let (ab, cd) = (&pairs[x], &pairs[y]);
        let (na, nb) = ab.components();
        let (nc, nd) = cd.components();
        let mut it = block.iter();
        for ka in 0..na {
            for kb in 0..nb {
                for kc in 0..nc {
                    for kd in 0..nd {
                        let v = *it.next().expect("a full block");
                        let (i, j, k, l) = (ab.a.0 + ka, ab.b.0 + kb, cd.a.0 + kc, cd.b.0 + kd);
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
                }
            }
        }
    }
    (eri, all - quartets.len(), quartets.len())
}

/// Overlap, kinetic, nuclear-attraction and electron-repulsion integrals
/// over Cartesian functions (`n × n` and `n⁴`, row-major), and the shell
/// quartets skipped and computed.
type Integrals = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, (usize, usize));

fn cartesian_integrals(
    molecule: &Molecule,
    functions: &[Function],
    shells: &[(usize, usize, usize)],
    threshold: f64,
) -> Integrals {
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
    let (eri, skipped, computed) = repulsion(functions, shells, threshold);
    (overlap, kinetic, nuclear, eri, (skipped, computed))
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
    /// Schwarz threshold, and shell quartets skipped and computed.
    screening: f64,
    quartets: (usize, usize),
}

/// Default Schwarz threshold of [`AoIntegrals::compute`] (Hartree): shell
/// quartets whose integrals are all bounded by it are skipped.
pub const SCREENING_THRESHOLD: f64 = 1e-12;

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
    /// Repulsion integrals are screened with [`SCREENING_THRESHOLD`].
    pub fn compute_with(molecule: &Molecule, basis: Basis, harmonics: Harmonics) -> Result<Self> {
        Self::compute_screened(molecule, basis, harmonics, SCREENING_THRESHOLD)
    }

    /// [`AoIntegrals::compute_with`] with Schwarz screening at `threshold`
    /// (Hartree; 0 computes every integral). A quartet of shells is
    /// skipped when `Q_ab·Q_cd < threshold`, with `Q_ab = √max |(ab|ab)|`
    /// over the Cartesian components of the pair: by the Cauchy–Schwarz
    /// inequality every integral of the quartet is smaller. Spherical
    /// functions combine Cartesian ones, so their integrals change by at
    /// most the threshold times the products of the combination's
    /// coefficients.
    pub fn compute_screened(
        molecule: &Molecule,
        basis: Basis,
        harmonics: Harmonics,
        threshold: f64,
    ) -> Result<Self> {
        if !(threshold >= 0.0 && threshold.is_finite()) {
            return Err(QsoError::invalid(
                "screening",
                format!("the screening threshold must be finite and non-negative, not {threshold}"),
            ));
        }
        Self::from_shells(molecule, basis, harmonics, threshold, &|z| {
            basis.shells(z).to_vec()
        })
    }

    /// The integrals with the shells `shells(z)` on every atom of atomic
    /// number `z` (`basis` only labels them).
    pub(crate) fn from_shells(
        molecule: &Molecule,
        basis: Basis,
        harmonics: Harmonics,
        threshold: f64,
        shells_of: &dyn Fn(u32) -> Vec<ShellData>,
    ) -> Result<Self> {
        // Cartesian components of every shell: (first component, l, atom).
        let mut functions = Vec::new();
        let mut shells = Vec::new();
        for (atom, (&z, &pos)) in molecule.charges.iter().zip(&molecule.positions).enumerate() {
            for (l, prims) in shells_of(z) {
                let l = l as u32;
                assert!(l <= MAX_SHELL_L, "shells up to g");
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
        // Groups of shells for the repulsion integrals: consecutive shells
        // of one atom with the same angular momentum and exponents (the
        // columns of a general contraction).
        let mut shell_list: Vec<(usize, usize, usize)> = Vec::new();
        let mut previous: Option<(usize, u32, usize)> = None;
        for &(first, l, atom) in &shells {
            let size = ((l + 1) * (l + 2) / 2) as usize;
            let same = previous.is_some_and(|(pfirst, pl, patom)| {
                pl == l
                    && patom == atom
                    && functions[pfirst].exponents == functions[first].exponents
            });
            match shell_list.last_mut() {
                Some(group) if same => group.2 += size,
                _ => shell_list.push((first, l as usize, size)),
            }
            previous = Some((first, l, atom));
        }
        let (overlap, kinetic, nuclear, eri, quartets) =
            cartesian_integrals(molecule, &functions, &shell_list, threshold);
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
            // At most two tensors alive at a time.
            let mut a = quarter_transform(&eri, [nc, nc, nc, nc], &columns);
            drop(eri);
            a = quarter_transform(&a, [n, nc, nc, nc], &columns);
            a = quarter_transform(&a, [n, n, nc, nc], &columns);
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
            screening: threshold,
            quartets,
        })
    }

    /// The Schwarz threshold the repulsion integrals were screened with.
    pub fn screening_threshold(&self) -> f64 {
        self.screening
    }

    /// Quartets of shells (groups of shells sharing primitives, unique up
    /// to the eightfold symmetry) skipped by screening, and computed.
    pub fn screened_quartets(&self) -> (usize, usize) {
        self.quartets
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
