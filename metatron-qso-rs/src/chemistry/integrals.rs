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

use super::basis_data::{SIX_31G, STO_3G, ShellData};
use crate::error::{QsoError, Result};
use crate::parallel;

/// Bohr per ångström (CODATA 2018, as in PySCF).
pub const BOHR_PER_ANGSTROM: f64 = 1.0 / 0.529_177_210_903;

/// Most basis functions: the electron-repulsion tensor has `n⁴` entries.
pub const MAX_BASIS_FUNCTIONS: usize = 64;

const SYMBOLS: [&str; 10] = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"];

/// A molecule: nuclei, total charge and spin multiplicity `2S + 1`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Molecule {
    /// Atomic numbers.
    pub charges: Vec<u32>,
    /// Positions in bohr.
    pub positions: Vec<[f64; 3]>,
    pub charge: i32,
    pub multiplicity: u32,
}

impl Molecule {
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
                        format!("element '{}' is not one of H to Ne", fields[0]),
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

    /// A molecule from atomic numbers (1 to 10) and positions in bohr.
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
        if charges.iter().any(|&z| !(1..=10).contains(&z)) {
            return Err(QsoError::invalid(
                "atoms",
                "atomic numbers must lie in 1..=10 (H to Ne)",
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

/// A basis set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Basis {
    /// STO-3G (minimal).
    Sto3g,
    /// 6-31G (split valence).
    SixThirtyOneG,
}

impl Basis {
    /// `"sto-3g"` or `"6-31g"` (case-insensitive).
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().replace(['_', ' '], "-").as_str() {
            "sto-3g" | "sto3g" => Ok(Basis::Sto3g),
            "6-31g" | "631g" => Ok(Basis::SixThirtyOneG),
            _ => Err(QsoError::invalid(
                "basis",
                format!("unknown basis '{name}' (expected sto-3g or 6-31g)"),
            )),
        }
    }

    fn shells(self, z: u32) -> &'static [ShellData] {
        match self {
            Basis::Sto3g => STO_3G[z as usize - 1],
            Basis::SixThirtyOneG => SIX_31G[z as usize - 1],
        }
    }
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

/// Boys function `F_n(T) = ∫₀¹ t^{2n} e^{−T t²} dt` for `n = 0..=nmax`.
fn boys(nmax: usize, t: f64) -> Vec<f64> {
    let mut f = vec![0.0; nmax + 1];
    if t < 1e-14 {
        for (n, v) in f.iter_mut().enumerate() {
            *v = 1.0 / (2 * n + 1) as f64;
        }
        return f;
    }
    if t > 60.0 {
        // Asymptotic: F_n = (2n−1)!! / 2^{n+1} √(π / T^{2n+1}); the
        // neglected terms are below e^{−60}.
        let mut v = 0.5 * (std::f64::consts::PI / t).sqrt();
        for (n, fv) in f.iter_mut().enumerate() {
            *fv = v;
            v *= (2 * n + 1) as f64 / (2.0 * t);
        }
        return f;
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
    f
}

/// Hermite Coulomb integrals `R_{tuv}` for `t + u + v ≤ l`, at exponent
/// `p` and separation `pc`, indexed `[t][u][v]`.
fn hermite_r(l: usize, p: f64, pc: [f64; 3]) -> Vec<Vec<Vec<f64>>> {
    let r2 = pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2];
    let f = boys(l, p * r2);
    // r[n][t][u][v]
    let dim = l + 1;
    let mut r = vec![vec![vec![vec![0.0; dim]; dim]; dim]; dim];
    let mut factor = 1.0;
    for n in 0..=l {
        r[n][0][0][0] = factor * f[n];
        factor *= -2.0 * p;
    }
    for total in 1..=l {
        for n in 0..=l - total {
            for t in 0..=total {
                for u in 0..=total - t {
                    let v = total - t - u;
                    let value = if t > 0 {
                        let prev = if t >= 2 {
                            (t - 1) as f64 * r[n + 1][t - 2][u][v]
                        } else {
                            0.0
                        };
                        prev + pc[0] * r[n + 1][t - 1][u][v]
                    } else if u > 0 {
                        let prev = if u >= 2 {
                            (u - 1) as f64 * r[n + 1][t][u - 2][v]
                        } else {
                            0.0
                        };
                        prev + pc[1] * r[n + 1][t][u - 1][v]
                    } else {
                        let prev = if v >= 2 {
                            (v - 1) as f64 * r[n + 1][t][u][v - 2]
                        } else {
                            0.0
                        };
                        prev + pc[2] * r[n + 1][t][u][v - 1]
                    };
                    r[n][t][u][v] = value;
                }
            }
        }
    }
    r.swap_remove(0)
}

fn nuclear_primitive(
    a: f64,
    pa: [u32; 3],
    ca: [f64; 3],
    b: f64,
    pb: [u32; 3],
    cb: [f64; 3],
    c: [f64; 3],
) -> f64 {
    let p = a + b;
    let pp: Vec<f64> = (0..3).map(|k| (a * ca[k] + b * cb[k]) / p).collect();
    let l = (pa.iter().sum::<u32>() + pb.iter().sum::<u32>()) as usize;
    let r = hermite_r(l, p, [pp[0] - c[0], pp[1] - c[1], pp[2] - c[2]]);
    let ex: Vec<Vec<f64>> = (0..3)
        .map(|k| {
            (0..=(pa[k] + pb[k]) as i32)
                .map(|t| hermite_e(pa[k] as i32, pb[k] as i32, t, ca[k] - cb[k], a, b))
                .collect()
        })
        .collect();
    let mut v = 0.0;
    for (t, et) in ex[0].iter().enumerate() {
        for (u, eu) in ex[1].iter().enumerate() {
            for (w, ev) in ex[2].iter().enumerate() {
                v += et * eu * ev * r[t][u][w];
            }
        }
    }
    2.0 * std::f64::consts::PI / p * v
}

/// `(ab|cd)` over primitives.
#[allow(clippy::too_many_arguments)]
fn eri_primitive(
    a: f64,
    pa: [u32; 3],
    ca: [f64; 3],
    b: f64,
    pb: [u32; 3],
    cb: [f64; 3],
    c: f64,
    pc: [u32; 3],
    cc: [f64; 3],
    d: f64,
    pd: [u32; 3],
    cd: [f64; 3],
) -> f64 {
    let p = a + b;
    let q = c + d;
    let alpha = p * q / (p + q);
    let pp: Vec<f64> = (0..3).map(|k| (a * ca[k] + b * cb[k]) / p).collect();
    let qq: Vec<f64> = (0..3).map(|k| (c * cc[k] + d * cd[k]) / q).collect();
    let l = (pa.iter().sum::<u32>()
        + pb.iter().sum::<u32>()
        + pc.iter().sum::<u32>()
        + pd.iter().sum::<u32>()) as usize;
    let r = hermite_r(l, alpha, [pp[0] - qq[0], pp[1] - qq[1], pp[2] - qq[2]]);
    let e1: Vec<Vec<f64>> = (0..3)
        .map(|k| {
            (0..=(pa[k] + pb[k]) as i32)
                .map(|t| hermite_e(pa[k] as i32, pb[k] as i32, t, ca[k] - cb[k], a, b))
                .collect()
        })
        .collect();
    let e2: Vec<Vec<f64>> = (0..3)
        .map(|k| {
            (0..=(pc[k] + pd[k]) as i32)
                .map(|t| hermite_e(pc[k] as i32, pd[k] as i32, t, cc[k] - cd[k], c, d))
                .collect()
        })
        .collect();
    let mut v = 0.0;
    for (t, et) in e1[0].iter().enumerate() {
        for (u, eu) in e1[1].iter().enumerate() {
            for (w, ev) in e1[2].iter().enumerate() {
                let left = et * eu * ev;
                if left == 0.0 {
                    continue;
                }
                for (tau, ft) in e2[0].iter().enumerate() {
                    for (nu, fu) in e2[1].iter().enumerate() {
                        for (phi, fv) in e2[2].iter().enumerate() {
                            let sign = if (tau + nu + phi) % 2 == 1 { -1.0 } else { 1.0 };
                            v += left * sign * ft * fu * fv * r[t + tau][u + nu][w + phi];
                        }
                    }
                }
            }
        }
    }
    2.0 * std::f64::consts::PI.powf(2.5) / (p * q * (p + q).sqrt()) * v
}

/// `(ij|kl)` over contracted functions.
fn contracted_eri(functions: &[Function], (i, j): (usize, usize), (k, l): (usize, usize)) -> f64 {
    let (fi, fj, fk, fl) = (&functions[i], &functions[j], &functions[k], &functions[l]);
    let mut v = 0.0;
    for a in 0..fi.exponents.len() {
        for b in 0..fj.exponents.len() {
            let cab = fi.coefficients[a] * fj.coefficients[b];
            for c in 0..fk.exponents.len() {
                for d in 0..fl.exponents.len() {
                    v += cab
                        * fk.coefficients[c]
                        * fl.coefficients[d]
                        * eri_primitive(
                            fi.exponents[a],
                            fi.powers,
                            fi.center,
                            fj.exponents[b],
                            fj.powers,
                            fj.center,
                            fk.exponents[c],
                            fk.powers,
                            fk.center,
                            fl.exponents[d],
                            fl.powers,
                            fl.center,
                        );
                }
            }
        }
    }
    v
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
}

impl AoIntegrals {
    /// The integrals of `molecule` in `basis`. Functions are ordered by
    /// atom, then by shell as the basis lists them, `p` functions as
    /// `x, y, z` (the order of PySCF).
    pub fn compute(molecule: &Molecule, basis: Basis) -> Result<Self> {
        let mut functions = Vec::new();
        let mut atoms = Vec::new();
        let mut angular = Vec::new();
        for (atom, (&z, &pos)) in molecule.charges.iter().zip(&molecule.positions).enumerate() {
            for &(l, prims) in basis.shells(z) {
                let powers: &[[u32; 3]] = match l {
                    0 => &[[0, 0, 0]],
                    1 => &[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    _ => unreachable!("the bundled bases have s and p shells only"),
                };
                for &p in powers {
                    functions.push(Function::new(pos, p, prims));
                    atoms.push(atom);
                    angular.push(l as u32);
                }
            }
        }
        let n = functions.len();
        if n > MAX_BASIS_FUNCTIONS {
            return Err(QsoError::invalid(
                "molecule",
                format!("{n} basis functions exceed the maximum {MAX_BASIS_FUNCTIONS}"),
            ));
        }
        let one = |op: &dyn Fn(&Function, usize, &Function, usize) -> f64| -> Vec<f64> {
            let mut m = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..=i {
                    let (fi, fj) = (&functions[i], &functions[j]);
                    let mut v = 0.0;
                    for a in 0..fi.exponents.len() {
                        for b in 0..fj.exponents.len() {
                            v += fi.coefficients[a] * fj.coefficients[b] * op(fi, a, fj, b);
                        }
                    }
                    m[i * n + j] = v;
                    m[j * n + i] = v;
                }
            }
            m
        };
        let overlap = one(&|fi, a, fj, b| {
            overlap_primitive(
                fi.exponents[a],
                fi.powers,
                fi.center,
                fj.exponents[b],
                fj.powers,
                fj.center,
            )
        });
        let kinetic = one(&|fi, a, fj, b| {
            kinetic_primitive(
                fi.exponents[a],
                fi.powers,
                fi.center,
                fj.exponents[b],
                fj.powers,
                fj.center,
            )
        });
        let nuclear = one(&|fi, a, fj, b| {
            molecule
                .charges
                .iter()
                .zip(&molecule.positions)
                .map(|(&z, &c)| {
                    -(z as f64)
                        * nuclear_primitive(
                            fi.exponents[a],
                            fi.powers,
                            fi.center,
                            fj.exponents[b],
                            fj.powers,
                            fj.center,
                            c,
                        )
                })
                .sum()
        });
        // Unique quartets i ≥ j, k ≥ l, ij ≥ kl, in parallel.
        let pairs: Vec<(usize, usize)> =
            (0..n).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();
        let quartets: Vec<(usize, usize)> = (0..pairs.len())
            .flat_map(|x| (0..=x).map(move |y| (x, y)))
            .collect();
        let chunk = 64;
        let values: Vec<f64> = parallel::map_tasks(quartets.len().div_ceil(chunk), |c| {
            (c * chunk..((c + 1) * chunk).min(quartets.len()))
                .map(|k| contracted_eri(&functions, pairs[quartets[k].0], pairs[quartets[k].1]))
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
        Ok(Self {
            n,
            overlap,
            kinetic,
            nuclear,
            eri,
            atoms,
            angular,
        })
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

    pub(crate) fn eri_tensor(&self) -> &[f64] {
        &self.eri
    }
}

#[cfg(test)]
mod tests;
