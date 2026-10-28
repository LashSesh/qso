//! Full configuration interaction without the Hamiltonian matrix.
//!
//! Determinants are pairs of α and β strings (`index = i_β·N_α + i_α`, the
//! order of [`MolecularIntegrals::determinants`]). With the spin-summed
//! excitations `E_kl = a†_kα a_lα + a†_kβ a_lβ`,
//!
//! ```text
//! H = E_core + Σ_kl h'_kl E_kl + ½ Σ_klmn (kl|mn) E_kl E_mn,   h'_kl = h_kl − ½ Σ_j (kj|jl)
//! ```
//!
//! so `σ = H c` follows from `D_mn = E_mn c`, `G_kl = Σ_mn (kl|mn) D_mn`
//! and `σ = E_core c + Σ_kl h'_kl D_kl + ½ Σ_kl E_kl G_kl` (Knowles and
//! Handy, Chem. Phys. Lett. 111, 315 (1984)). The single excitations of
//! every string are tabulated once; `D` and `G` hold `m²` numbers per
//! determinant. The lowest roots come from Davidson's method with the
//! diagonal as preconditioner.

use super::MolecularIntegrals;
use crate::error::{QsoError, Result};
use crate::linalg::{RMatrix, eigh_real};
use crate::parallel;

/// Most numbers `D` and `G` may hold together (`2·N·m²`; 2 GiB).
pub(super) const MAX_DIRECT_CI_ENTRIES: usize = 1 << 28;

/// `C(n, k)` for small arguments.
fn choose(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut c: u128 = 1;
    for i in 0..k.min(n - k) {
        c = c * (n - i) as u128 / (i + 1) as u128;
    }
    c as usize
}

/// Position of a string among all strings with as many electrons, in
/// increasing numeric order (the combinatorial number system).
fn rank(mut s: u64) -> usize {
    let mut r = 0;
    let mut i = 1;
    while s != 0 {
        let p = s.trailing_zeros() as usize;
        r += choose(p, i);
        i += 1;
        s &= s - 1;
    }
    r
}

/// The single excitations `E_kl` of every string: `(k·m + l, target,
/// sign)`, the diagonal `E_ll` included.
fn excitations(strings: &[u64], m: usize) -> Vec<Vec<(u32, u32, f64)>> {
    strings
        .iter()
        .map(|&s| {
            let mut out = Vec::new();
            for l in 0..m {
                if s >> l & 1 == 0 {
                    continue;
                }
                for k in 0..m {
                    if k != l && s >> k & 1 == 1 {
                        continue;
                    }
                    let removed = s & !(1u64 << l);
                    let below_l = (s & ((1u64 << l) - 1)).count_ones();
                    let below_k = (removed & ((1u64 << k) - 1)).count_ones();
                    let sign = if (below_l + below_k) % 2 == 1 {
                        -1.0
                    } else {
                        1.0
                    };
                    let target = removed | (1u64 << k);
                    out.push(((k * m + l) as u32, rank(target) as u32, sign));
                }
            }
            out
        })
        .collect()
}

/// Strings with `k` of `m` bits, in increasing order.
fn strings(m: usize, k: usize) -> Vec<u64> {
    super::combinations(m, k)
}

pub(super) struct DirectCi<'a> {
    mol: &'a MolecularIntegrals,
    m: usize,
    n_alpha: usize,
    alpha: Vec<Vec<(u32, u32, f64)>>,
    beta: Vec<Vec<(u32, u32, f64)>>,
    /// `h'_kl`.
    one: Vec<f64>,
    /// `(kl|mn)` as an `m² × m²` matrix.
    eri: RMatrix,
}

impl<'a> DirectCi<'a> {
    pub(super) fn new(mol: &'a MolecularIntegrals) -> Result<Self> {
        let m = mol.n_orbitals;
        let sa = strings(m, mol.n_alpha);
        let sb = strings(m, mol.n_beta);
        let dim = sa.len() * sb.len();
        if 2 * dim * m * m > MAX_DIRECT_CI_ENTRIES {
            return Err(QsoError::invalid(
                "molecule",
                format!(
                    "{dim} determinants of {m} orbitals need more than {MAX_DIRECT_CI_ENTRIES} intermediate numbers"
                ),
            ));
        }
        let m2 = m * m;
        let g =
            |p: usize, q: usize, r: usize, s: usize| mol.two_body[((p * m + q) * m + r) * m + s];
        let mut one = vec![0.0; m2];
        for k in 0..m {
            for l in 0..m {
                let mut v = mol.one_body[k * m + l];
                for j in 0..m {
                    v -= 0.5 * g(k, j, j, l);
                }
                one[k * m + l] = v;
            }
        }
        let eri = RMatrix::from_fn(m2, m2, |kl, mn| g(kl / m, kl % m, mn / m, mn % m));
        Ok(Self {
            mol,
            m,
            n_alpha: sa.len(),
            alpha: excitations(&sa, m),
            beta: excitations(&sb, m),
            one,
            eri,
        })
    }

    pub(super) fn dim(&self) -> usize {
        self.alpha.len() * self.beta.len()
    }

    /// `out_J = Σ_kl w_kl(I) …`: for determinant `j`, calls `f(kl, i, sign)`
    /// for every `E_kl` with `⟨j|E_kl|i⟩ = sign` — that is, every excitation
    /// `E_lk` of `j` reaching `i`.
    fn for_each_excitation(&self, j: usize, mut f: impl FnMut(usize, usize, f64)) {
        let (ja, jb) = (j % self.n_alpha, j / self.n_alpha);
        let m = self.m;
        for &(pq, ia, sign) in &self.alpha[ja] {
            let (p, q) = (pq as usize / m, pq as usize % m);
            f(q * m + p, jb * self.n_alpha + ia as usize, sign);
        }
        for &(pq, ib, sign) in &self.beta[jb] {
            let (p, q) = (pq as usize / m, pq as usize % m);
            f(q * m + p, ib as usize * self.n_alpha + ja, sign);
        }
    }

    /// `σ = H c`.
    pub(super) fn sigma(&self, c: &[f64]) -> Vec<f64> {
        let (dim, m2) = (self.dim(), self.m * self.m);
        let chunk = 256;
        let chunks = dim.div_ceil(chunk);
        // D_J = (E_kl c)_J for all kl, then G_J = D_J · (kl|mn), by blocks.
        let blocks: Vec<(Vec<f64>, Vec<f64>)> = parallel::map_tasks(chunks, |b| {
            let (start, end) = (b * chunk, ((b + 1) * chunk).min(dim));
            let mut d = RMatrix::zeros(end - start, m2);
            for j in start..end {
                self.for_each_excitation(j, |kl, i, sign| d[(j - start, kl)] += sign * c[i]);
            }
            let one: Vec<f64> = (0..end - start)
                .map(|r| (0..m2).map(|kl| self.one[kl] * d[(r, kl)]).sum())
                .collect();
            let g = d * &self.eri;
            // Row-major copy of G for the gathers below.
            let rows: Vec<f64> = (0..end - start)
                .flat_map(|r| (0..m2).map(move |kl| (r, kl)))
                .map(|(r, kl)| g[(r, kl)])
                .collect();
            (one, rows)
        });
        let mut sigma = vec![0.0; dim];
        let mut g = Vec::with_capacity(dim * m2);
        for (b, (one, rows)) in blocks.into_iter().enumerate() {
            for (r, v) in one.into_iter().enumerate() {
                sigma[b * chunk + r] = v;
            }
            g.extend(rows);
        }
        // σ_J += ½ Σ_kl (E_kl G_kl)_J + E_core c_J.
        let core = self.mol.core_energy;
        let parts: Vec<Vec<f64>> = parallel::map_tasks(chunks, |b| {
            let (start, end) = (b * chunk, ((b + 1) * chunk).min(dim));
            (start..end)
                .map(|j| {
                    let mut v = 0.0;
                    self.for_each_excitation(j, |kl, i, sign| v += sign * g[i * m2 + kl]);
                    0.5 * v + core * c[j]
                })
                .collect()
        });
        for (b, part) in parts.into_iter().enumerate() {
            for (r, v) in part.into_iter().enumerate() {
                sigma[b * chunk + r] += v;
            }
        }
        sigma
    }

    /// Diagonal of `H` (Slater–Condon).
    pub(super) fn diagonal(&self, determinants: &[u64]) -> Vec<f64> {
        parallel::map_tasks(determinants.len().div_ceil(1024), |b| {
            let end = ((b + 1) * 1024).min(determinants.len());
            determinants[b * 1024..end]
                .iter()
                .map(|&d| self.mol.diagonal_element(d))
                .collect::<Vec<f64>>()
        })
        .into_iter()
        .flatten()
        .collect()
    }
}

/// The `k` lowest eigenvalues of a real symmetric operator by Davidson's
/// method (block, with the diagonal as preconditioner and restarts),
/// converged when every residual is below `tol`.
pub(super) fn davidson(
    apply: impl Fn(&[f64]) -> Vec<f64>,
    diagonal: &[f64],
    k: usize,
    tol: f64,
    max_iterations: usize,
) -> Result<Vec<f64>> {
    let dim = diagonal.len();
    let max_space = (8 * k).max(24).min(dim);
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    // Start from the unit vectors of the lowest diagonal elements.
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| diagonal[a].total_cmp(&diagonal[b]));
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut images: Vec<Vec<f64>> = Vec::new();
    let add = |mut v: Vec<f64>, basis: &mut Vec<Vec<f64>>, images: &mut Vec<Vec<f64>>| {
        for _ in 0..2 {
            for b in basis.iter() {
                let c = dot(&v, b);
                v.iter_mut().zip(b).for_each(|(x, y)| *x -= c * y);
            }
        }
        let norm = dot(&v, &v).sqrt();
        if norm < 1e-10 {
            return false;
        }
        v.iter_mut().for_each(|x| *x /= norm);
        images.push(apply(&v));
        basis.push(v);
        true
    };
    // Unit vectors meet only the symmetries of their determinants; random
    // vectors (seeded) reach every eigenvector, also of degenerate levels.
    let mut rng = crate::Rng::seed_from_u64(0x5eed);
    for &i in order.iter().take(k.min(dim)) {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        add(v, &mut basis, &mut images);
    }
    for _ in 0..k.min(dim.saturating_sub(basis.len())) {
        let v: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
        add(v, &mut basis, &mut images);
    }
    let mut worst = f64::INFINITY;
    for _ in 0..max_iterations {
        let s = basis.len();
        let h = RMatrix::from_fn(s, s, |a, b| dot(&basis[a], &images[b]));
        let h = (&h + h.transpose()) * 0.5;
        let (theta, y) = eigh_real(&h);
        let ritz = |r: usize, vectors: &[Vec<f64>]| -> Vec<f64> {
            let mut x = vec![0.0; dim];
            for a in 0..s {
                let w = y[(a, r)];
                x.iter_mut()
                    .zip(&vectors[a])
                    .for_each(|(xi, v)| *xi += w * v);
            }
            x
        };
        let mut corrections = Vec::new();
        worst = 0.0f64;
        // The extra vectors keep refining every Ritz vector up to 2k.
        for (r, &value) in theta.iter().enumerate().take((2 * k).min(s)) {
            let x = ritz(r, &basis);
            let ax = ritz(r, &images);
            let residual: Vec<f64> = ax.iter().zip(&x).map(|(a, b)| a - value * b).collect();
            let norm = dot(&residual, &residual).sqrt();
            worst = worst.max(norm);
            if norm > tol {
                let t: Vec<f64> = residual
                    .iter()
                    .zip(diagonal)
                    .map(|(&ri, &di)| {
                        let denom = value - di;
                        ri / if denom.abs() < 1e-8 {
                            1e-8_f64.copysign(denom)
                        } else {
                            denom
                        }
                    })
                    .collect();
                corrections.push(t);
            }
        }
        if corrections.is_empty() {
            return Ok(theta[..k].to_vec());
        }
        if s + corrections.len() > max_space {
            // Restart from the current Ritz vectors.
            let keep = (2 * k).min(s);
            let new_basis: Vec<Vec<f64>> = (0..keep).map(|r| ritz(r, &basis)).collect();
            let new_images: Vec<Vec<f64>> = (0..keep).map(|r| ritz(r, &images)).collect();
            basis = new_basis;
            images = new_images;
        }
        let before = basis.len();
        for t in corrections {
            add(t, &mut basis, &mut images);
        }
        if basis.len() == before {
            // Nothing new: the residuals are at rounding level.
            let s = basis.len();
            let h = RMatrix::from_fn(s, s, |a, b| dot(&basis[a], &images[b]));
            let (theta, _) = eigh_real(&((&h + h.transpose()) * 0.5));
            return Ok(theta[..k].to_vec());
        }
    }
    Err(QsoError::NoConvergence {
        method: "Davidson",
        iterations: max_iterations,
        residual: worst,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chemistry::integrals::{AoIntegrals, Basis, Molecule};
    use crate::chemistry::scf::{ScfOptions, rhf};
    use std::collections::HashMap;

    /// Water in 6-31G with a frozen core: random-ish real integrals of a
    /// real molecule, in sectors with given electron numbers.
    fn water(active: usize, alpha: usize, beta: usize) -> MolecularIntegrals {
        let mol = Molecule::from_atoms(
            "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
            0,
            1,
        )
        .unwrap();
        let ints = AoIntegrals::compute(&mol, Basis::SixThirtyOneG).unwrap();
        let hf = rhf(&mol, &ints, &ScfOptions::default()).unwrap();
        let mo = hf.molecular_integrals(&ints, 1, Some(active)).unwrap();
        MolecularIntegrals {
            n_alpha: alpha,
            n_beta: beta,
            ..mo
        }
    }

    #[test]
    fn sigma_is_the_slater_condon_matrix_times_c() {
        for (active, a, b) in [(5, 2, 2), (6, 3, 1), (6, 4, 3), (7, 2, 0)] {
            let mol = water(active, a, b);
            let dets = mol.determinants().unwrap();
            let index: HashMap<u64, usize> =
                dets.iter().enumerate().map(|(i, &d)| (d, i)).collect();
            let ci = DirectCi::new(&mol).unwrap();
            assert_eq!(ci.dim(), dets.len());
            let c: Vec<f64> = (0..dets.len())
                .map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0)
                .collect();
            let sigma = ci.sigma(&c);
            let diagonal = ci.diagonal(&dets);
            for (j, &d) in dets.iter().enumerate() {
                let col = mol.slater_condon(d, &index);
                // H is symmetric: row j equals column j.
                let want: f64 = col.iter().map(|&(i, v)| v * c[i]).sum();
                assert!(
                    (sigma[j] - want).abs() < 1e-11,
                    "{active} {a} {b} {j}: {} vs {want}",
                    sigma[j]
                );
                let diag = col.iter().find(|e| e.0 == j).unwrap().1;
                assert!((diagonal[j] - diag).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn davidson_finds_the_lowest_roots_also_when_degenerate() {
        for (active, a, b, k) in [(6, 3, 3, 6), (7, 3, 2, 5), (7, 4, 3, 4), (8, 2, 2, 8)] {
            let mol = water(active, a, b);
            let dense = mol.fci_energies(k).unwrap();
            let dets = mol.determinants().unwrap();
            let ci = DirectCi::new(&mol).unwrap();
            let diagonal = ci.diagonal(&dets);
            let got = davidson(|c| ci.sigma(c), &diagonal, k, 1e-7, 1000).unwrap();
            for (x, y) in got.iter().zip(&dense) {
                assert!(
                    (x - y).abs() < 1e-10,
                    "{active} {a} {b}: {got:?} vs {dense:?}"
                );
            }
        }
    }

    #[test]
    fn ranks_follow_the_order_of_combinations() {
        for (m, k) in [(6, 3), (10, 4), (12, 1), (5, 0)] {
            for (i, &s) in strings(m, k).iter().enumerate() {
                assert_eq!(rank(s), i);
            }
        }
    }
}
