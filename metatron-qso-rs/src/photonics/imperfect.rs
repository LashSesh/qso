//! Boson sampling with imperfect photons: partial distinguishability and
//! loss.
//!
//! Photon `j` carries an internal state `|ψ_j⟩` (time, frequency,
//! polarisation) that the interferometer does not touch; only the Gram
//! matrix `S_jk = ⟨ψ_j|ψ_k⟩` matters. For `n` photons leaving through
//! `T` (Tichy 2015; Shchesnovich 2015),
//!
//! ```text
//! P(T) = Σ_{σ ∈ S_n} (Π_j S_{σ(j), j}) · perm(M ∘ M*_σ) / (Π_i t_i! · Π_modes perm(S_block))
//! ```
//!
//! with `M = U_{T,photons}`, `M*_σ` its conjugate with columns permuted by
//! `σ`, and the blocks of `S` belonging to photons in the same input mode
//! (the norm of the input state). `S = J` gives indistinguishable photons,
//! `S = I` distinguishable ones; both use the faster formulas of
//! [`BosonSampler`]. The sum over `σ` costs `n!` permanents, so partially
//! distinguishable photons are limited to [`MAX_PARTIAL_PHOTONS`].
//!
//! Loss before the interferometer (transmission `η_j` of input mode `j`)
//! removes every photon independently; finite detector efficiencies
//! `ε_i` thin the counts of output mode `i` binomially. Both are exact.

use super::boson::{BosonSampler, Distribution, MAX_PATTERNS, occupation_patterns};
use super::permanent::glynn;
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
use crate::rng::sample_cumulative;
use crate::{C64, Rng};

/// Most partially distinguishable photons (`n!` permanents per
/// probability).
pub const MAX_PARTIAL_PHOTONS: usize = 8;

/// A [`BosonSampler`] with partially distinguishable photons and losses.
#[derive(Clone, Debug)]
pub struct ImperfectBosonSampler {
    sampler: BosonSampler,
    /// Gram matrix of the internal states, photons in the order of their
    /// input modes.
    gram: CMatrix,
    /// Transmission of every input mode.
    transmission: Vec<f64>,
    /// Detection efficiency of every output mode.
    efficiency: Vec<f64>,
}

fn check_probabilities(name: &'static str, v: &[f64], len: usize) -> Result<()> {
    if v.len() != len {
        return Err(QsoError::DimensionMismatch {
            expected: len,
            actual: v.len(),
        });
    }
    if v.iter().any(|x| !(0.0..=1.0).contains(x)) {
        return Err(QsoError::invalid(name, "values must lie in [0, 1]"));
    }
    Ok(())
}

impl ImperfectBosonSampler {
    /// Perfect photons and detectors (identical to `sampler`).
    pub fn new(sampler: BosonSampler) -> Self {
        let (n, m) = (sampler.n_photons(), sampler.n_modes());
        Self {
            sampler,
            gram: CMatrix::from_element(n, n, C64::new(1.0, 0.0)),
            transmission: vec![1.0; m],
            efficiency: vec![1.0; m],
        }
    }

    /// Gram matrix `S_jk = ⟨ψ_j|ψ_k⟩` of the photons (in the order of their
    /// input modes): Hermitian, positive semidefinite, unit diagonal.
    pub fn with_gram(mut self, gram: CMatrix) -> Result<Self> {
        let n = self.sampler.n_photons();
        if gram.nrows() != n || gram.ncols() != n {
            return Err(QsoError::DimensionMismatch {
                expected: n,
                actual: gram.nrows(),
            });
        }
        if gram.iter().any(|z| !z.is_finite()) {
            return Err(QsoError::invalid("gram", "entries must be finite"));
        }
        if (0..n).any(|i| (gram[(i, i)] - C64::new(1.0, 0.0)).norm() > 1e-10) {
            return Err(QsoError::invalid(
                "gram",
                "the diagonal must be 1 (normalised states)",
            ));
        }
        if (&gram - gram.adjoint()).iter().any(|z| z.norm() > 1e-10) {
            return Err(QsoError::invalid(
                "gram",
                "the Gram matrix must be Hermitian",
            ));
        }
        let min = crate::linalg::eigh(&gram).0.first().copied().unwrap_or(0.0);
        if min < -1e-10 {
            return Err(QsoError::invalid(
                "gram",
                format!("the Gram matrix is not positive semidefinite (eigenvalue {min:.3e})"),
            ));
        }
        self.gram = gram;
        Ok(self)
    }

    /// The Gram matrix of the internal states `states[j]` (normalised to
    /// one) of the photons, in the order of their input modes.
    pub fn with_internal_states(self, states: &[Vec<C64>]) -> Result<Self> {
        let n = self.sampler.n_photons();
        if states.len() != n {
            return Err(QsoError::DimensionMismatch {
                expected: n,
                actual: states.len(),
            });
        }
        let d = states.first().map_or(0, Vec::len);
        if d == 0 || states.iter().any(|s| s.len() != d) {
            return Err(QsoError::invalid(
                "states",
                "states of one common, non-zero dimension",
            ));
        }
        let normalised: Vec<Vec<C64>> = states
            .iter()
            .map(|s| {
                let norm = s.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                if !(norm.is_finite() && norm > 0.0) {
                    return Err(QsoError::invalid(
                        "states",
                        "states must be non-zero and finite",
                    ));
                }
                Ok(s.iter().map(|z| z / norm).collect())
            })
            .collect::<Result<_>>()?;
        let gram = CMatrix::from_fn(n, n, |j, k| {
            normalised[j]
                .iter()
                .zip(&normalised[k])
                .map(|(a, b)| a.conj() * b)
                .sum()
        });
        self.with_gram(gram)
    }

    /// Every pair of photons with overlap `|⟨ψ_j|ψ_k⟩|² = x` (`S_jk = √x`):
    /// `x = 1` indistinguishable, `x = 0` distinguishable.
    pub fn with_indistinguishability(self, x: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&x) {
            return Err(QsoError::invalid("x", "the overlap must lie in [0, 1]"));
        }
        let n = self.sampler.n_photons();
        let s = x.sqrt();
        let gram = CMatrix::from_fn(n, n, |j, k| C64::new(if j == k { 1.0 } else { s }, 0.0));
        self.with_gram(gram)
    }

    /// Transmission of every input mode (loss before the interferometer).
    pub fn with_transmission(mut self, eta: &[f64]) -> Result<Self> {
        check_probabilities("transmission", eta, self.sampler.n_modes())?;
        self.transmission = eta.to_vec();
        Ok(self)
    }

    /// Detection efficiency of every output mode.
    pub fn with_efficiency(mut self, eps: &[f64]) -> Result<Self> {
        check_probabilities("efficiency", eps, self.sampler.n_modes())?;
        self.efficiency = eps.to_vec();
        Ok(self)
    }

    pub fn sampler(&self) -> &BosonSampler {
        &self.sampler
    }

    pub fn gram(&self) -> &CMatrix {
        &self.gram
    }

    /// Input modes of the photons.
    fn photon_modes(&self) -> Vec<usize> {
        self.sampler
            .input()
            .iter()
            .enumerate()
            .flat_map(|(j, &s)| std::iter::repeat_n(j, s))
            .collect()
    }

    /// Probability of detecting `output` (any number of photons up to `n`).
    pub fn probability(&self, output: &[usize]) -> Result<f64> {
        let m = self.sampler.n_modes();
        if output.len() != m {
            return Err(QsoError::DimensionMismatch {
                expected: m,
                actual: output.len(),
            });
        }
        let n = self.sampler.n_photons();
        let k: usize = output.iter().sum();
        if k > n {
            return Ok(0.0);
        }
        if self.efficiency.iter().all(|&e| e == 1.0) {
            return self.probability_before_detection(output);
        }
        // Σ over the patterns T' ≥ T that the detectors thin to T.
        let mut total = 0.0;
        let mut larger = output.to_vec();
        self.thinned(output, &mut larger, 0, n - k, &mut total)?;
        Ok(total)
    }

    /// Add `P(T') Π_i C(T'_i, T_i) ε_i^{T_i} (1 − ε_i)^{T'_i − T_i}` for every
    /// `T' ≥ T` with at most `extra` more photons, from mode `mode` on.
    fn thinned(
        &self,
        t: &[usize],
        larger: &mut Vec<usize>,
        mode: usize,
        extra: usize,
        total: &mut f64,
    ) -> Result<()> {
        if mode == t.len() {
            let p = self.probability_before_detection(larger)?;
            if p != 0.0 {
                let w: f64 = (0..t.len())
                    .map(|i| {
                        let (big, small, e) = (larger[i], t[i], self.efficiency[i]);
                        binomial(big, small)
                            * e.powi(small as i32)
                            * (1.0 - e).powi((big - small) as i32)
                    })
                    .product();
                *total += p * w;
            }
            return Ok(());
        }
        for add in 0..=extra {
            if add > 0 && self.efficiency[mode] == 1.0 {
                break;
            }
            larger[mode] = t[mode] + add;
            self.thinned(t, larger, mode + 1, extra - add, total)?;
        }
        larger[mode] = t[mode];
        Ok(())
    }

    /// Probability of `output` after the input losses, before detection.
    fn probability_before_detection(&self, output: &[usize]) -> Result<f64> {
        let photons = self.photon_modes();
        let n = photons.len();
        let k: usize = output.iter().sum();
        if k > n {
            return Ok(0.0);
        }
        let eta: Vec<f64> = photons.iter().map(|&j| self.transmission[j]).collect();
        if eta.iter().all(|&e| e == 1.0) {
            if k != n {
                return Ok(0.0);
            }
            return self.ideal(&(0..n).collect::<Vec<_>>(), output);
        }
        // Sum over the subsets of k surviving photons.
        let mut total = 0.0;
        for mask in 0u64..(1u64 << n) {
            if mask.count_ones() as usize != k {
                continue;
            }
            let weight: f64 = (0..n)
                .map(|j| {
                    if (mask >> j) & 1 == 1 {
                        eta[j]
                    } else {
                        1.0 - eta[j]
                    }
                })
                .product();
            if weight == 0.0 {
                continue;
            }
            let kept: Vec<usize> = (0..n).filter(|&j| (mask >> j) & 1 == 1).collect();
            total += weight * self.ideal(&kept, output)?;
        }
        Ok(total)
    }

    /// Lossless probability that the photons `kept` leave through `output`.
    fn ideal(&self, kept: &[usize], output: &[usize]) -> Result<f64> {
        let n = kept.len();
        if output.iter().sum::<usize>() != n {
            return Ok(0.0);
        }
        if n == 0 {
            return Ok(1.0);
        }
        let photons = self.photon_modes();
        let u = self.sampler.unitary();
        let rows: Vec<usize> = output
            .iter()
            .enumerate()
            .flat_map(|(i, &t)| std::iter::repeat_n(i, t))
            .collect();
        let cols: Vec<usize> = kept.iter().map(|&j| photons[j]).collect();
        let mt: Vec<C64> = rows
            .iter()
            .flat_map(|&r| cols.iter().map(move |&c| u[(r, c)]))
            .collect();
        let s = |a: usize, b: usize| self.gram[(kept[a], kept[b])];
        let all_one = (0..n).all(|a| (0..n).all(|b| (s(a, b) - C64::new(1.0, 0.0)).norm() < 1e-14));
        let identity = (0..n).all(|a| (0..n).all(|b| a == b || s(a, b).norm() < 1e-14));
        let out_fact: f64 = output.iter().map(|&t| factorial(t)).product();
        // Norm of the input state: Π over input modes of perm(S_block).
        let mut norm = 1.0;
        let mut a = 0;
        while a < n {
            let mut b = a;
            while b < n && cols[b] == cols[a] {
                b += 1;
            }
            let block: Vec<C64> = (a..b)
                .flat_map(|x| (a..b).map(move |y| (x, y)))
                .map(|(x, y)| s(x, y))
                .collect();
            norm *= glynn(&block, b - a).re;
            a = b;
        }
        if all_one {
            return Ok(glynn(&mt, n).norm_sqr() / (out_fact * norm));
        }
        if identity {
            let abs: Vec<f64> = mt.iter().map(|z| z.norm_sqr()).collect();
            return Ok(glynn(&abs, n) / (out_fact * norm));
        }
        if n > MAX_PARTIAL_PHOTONS {
            return Err(QsoError::invalid(
                "gram",
                format!("partially distinguishable photons are limited to {MAX_PARTIAL_PHOTONS}"),
            ));
        }
        // Σ_σ Π_j S_{σ(j), j} · perm(M ∘ M*_σ), over permutations by Heap's
        // algorithm.
        let mut sigma: Vec<usize> = (0..n).collect();
        let mut c = vec![0usize; n];
        let mut total = C64::new(0.0, 0.0);
        let mut product = vec![C64::new(0.0, 0.0); n * n];
        let mut add = |sigma: &[usize], total: &mut C64| {
            let w: C64 = (0..n).map(|j| s(sigma[j], j)).product();
            if w.norm() == 0.0 {
                return;
            }
            for r in 0..n {
                for col in 0..n {
                    product[r * n + col] = mt[r * n + col] * mt[r * n + sigma[col]].conj();
                }
            }
            *total += w * glynn(&product, n);
        };
        add(&sigma, &mut total);
        let mut i = 0;
        while i < n {
            if c[i] < i {
                if i % 2 == 0 {
                    sigma.swap(0, i);
                } else {
                    sigma.swap(c[i], i);
                }
                add(&sigma, &mut total);
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        Ok((total.re / (out_fact * norm)).max(0.0))
    }

    /// Exact distribution over all patterns of 0 to `n` detected photons.
    pub fn distribution(&self) -> Result<Distribution> {
        let (n, m) = (self.sampler.n_photons(), self.sampler.n_modes());
        let mut patterns = Vec::new();
        for k in (0..=n).rev() {
            patterns.extend(occupation_patterns(k, m)?);
            if patterns.len() > MAX_PATTERNS {
                return Err(QsoError::invalid(
                    "n_photons",
                    format!("more than {MAX_PATTERNS} output patterns"),
                ));
            }
        }
        let probabilities =
            crate::parallel::map_tasks(patterns.len(), |i| self.probability(&patterns[i]))
                .into_iter()
                .collect::<Result<Vec<f64>>>()?;
        Ok(Distribution {
            patterns,
            probabilities,
        })
    }

    /// `shots` exact samples, drawn from [`ImperfectBosonSampler::distribution`].
    pub fn sample(&self, shots: usize, rng: &mut Rng) -> Result<Vec<Vec<usize>>> {
        let d = self.distribution()?;
        let mut acc = 0.0;
        let cumulative: Vec<f64> = d
            .probabilities
            .iter()
            .map(|p| {
                acc += p;
                acc
            })
            .collect();
        Ok((0..shots)
            .map(|_| d.patterns[sample_cumulative(&cumulative, rng)].clone())
            .collect())
    }
}

fn factorial(k: usize) -> f64 {
    (1..=k).map(|i| i as f64).product()
}

fn binomial(n: usize, k: usize) -> f64 {
    factorial(n) / (factorial(k) * factorial(n - k))
}

#[cfg(test)]
mod tests;
