//! The hypercube with dephasing on orbitals that are never enumerated.

use num_complex::Complex64 as C64;

use super::OrbitalMap;
use super::transport::{OrbitalProblem, TraceReads, trace};
use crate::error::{QsoError, Result};
use crate::walks::transport::{
    DephasingOptimum, TransportOptions, TransportResult, TransportState, check_energies,
    check_rates, check_traps,
};

/// Most orbitals of a [`HypercubeOpenWalk`] (`C(d + 3, 3)`, so `d ≤ 230`).
pub const MAX_HYPERCUBE_ORBITALS: usize = 1 << 21;

/// The continuous-time walk on the hypercube `Q_d` with site dephasing and
/// dephasing of every coordinate, from states invariant under the
/// permutations of the coordinates (node `0…0`, the uniform superposition).
///
/// Under those permutations the pair of nodes `(u, v)` is characterised by
/// the numbers `(n₀₀, n₀₁, n₁₀, n₁₁)` of coordinates with bits
/// `(u_k, v_k)`, so the state has `C(d + 3, 3)` coefficients however large
/// `2^d` is, and the graph is never built: `Q₁₀₀` with dephasing is a
/// problem in 176 851 dimensions. `H = γL` and `H = −γA` differ by `γd·I`
/// on the hypercube, so they give the same density matrix. Entry `ρ_uv`
/// decays at `κ_site·[u ≠ v] + κ_coordinate·dist(u, v)`. Site energies,
/// traps, losses and dephasing by Hamming weight
/// ([`Self::with_site_energies`], [`Self::with_traps`], ...) keep the
/// symmetry; [`Self::transport`] gives the efficiency and mean arrival time
/// at the traps.
///
/// ```
/// use metatron_qso::walks::symmetric::HypercubeOpenWalk;
///
/// // Q₆₀ with site dephasing: 39 711 orbitals for 2⁶⁰ nodes.
/// let walk = HypercubeOpenWalk::new(60, 1.0, 0.5, 0.0).unwrap();
/// assert_eq!(walk.num_orbitals(), 39_711);
/// let p = walk.distance_distribution(&walk.evolve(&walk.localized(), 0.5).unwrap()).unwrap();
/// assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-10);
/// ```
#[derive(Clone, Debug)]
pub struct HypercubeOpenWalk {
    d: usize,
    generator: OrbitalMap,
    /// The hopping rate `γ`.
    gamma: f64,
    site_rate: f64,
    coordinate_rate: f64,
    /// Traps, losses, site energies and per-node dephasing by Hamming
    /// weight (none until a builder sets them).
    sites: Option<Box<HypercubeSites>>,
}

/// The site terms of a [`HypercubeOpenWalk`], by Hamming weight `0..=d`.
#[derive(Clone, Debug)]
struct HypercubeSites {
    energies: Vec<f64>,
    kappa: Vec<f64>,
    losses: Vec<f64>,
    dephasing: Vec<f64>,
    /// Traps `(weight, κ)` in the order given.
    traps: Vec<(usize, f64)>,
}

/// `C(d, k)` as a float, from [`log_size`].
fn binomial(d: usize, k: usize) -> f64 {
    log_size(d, [d - k, 0, 0, k]).exp()
}

/// Number of orbitals of `Q_d`: `(n₀₁, n₁₀, n₁₁)` with sum at most `d`.
fn count(d: usize) -> Option<usize> {
    let d = d.checked_add(3)?;
    d.checked_mul(d - 1)?.checked_mul(d - 2).map(|x| x / 6)
}

/// Orbitals in lexicographic order of `(n₀₁, n₁₀, n₁₁)`.
fn index(d: usize, n01: usize, n10: usize, n11: usize) -> usize {
    // Orbitals with a smaller n₀₁: Σ_{i < n₀₁} triangle(d − i).
    let before = count(d).unwrap() - count(d - n01).unwrap_or(0);
    let m = d - n01;
    // Within n₀₁: Σ_{j < n₁₀} (m − j + 1) with a smaller n₁₀.
    before + n10 * (m + 1) - n10 * n10.saturating_sub(1) / 2 + n11
}

/// `ln |O|`: the multinomial `d! / (n₀₀! n₀₁! n₁₀! n₁₁!)`.
fn log_size(d: usize, n: [usize; 4]) -> f64 {
    let ln_factorial = |m: usize| (2..=m).map(|j| (j as f64).ln()).sum::<f64>();
    ln_factorial(d) - n.iter().map(|&m| ln_factorial(m)).sum::<f64>()
}

impl HypercubeOpenWalk {
    /// The walk on `Q_d` with hopping rate `gamma`, site dephasing at
    /// `site_rate` and dephasing of every coordinate at `coordinate_rate`.
    pub fn new(d: usize, gamma: f64, site_rate: f64, coordinate_rate: f64) -> Result<Self> {
        if d == 0 {
            return Err(QsoError::invalid("d", "need at least one coordinate"));
        }
        let total = count(d)
            .filter(|&c| c <= MAX_HYPERCUBE_ORBITALS)
            .ok_or_else(|| {
                QsoError::invalid(
                    "d",
                    format!("C(d + 3, 3) orbitals exceed {MAX_HYPERCUBE_ORBITALS}: d ≤ 230"),
                )
            })?;
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(QsoError::invalid(
                "gamma",
                "the hopping rate must be positive",
            ));
        }
        for (name, r) in [
            ("site_rate", site_rate),
            ("coordinate_rate", coordinate_rate),
        ] {
            if !(r.is_finite() && r >= 0.0) {
                return Err(QsoError::invalid(name, "rates must be finite and ≥ 0"));
            }
        }
        let i = C64::new(0.0, 1.0);
        let mut rows = Vec::with_capacity(total);
        // In the orthonormal basis the entry `γ n_c` of moving a coordinate
        // from class c to c' becomes γ n_c √(|O|/|P|) = γ √(n_c (n_c' + 1)),
        // with n_c' the count of O.
        for n01 in 0..=d {
            for n10 in 0..=d - n01 {
                for n11 in 0..=d - n01 - n10 {
                    let n00 = d - n01 - n10 - n11;
                    let mut row = Vec::with_capacity(9);
                    let differ = n01 + n10;
                    let decay =
                        coordinate_rate * differ as f64 + if differ > 0 { site_rate } else { 0.0 };
                    row.push((index(d, n01, n10, n11) as u32, C64::new(-decay, 0.0)));
                    // H = c·I − γ Σ_k X_k: (Hρ)_ab flips bit k of a, (ρH)_ab
                    // bit k of b, moving coordinate k to another class.
                    let n = [n00, n01, n10, n11];
                    // Classes 00, 01, 10, 11 as indices 0..4; flipping u's
                    // bit maps them to 10, 11, 00, 01; flipping v's to 01,
                    // 00, 11, 10.
                    for (flip, sign) in [([2, 3, 0, 1], -i), ([1, 0, 3, 2], i)] {
                        for c in 0..4 {
                            if n[c] == 0 {
                                continue;
                            }
                            let mut m = n;
                            m[c] -= 1;
                            m[flip[c]] += 1;
                            let col = index(d, m[1], m[2], m[3]) as u32;
                            // −i·(−γ)·n_c for Hρ, +i·(−γ)·n_c for ρH.
                            let w = (n[c] as f64 * (n[flip[c]] + 1) as f64).sqrt();
                            row.push((col, -sign * (gamma * w)));
                        }
                    }
                    rows.push(row);
                }
            }
        }
        Ok(HypercubeOpenWalk {
            d,
            generator: OrbitalMap::from_rows(rows),
            gamma,
            site_rate,
            coordinate_rate,
            sites: None,
        })
    }

    /// Site energies by Hamming weight (`d + 1` finite values, not scaled
    /// by the hopping rate): every node of weight `k` at `by_weight[k]`.
    /// Replaces earlier energies.
    pub fn with_site_energies(mut self, by_weight: &[f64]) -> Result<Self> {
        check_energies(by_weight, self.d + 1)?;
        self.sites_mut().energies = by_weight.to_vec();
        self.update_diagonal();
        Ok(self)
    }

    /// Traps `(k, κ)` by Hamming weight: every node of weight `k ≤ d` is a
    /// trap of rate `κ > 0` (`k = d` the antipode of `0…0`), weights
    /// distinct. Replaces earlier traps (`&[]` removes them).
    pub fn with_traps(mut self, traps: &[(usize, f64)]) -> Result<Self> {
        let d = self.d;
        if let Some(&(k, _)) = traps.iter().find(|t| t.0 > d) {
            return Err(QsoError::invalid(
                "traps",
                format!("weight {k} exceeds d = {d}"),
            ));
        }
        check_traps(traps, d + 1)?;
        let sites = self.sites_mut();
        sites.kappa = vec![0.0; d + 1];
        for &(k, kappa) in traps {
            sites.kappa[k] = kappa;
        }
        sites.traps = traps.to_vec();
        self.update_diagonal();
        Ok(self)
    }

    /// Loss rates by Hamming weight (`d + 1` values `≥ 0`). Replaces
    /// earlier losses.
    pub fn with_losses(mut self, by_weight: &[f64]) -> Result<Self> {
        check_rates("losses", by_weight, self.d + 1)?;
        self.sites_mut().losses = by_weight.to_vec();
        self.update_diagonal();
        Ok(self)
    }

    /// Dephasing rates by Hamming weight (`d + 1` values `≥ 0`), added to
    /// the `site_rate` of [`Self::new`]: entry `ρ_ab` (`a ≠ b`) decays at
    /// `(γ(|a|) + γ(|b|))/2` more. Replaces earlier rates.
    pub fn with_node_dephasing(mut self, by_weight: &[f64]) -> Result<Self> {
        check_rates("dephasing", by_weight, self.d + 1)?;
        self.sites_mut().dephasing = by_weight.to_vec();
        self.update_diagonal();
        Ok(self)
    }

    /// The traps `(weight, κ)` in the order given.
    pub fn traps(&self) -> &[(usize, f64)] {
        self.sites.as_ref().map_or(&[], |s| &s.traps)
    }

    fn sites_mut(&mut self) -> &mut HypercubeSites {
        let d = self.d;
        self.sites.get_or_insert_with(|| {
            Box::new(HypercubeSites {
                energies: vec![0.0; d + 1],
                kappa: vec![0.0; d + 1],
                losses: vec![0.0; d + 1],
                dephasing: vec![0.0; d + 1],
                traps: Vec::new(),
            })
        })
    }

    /// The diagonal of every row `(n₀₁, n₁₀, n₁₁)` from the rates of
    /// [`Self::new`] and the site terms, with `|a| = n₁₀ + n₁₁`,
    /// `|b| = n₀₁ + n₁₁` and `k = κ + Γ/2`:
    /// `−(base + k(|a|) + k(|b|) + [a ≠ b](γ(|a|) + γ(|b|))/2) − i(ε(|a|) − ε(|b|))`;
    /// rows without terms keep the entry of `new` bit for bit.
    fn update_diagonal(&mut self) {
        let d = self.d;
        let Some(sites) = &self.sites else {
            return;
        };
        let k = |w: usize| sites.kappa[w] + 0.5 * sites.losses[w];
        let mut row = 0;
        for n01 in 0..=d {
            for n10 in 0..=d - n01 {
                for n11 in 0..=d - n01 - n10 {
                    let differ = n01 + n10;
                    let (wa, wb) = (n10 + n11, n01 + n11);
                    let base = self.coordinate_rate * differ as f64
                        + if differ > 0 { self.site_rate } else { 0.0 };
                    let mut term = k(wa) + k(wb);
                    if differ > 0 {
                        term += 0.5 * (sites.dephasing[wa] + sites.dephasing[wb]);
                    }
                    let rotation = sites.energies[wa] - sites.energies[wb];
                    let at = self.generator.diagonal(row);
                    self.generator.vals[at] = if term == 0.0 && rotation == 0.0 {
                        C64::new(-base, 0.0)
                    } else {
                        C64::new(-(base + term), -rotation)
                    };
                    row += 1;
                }
            }
        }
    }

    /// The transport problem from `0…0`.
    fn problem(&self, options: &TransportOptions) -> Result<OrbitalProblem<'_>> {
        options.validate()?;
        let d = self.d;
        let sites = match &self.sites {
            Some(s) if !s.traps.is_empty() => s,
            _ => {
                return Err(QsoError::invalid(
                    "traps",
                    "transport needs at least one trap (with_traps)",
                ));
            }
        };
        let n = self.num_orbitals();
        let mut sqrt_size = vec![0.0; n];
        let mut coherence = vec![false; n];
        for n01 in 0..=d {
            for n10 in 0..=d - n01 {
                for n11 in 0..=d - n01 - n10 {
                    let o = index(d, n01, n10, n11);
                    let n00 = d - n01 - n10 - n11;
                    sqrt_size[o] = (0.5 * log_size(d, [n00, n01, n10, n11])).exp();
                    coherence[o] = n01 + n10 > 0;
                }
            }
        }
        // The diagonal orbital of weight k is index(d, 0, 0, k) = k.
        let root: Vec<f64> = (0..=d).map(|k| sqrt_size[k]).collect();
        let mut traps = vec![C64::new(0.0, 0.0); n];
        let mut losses = vec![C64::new(0.0, 0.0); n];
        let mut lost = Vec::new();
        for k in 0..=d {
            traps[k] = C64::new(root[k] * 2.0 * sites.kappa[k], 0.0);
            if sites.losses[k] > 0.0 {
                losses[k] = C64::new(root[k] * sites.losses[k], 0.0);
                lost.push((k, sites.losses[k] * root[k]));
            }
        }
        let trapped = sites
            .traps
            .iter()
            .map(|&(k, kappa)| (k, 2.0 * kappa * root[k]))
            .collect();
        let sqrt = sqrt_size.clone();
        let bright = Box::new(move |p: &[C64]| -> (Vec<C64>, f64) {
            // q_w = Π_B[u, 0…0] for |u| = w: the plain coefficient of the
            // orbital (n₀₁, n₁₀, n₁₁) = (0, w, 0).
            let q: Vec<C64> = (0..=d).map(|w| p[index(d, 0, w, 0)]).collect();
            let qn = (0..=d)
                .map(|w| binomial(d, w) * q[w].norm_sqr())
                .sum::<f64>();
            let mut x0 = vec![C64::new(0.0, 0.0); sqrt.len()];
            for n01 in 0..=d {
                for n10 in 0..=d - n01 {
                    for n11 in 0..=d - n01 - n10 {
                        let o = index(d, n01, n10, n11);
                        x0[o] = q[n10 + n11] * q[n01 + n11].conj() * sqrt[o];
                    }
                }
            }
            (x0, qn)
        });
        Ok(OrbitalProblem {
            m: self.generator.clone(),
            coherence,
            hopping: self.gamma,
            traps,
            losses: if lost.is_empty() { None } else { Some(losses) },
            start: 0,
            all_lose: sites.losses.iter().all(|&g| g > 0.0),
            trapped,
            lost,
            sqrt_size,
            bright,
        })
    }

    /// Efficiency, mean arrival time, losses and dark weight of a walker
    /// started at `0…0`, by linear solves on the orbitals (see
    /// [`crate::walks::transport`]); needs a trap. Up to `d = 61` (the
    /// banded factorisation of the preconditioner needs at most
    /// [`crate::walks::transport::MAX_TRANSPORT_FACTOR_ENTRIES`] entries).
    /// Without dephasing at interior weights the walk stays on the
    /// `d + 1` symmetric states, and the [`crate::walks::DephasingWalk`]
    /// on the path with couplings `−√((k + 1)(d − k))`
    /// ([`crate::walks::Generator::Adjacency`]) gives the same result at
    /// any `d`. Coherent transport on a large cube is ill-conditioned:
    /// the chain's eigenstates near its ends reach the trap at rates of
    /// order `2^{−d}`, and `τ` depends on their tiny populations, so it
    /// loses digits that [`TransportResult::error`] (an estimate for `η`)
    /// does not show (`Q₄₀` without dephasing: `τ` to about `10⁻⁴`
    /// relative here, `4·10⁻⁷` on the chain).
    pub fn transport(&self, options: &TransportOptions) -> Result<TransportResult> {
        self.problem(options)?.transport(options)
    }

    /// One result per rate `e ≥ 0` (in input order), with `e` added to
    /// every node's dephasing; one factorisation per rate.
    pub fn dephasing_scan(
        &self,
        rates: &[f64],
        options: &TransportOptions,
    ) -> Result<Vec<TransportResult>> {
        self.problem(options)?.scan(rates, options)
    }

    /// The uniform rate in `[low, high]` (`0 < low < high`, finite) which,
    /// added to every node's dephasing, gives the highest efficiency.
    pub fn optimal_dephasing(
        &self,
        low: f64,
        high: f64,
        options: &TransportOptions,
    ) -> Result<DephasingOptimum> {
        self.problem(options)?.optimum(low, high, options)
    }

    /// Propagated populations per Hamming weight `0..=d`, sink populations
    /// (per listed trap weight) and the population lost at each of `times`
    /// (finite, `≥ 0`, non-decreasing), from `0…0`.
    pub fn transport_trace(&self, times: &[f64]) -> Result<Vec<TransportState>> {
        let d = self.d;
        let root: Vec<f64> = (0..=d).map(|k| binomial(d, k).sqrt()).collect();
        let mut reads = TraceReads {
            populations: (0..=d).map(|k| (k, root[k])).collect(),
            trapped: Vec::new(),
            lost: Vec::new(),
        };
        if let Some(sites) = &self.sites {
            reads.trapped = sites
                .traps
                .iter()
                .map(|&(k, kappa)| (k, 2.0 * kappa * root[k]))
                .collect();
            reads.lost = (0..=d)
                .filter(|&k| sites.losses[k] > 0.0)
                .map(|k| (k, sites.losses[k] * root[k]))
                .collect();
        }
        trace(&self.generator, &self.localized(), times, &reads)
    }

    /// Number of coordinates.
    pub fn dimension(&self) -> usize {
        self.d
    }

    pub fn num_orbitals(&self) -> usize {
        count(self.d).unwrap()
    }

    /// The orbital of the pairs of nodes with `n01`, `n10` and `n11`
    /// coordinates of bits `01`, `10` and `11`.
    pub fn orbital(&self, n01: usize, n10: usize, n11: usize) -> Result<usize> {
        if n01 + n10 + n11 > self.d {
            return Err(QsoError::invalid(
                "orbital",
                format!("{n01} + {n10} + {n11} coordinates exceed d = {}", self.d),
            ));
        }
        Ok(index(self.d, n01, n10, n11))
    }

    /// The walker at node `0…0`.
    pub fn localized(&self) -> Vec<C64> {
        let mut x = vec![C64::new(0.0, 0.0); self.num_orbitals()];
        x[0] = C64::new(1.0, 0.0);
        x
    }

    /// The uniform superposition of all nodes: every entry `2^{-d}`.
    pub fn uniform(&self) -> Vec<C64> {
        let d = self.d;
        let half = (d as f64) * std::f64::consts::LN_2;
        let mut x = vec![C64::new(0.0, 0.0); self.num_orbitals()];
        for n01 in 0..=d {
            for n10 in 0..=d - n01 {
                for n11 in 0..=d - n01 - n10 {
                    let size = log_size(d, [d - n01 - n10 - n11, n01, n10, n11]);
                    x[index(d, n01, n10, n11)] = C64::new((0.5 * size - half).exp(), 0.0);
                }
            }
        }
        x
    }

    /// The entry `ρ_uv` of any pair `(u, v)` of the orbital with `n01`,
    /// `n10` and `n11` coordinates of bits `01`, `10` and `11`.
    pub fn entry(&self, x: &[C64], n01: usize, n10: usize, n11: usize) -> Result<C64> {
        self.check_len(x.len())?;
        let o = self.orbital(n01, n10, n11)?;
        let size = log_size(self.d, [self.d - n01 - n10 - n11, n01, n10, n11]);
        Ok(x[o] * (-0.5 * size).exp())
    }

    /// The coefficients after time `t ≥ 0` from `x0`.
    pub fn evolve(&self, x0: &[C64], t: f64) -> Result<Vec<C64>> {
        self.check_len(x0.len())?;
        if !(t.is_finite() && t >= 0.0) {
            return Err(QsoError::invalid("t", "time must be finite and ≥ 0"));
        }
        if x0.iter().any(|z| !z.is_finite()) {
            return Err(QsoError::invalid("x0", "coefficients must be finite"));
        }
        self.generator.exp_apply(x0, t)
    }

    /// The probability of every distance `k` from `0…0`: `√C(d, k)` times
    /// the coefficient of the orbital `(n₀₁, n₁₀, n₁₁) = (0, 0, k)` (see
    /// the type's documentation for its accuracy).
    pub fn distance_distribution(&self, x: &[C64]) -> Result<Vec<f64>> {
        self.check_len(x.len())?;
        let d = self.d;
        Ok((0..=d)
            .map(|k| (0.5 * log_size(d, [d - k, 0, 0, k])).exp() * x[index(d, 0, 0, k)].re)
            .collect())
    }

    fn check_len(&self, len: usize) -> Result<()> {
        if len != self.num_orbitals() {
            return Err(QsoError::DimensionMismatch {
                expected: self.num_orbitals(),
                actual: len,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::generators;
    use crate::walks::Generator;
    use crate::walks::symmetric::{Dephasing, SymmetricOpenWalk, coordinate_permutations};

    fn bloch_z(gamma: f64, kappa: f64, t: f64) -> f64 {
        let w = (4.0 * gamma * gamma - kappa * kappa / 4.0).sqrt();
        (-kappa * t / 2.0).exp() * ((w * t).cos() + kappa / (2.0 * w) * (w * t).sin())
    }

    #[test]
    fn orbital_indices_enumerate_the_orbitals() {
        for d in 1..=12 {
            let walk = HypercubeOpenWalk::new(d, 1.0, 0.0, 0.0).unwrap();
            let mut seen = vec![false; walk.num_orbitals()];
            for n01 in 0..=d {
                for n10 in 0..=d - n01 {
                    for n11 in 0..=d - n01 - n10 {
                        let k = walk.orbital(n01, n10, n11).unwrap();
                        assert!(!std::mem::replace(&mut seen[k], true));
                    }
                }
            }
            assert!(seen.iter().all(|&s| s));
        }
        assert_eq!(count(230), Some(2_081_156));
        assert!(HypercubeOpenWalk::new(231, 1.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn matches_the_explicit_reduction_on_q6() {
        let (d, t) = (6, 1.4);
        let n = 1 << d;
        let g = generators::hypercube(d);
        let mut dephasing = vec![Dephasing::sites(n, 0.3)];
        dephasing.extend((0..d).map(|k| Dephasing::coordinate(n, k, 0.7)));
        for generator in [Generator::Laplacian, Generator::Adjacency] {
            let explicit = SymmetricOpenWalk::new(
                &g,
                generator,
                0.9,
                &dephasing,
                &coordinate_permutations(d).unwrap(),
            )
            .unwrap();
            let x = explicit.evolve(&explicit.localized(0).unwrap(), t).unwrap();
            let implicit = HypercubeOpenWalk::new(d, 0.9, 0.3, 0.7).unwrap();
            let y = implicit.evolve(&implicit.localized(), t).unwrap();
            let o = explicit.orbitals();
            for u in 0..n {
                for v in 0..n {
                    let n01 = (!u & v & (n - 1)).count_ones() as usize;
                    let n10 = (u & !v & (n - 1)).count_ones() as usize;
                    let n11 = (u & v).count_ones() as usize;
                    let a = x[o.of(u, v)];
                    let b = implicit.entry(&y, n01, n10, n11).unwrap();
                    assert!((a - b).norm() < 1e-12, "({u}, {v}): {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn coordinate_dephasing_at_sixty_coordinates() {
        let (d, gamma, kappa, t) = (60, 0.7, 0.9, 1.3);
        let walk = HypercubeOpenWalk::new(d, gamma, 0.0, kappa).unwrap();
        let p = walk
            .distance_distribution(&walk.evolve(&walk.localized(), t).unwrap())
            .unwrap();
        let flip = (1.0 - bloch_z(gamma, kappa, t)) / 2.0;
        let mut binomial = 1.0f64;
        for (k, &pk) in p.iter().enumerate() {
            let expected = binomial * flip.powi(k as i32) * (1.0 - flip).powi((d - k) as i32);
            let tol = 1e-15 * binomial.sqrt();
            assert!((pk - expected).abs() < tol, "k = {k}: {pk} vs {expected}");
            binomial *= (d - k) as f64 / (k + 1) as f64;
        }
    }

    #[test]
    fn mixes_uniformly_at_the_moore_russell_time() {
        // Without dephasing every coordinate flips with sin²(γt): at
        // γt = π/4 every node has probability 2^{-d} (Moore and Russell).
        let d = 64;
        let walk = HypercubeOpenWalk::new(d, 1.0, 0.0, 0.0).unwrap();
        let x = walk
            .evolve(&walk.localized(), std::f64::consts::FRAC_PI_4)
            .unwrap();
        // The state stays pure: Tr ρ² = Σ |x_O|² = 1.
        let purity: f64 = x.iter().map(|z| z.norm_sqr()).sum();
        assert!((purity - 1.0).abs() < 1e-12, "Tr ρ² = {purity}");
        let p = walk.distance_distribution(&x).unwrap();
        let mut binomial = 1.0f64;
        for (k, &pk) in p.iter().enumerate() {
            let expected = binomial * 0.5f64.powi(d as i32);
            let tol = 1e-15 * binomial.sqrt();
            assert!((pk - expected).abs() < tol, "k = {k}: {pk} vs {expected}");
            binomial *= (d - k) as f64 / (k + 1) as f64;
        }
        // Every entry has modulus 2^{-d}, to rounding in its orbital.
        for (n01, n10, n11) in [(0, 0, 0), (0, 0, 32), (5, 7, 11), (32, 0, 0), (20, 20, 4)] {
            let e = walk.entry(&x, n01, n10, n11).unwrap();
            let size = (0.5 * log_size(d, [d - n01 - n10 - n11, n01, n10, n11])).exp();
            assert!((e.norm() - 0.5f64.powi(d as i32)).abs() < 1e-15 / size);
        }
    }

    #[test]
    fn the_maximally_mixed_state_is_stationary() {
        let d = 20;
        let walk = HypercubeOpenWalk::new(d, 1.0, 0.4, 0.3).unwrap();
        let mut mixed = vec![C64::new(0.0, 0.0); walk.num_orbitals()];
        let mut binomial = 1.0f64;
        for k in 0..=d {
            mixed[walk.orbital(0, 0, k).unwrap()] =
                C64::new(binomial.sqrt() * 0.5f64.powi(d as i32), 0.0);
            binomial *= (d - k) as f64 / (k + 1) as f64;
        }
        let x = walk.evolve(&mixed, 3.0).unwrap();
        assert!(x.iter().zip(&mixed).all(|(a, b)| (a - b).norm() < 1e-15));
        // The uniform superposition is an eigenstate without dephasing.
        let coherent = HypercubeOpenWalk::new(d, 1.0, 0.0, 0.0).unwrap();
        let u = coherent.uniform();
        assert!((u.iter().map(|z| z.norm_sqr()).sum::<f64>() - 1.0).abs() < 1e-12);
        let x = coherent.evolve(&u, 2.0).unwrap();
        assert!(x.iter().zip(&u).all(|(a, b)| (a - b).norm() < 1e-13));
        // Dephasing drives the walker there from node 0.
        let x = walk.evolve(&walk.localized(), 60.0).unwrap();
        let p = walk.distance_distribution(&x).unwrap();
        let q = walk.distance_distribution(&mixed).unwrap();
        assert!(p.iter().zip(&q).all(|(a, b)| (a - b).abs() < 1e-9));
    }

    #[test]
    fn rejects_invalid_input() {
        assert!(HypercubeOpenWalk::new(0, 1.0, 0.0, 0.0).is_err());
        assert!(HypercubeOpenWalk::new(3, 0.0, 0.0, 0.0).is_err());
        assert!(HypercubeOpenWalk::new(3, 1.0, -1.0, 0.0).is_err());
        assert!(HypercubeOpenWalk::new(3, 1.0, 0.0, f64::NAN).is_err());
        let walk = HypercubeOpenWalk::new(3, 1.0, 0.0, 0.0).unwrap();
        assert!(walk.orbital(2, 2, 0).is_err());
        assert!(walk.evolve(&walk.localized()[1..], 1.0).is_err());
        assert!(walk.evolve(&walk.localized(), -1.0).is_err());
        assert!(walk.distance_distribution(&[]).is_err());
        assert!(walk.entry(&walk.localized(), 4, 0, 0).is_err());
    }
}
