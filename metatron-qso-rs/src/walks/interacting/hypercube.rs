//! Interacting particles on the hypercube, on orbits of the coordinate
//! permutations that are written down without building the graph.
//!
//! `N` particles on `Q_d` sit at nodes `x₁, …, x_N ∈ {0,1}^d`. Read as a
//! `d × N` matrix of bits, a configuration has a *pattern* `p ∈ {0,1}^N` in
//! every coordinate, and the permutations of the coordinates leave only
//! the number `n_p` of coordinates with each pattern: a composition of `d`
//! into `2^N` parts, `C(d + 2^N − 1, 2^N − 1)` of them however large `2^d`
//! is. Particle `j` hopping along coordinate `k` turns that coordinate's
//! pattern `p` into `p ⊕ e_j`; particles `i` and `j` are `Σ_{p_i ≠ p_j} n_p`
//! apart. For bosons and fermions the permutations of the particles act on
//! the patterns too, and their orbits of compositions (with the sign of the
//! permutation for fermions) are the states.
//!
//! States invariant under the coordinate permutations include every
//! particle at node `0…0`, and the walk keeps them invariant: on `Q₂₀`
//! three bosons have 1.9 · 10¹⁷ configurations but 156 240 states.

use std::sync::OnceLock;

use num_complex::Complex64 as C64;

use super::{Statistics, propagate};
use crate::error::{QsoError, Result};
use crate::linalg::{CsrMatrix, RMatrix, eigh_real};
use crate::walks::{DENSE_LIMIT, Generator};

/// Most particles on the hypercube (`2⁴ = 16` patterns).
pub const MAX_HYPERCUBE_PARTICLES: usize = 4;

/// Most compositions of `d` into `2^N` parts that are enumerated (a 4-byte
/// word each, 128 MiB).
pub const MAX_HYPERCUBE_COMPOSITIONS: usize = 1 << 25;

const SIGN: u32 = 1 << 31;
const DEAD: u32 = SIGN - 1;

/// `C(m, k)` for `m ≤ top`, `k ≤ kmax`, saturating.
fn binomials(top: usize, kmax: usize) -> Vec<Vec<usize>> {
    let mut b = vec![vec![0usize; kmax + 1]; top + 1];
    for m in 0..=top {
        b[m][0] = 1;
        for k in 1..=kmax.min(m) {
            b[m][k] = b[m - 1][k - 1].saturating_add(if k < m { b[m - 1][k] } else { 0 });
        }
    }
    b
}

/// Ranks of compositions of `d` into `parts` parts: the colex rank of the
/// set of bars `B_k = n₀ + … + n_k + k`, `k < parts − 1`.
#[derive(Clone, Debug)]
struct Compositions {
    parts: usize,
    binom: Vec<Vec<usize>>,
}

impl Compositions {
    fn rank(&self, counts: &[u16]) -> usize {
        let mut prefix = 0;
        let mut rank = 0;
        for (k, &c) in counts[..self.parts - 1].iter().enumerate() {
            prefix += c as usize;
            rank += self.binom[prefix + k][k + 1];
        }
        rank
    }
}

/// The permutations of `0..n`, by Heap's algorithm, with their signs.
fn permutations(n: usize) -> Vec<(Vec<usize>, bool)> {
    fn heap(k: usize, a: &mut Vec<usize>, odd: bool, out: &mut Vec<(Vec<usize>, bool)>) {
        if k <= 1 {
            out.push((a.clone(), odd));
            return;
        }
        let mut odd = odd;
        for i in 0..k {
            heap(k - 1, a, odd, out);
            if i + 1 < k {
                if k.is_multiple_of(2) {
                    a.swap(i, k - 1);
                } else {
                    a.swap(0, k - 1);
                }
                odd = !odd;
            }
        }
    }
    let mut out = Vec::new();
    heap(n, &mut (0..n).collect(), false, &mut out);
    out
}

/// The continuous-time walk of `N` interacting particles on the hypercube
/// `Q_d`, on states invariant under the permutations of the coordinates
/// (see the module documentation): hopping `h = −γA` or `γL` (they differ
/// by `γdN` on the hypercube, a phase), on-site interaction U and
/// interaction V between particles on adjacent nodes.
///
/// ```
/// use metatron_qso::walks::{Generator, HypercubeParticles, Statistics};
///
/// // Two bosons start together at node 0 of Q₄₀ (2⁴⁰ nodes) and repel
/// // each other: the distance between them spreads.
/// let walk = HypercubeParticles::new(40, Statistics::Bosons, 2, Generator::Adjacency, 1.0)
///     .unwrap()
///     .with_interaction(2.0, 0.0)
///     .unwrap();
/// let x = walk.evolve(&walk.localized().unwrap(), 0.5).unwrap();
/// let distances = walk.pair_distances(&x).unwrap();
/// assert_eq!(distances.len(), 41);
/// assert!((distances.iter().sum::<f64>() - 1.0).abs() < 1e-10);
/// ```
#[derive(Clone, Debug)]
pub struct HypercubeParticles {
    d: usize,
    particles: usize,
    statistics: Statistics,
    generator: Generator,
    gamma: f64,
    on_site: f64,
    neighbour: f64,
    patterns: usize,
    compositions: Compositions,
    /// Per composition (by rank): its class, or [`DEAD`], and the sign of
    /// the particle permutation to its class's representative (bit 31).
    of: Vec<u32>,
    /// Counts of the representative of every class, `patterns` each.
    counts: Vec<u16>,
    /// Compositions in every class (orbit sizes under the particle
    /// permutations).
    images: Vec<u32>,
    hamiltonian: CsrMatrix,
    spectral: OnceLock<Option<(Vec<f64>, RMatrix)>>,
}

impl HypercubeParticles {
    /// `particles` (1 to [`MAX_HYPERCUBE_PARTICLES`]) particles of the
    /// given statistics on `Q_d` with hopping rate `gamma`, no interaction.
    pub fn new(
        d: usize,
        statistics: Statistics,
        particles: usize,
        generator: Generator,
        gamma: f64,
    ) -> Result<Self> {
        if d == 0 {
            return Err(QsoError::invalid("d", "need at least one coordinate"));
        }
        if !(1..=MAX_HYPERCUBE_PARTICLES).contains(&particles) {
            return Err(QsoError::invalid(
                "particles",
                format!("1 to {MAX_HYPERCUBE_PARTICLES} particles on the hypercube"),
            ));
        }
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(QsoError::invalid(
                "gamma",
                "hopping rate must be positive and finite",
            ));
        }
        if d > u16::MAX as usize {
            return Err(QsoError::invalid("d", "at most 65 535 coordinates"));
        }
        let patterns = 1usize << particles;
        let top = d + patterns;
        let binom = binomials(top, patterns);
        let total = binom[d + patterns - 1][patterns - 1];
        if total > MAX_HYPERCUBE_COMPOSITIONS {
            return Err(QsoError::invalid(
                "d",
                format!(
                    "C(d + {}, {}) compositions exceed {MAX_HYPERCUBE_COMPOSITIONS}",
                    patterns - 1,
                    patterns - 1
                ),
            ));
        }
        let compositions = Compositions {
            parts: patterns,
            binom,
        };
        // Particle permutations act on the patterns: bit j to bit π(j).
        // Distinguishable particles are not permuted.
        let perms = if statistics == Statistics::Distinguishable {
            vec![((0..particles).collect::<Vec<_>>(), false)]
        } else {
            permutations(particles)
        };
        let moved: Vec<(Vec<usize>, bool)> = perms
            .iter()
            .map(|(pi, odd)| {
                let image = (0..patterns)
                    .map(|p| {
                        (0..particles)
                            .filter(|&j| p >> j & 1 == 1)
                            .fold(0, |q, j| q | 1 << pi[j])
                    })
                    .collect();
                (image, *odd)
            })
            .collect();
        let mut of = vec![DEAD; total];
        let mut counts: Vec<u16> = Vec::new();
        let mut images: Vec<u32> = Vec::new();
        // Enumerate the compositions in rank order: the bars B₀ < … <
        // B_{P−2} of {0, …, d + P − 2}, colex successor.
        let bars_len = patterns - 1;
        let mut bars: Vec<usize> = (0..bars_len).collect();
        let mut n = vec![0u16; patterns];
        let mut m = vec![0u16; patterns];
        for rank in 0..total {
            // Counts from the bars.
            let mut previous: isize = -1;
            for (k, &b) in bars.iter().enumerate() {
                n[k] = (b as isize - previous - 1) as u16;
                previous = b as isize;
            }
            n[patterns - 1] = (d + patterns - 2 - previous as usize) as u16;
            debug_assert_eq!(compositions.rank(&n), rank);
            // The lowest rank of the composition's images, and whether an
            // odd permutation fixes it (fermions: no state) or maps it to
            // that image.
            let (mut best, mut best_odd, mut odd_fixes) = (rank, false, false);
            let mut stabiliser = 0u32;
            for (image, odd) in &moved {
                for (p, &c) in n.iter().enumerate() {
                    m[image[p]] = c;
                }
                let r = compositions.rank(&m);
                if r == rank {
                    stabiliser += 1;
                    odd_fixes |= *odd;
                }
                if r < best {
                    (best, best_odd) = (r, *odd);
                }
            }
            if best == rank {
                // A new class, unless fermions vanish on it.
                if statistics == Statistics::Fermions && odd_fixes {
                    of[rank] = DEAD;
                } else {
                    of[rank] = images.len() as u32;
                    counts.extend_from_slice(&n);
                    images.push(moved.len() as u32 / stabiliser);
                }
            } else {
                let label = of[best];
                let sign = if statistics == Statistics::Fermions && best_odd {
                    SIGN
                } else {
                    0
                };
                of[rank] = if label == DEAD { DEAD } else { label | sign };
            }
            // Colex successor of the bars.
            for k in 0..bars_len {
                let limit = if k + 1 < bars_len {
                    bars[k + 1]
                } else {
                    d + patterns - 1
                };
                if bars[k] + 1 < limit {
                    bars[k] += 1;
                    for (j, b) in bars[..k].iter_mut().enumerate() {
                        *b = j;
                    }
                    break;
                }
            }
        }
        let mut walk = HypercubeParticles {
            d,
            particles,
            statistics,
            generator,
            gamma,
            on_site: 0.0,
            neighbour: 0.0,
            patterns,
            compositions,
            of,
            counts,
            images,
            hamiltonian: CsrMatrix::from_triplets(0, &[])?,
            spectral: OnceLock::new(),
        };
        walk.build()?;
        Ok(walk)
    }

    /// The same walk with on-site interaction `on_site` (U) and interaction
    /// `neighbour` (V) between particles on adjacent nodes.
    pub fn with_interaction(mut self, on_site: f64, neighbour: f64) -> Result<Self> {
        if !(on_site.is_finite() && neighbour.is_finite()) {
            return Err(QsoError::invalid(
                "interaction",
                "interaction strengths must be finite",
            ));
        }
        if (on_site, neighbour) != (self.on_site, self.neighbour) {
            self.on_site = on_site;
            self.neighbour = neighbour;
            self.build()?;
        }
        Ok(self)
    }

    pub fn dimension(&self) -> usize {
        self.d
    }

    pub fn particles(&self) -> usize {
        self.particles
    }

    pub fn statistics(&self) -> Statistics {
        self.statistics
    }

    pub fn generator(&self) -> Generator {
        self.generator
    }

    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    pub fn on_site(&self) -> f64 {
        self.on_site
    }

    pub fn neighbour(&self) -> f64 {
        self.neighbour
    }

    /// Number of states: the dimension of the reduced walk.
    pub fn num_states(&self) -> usize {
        self.images.len()
    }

    /// The reduced Hamiltonian.
    pub fn hamiltonian(&self) -> &CsrMatrix {
        &self.hamiltonian
    }

    /// Eigenvalues of the reduced Hamiltonian (ascending), up to
    /// [`DENSE_LIMIT`] states.
    pub fn spectrum(&self) -> Option<&[f64]> {
        self.spectral().map(|(v, _)| v.as_slice())
    }

    /// Every particle at node `0…0` (bosons and distinguishable particles,
    /// or a single fermion).
    pub fn localized(&self) -> Result<Vec<C64>> {
        let mut counts = vec![0usize; self.patterns];
        counts[0] = self.d;
        self.state_of(&counts)
    }

    /// The invariant state of the configurations with `counts[p]`
    /// coordinates of pattern `p` (bit `j` of `p`: the coordinate of
    /// particle `j`), signed so that those configurations have positive
    /// amplitudes. An error if no invariant state contains them (fermions
    /// that coincide, or that a swap maps to minus themselves).
    pub fn state_of(&self, counts: &[usize]) -> Result<Vec<C64>> {
        if counts.len() != self.patterns {
            return Err(QsoError::DimensionMismatch {
                expected: self.patterns,
                actual: counts.len(),
            });
        }
        if counts.iter().sum::<usize>() != self.d {
            return Err(QsoError::invalid(
                "counts",
                format!("the counts must add up to d = {}", self.d),
            ));
        }
        let n: Vec<u16> = counts.iter().map(|&c| c as u16).collect();
        let word = self.of[self.compositions.rank(&n)];
        if word & !SIGN == DEAD {
            return Err(QsoError::invalid(
                "counts",
                "no invariant state contains these configurations",
            ));
        }
        let mut x = vec![C64::new(0.0, 0.0); self.num_states()];
        x[(word & !SIGN) as usize] = C64::new(if word & SIGN == 0 { 1.0 } else { -1.0 }, 0.0);
        Ok(x)
    }

    /// `exp(−iHt)` applied to the state `x0`.
    pub fn evolve(&self, x0: &[C64], t: f64) -> Result<Vec<C64>> {
        self.check_state(x0)?;
        propagate(&self.hamiltonian, self.spectral(), x0, t)
    }

    /// Probability that every particle is at the node opposite `0…0`.
    pub fn antipode_probability(&self, x: &[C64]) -> Result<f64> {
        self.check_state(x)?;
        let mut counts = vec![0u16; self.patterns];
        counts[self.patterns - 1] = self.d as u16;
        let word = self.of[self.compositions.rank(&counts)];
        Ok(if word & !SIGN == DEAD {
            0.0
        } else {
            x[(word & !SIGN) as usize].norm_sqr()
        })
    }

    /// Expected number of particles at each Hamming weight `0..=d` (they
    /// add up to the number of particles).
    pub fn weights(&self, x: &[C64]) -> Result<Vec<f64>> {
        self.check_state(x)?;
        let mut out = vec![0.0; self.d + 1];
        for (c, amp) in x.iter().enumerate() {
            let p = amp.norm_sqr();
            let n = self.class_counts(c);
            for j in 0..self.particles {
                let w: usize = (0..self.patterns)
                    .filter(|&q| q >> j & 1 == 1)
                    .map(|q| n[q] as usize)
                    .sum();
                out[w] += p;
            }
        }
        Ok(out)
    }

    /// Distribution of the Hamming distance between two particles, averaged
    /// over the pairs (it adds up to 1; at least two particles).
    pub fn pair_distances(&self, x: &[C64]) -> Result<Vec<f64>> {
        self.check_state(x)?;
        if self.particles < 2 {
            return Err(QsoError::invalid(
                "particles",
                "pair distances need two particles",
            ));
        }
        let pairs = (self.particles * (self.particles - 1) / 2) as f64;
        let mut out = vec![0.0; self.d + 1];
        for (c, amp) in x.iter().enumerate() {
            let p = amp.norm_sqr() / pairs;
            let n = self.class_counts(c);
            for i in 0..self.particles {
                for j in i + 1..self.particles {
                    out[self.distance(n, i, j)] += p;
                }
            }
        }
        Ok(out)
    }

    fn class_counts(&self, c: usize) -> &[u16] {
        &self.counts[c * self.patterns..(c + 1) * self.patterns]
    }

    /// Hamming distance between particles `i` and `j`.
    fn distance(&self, n: &[u16], i: usize, j: usize) -> usize {
        (0..self.patterns)
            .filter(|&p| (p >> i ^ p >> j) & 1 == 1)
            .map(|p| n[p] as usize)
            .sum()
    }

    fn check_state(&self, x: &[C64]) -> Result<()> {
        if x.len() != self.num_states() {
            return Err(QsoError::DimensionMismatch {
                expected: self.num_states(),
                actual: x.len(),
            });
        }
        Ok(())
    }

    fn spectral(&self) -> Option<&(Vec<f64>, RMatrix)> {
        self.spectral
            .get_or_init(|| {
                (self.num_states() <= DENSE_LIMIT).then(|| eigh_real(&self.hamiltonian.to_dense()))
            })
            .as_ref()
    }

    /// The reduced Hamiltonian, read off at the representatives: a hop of
    /// particle `j` in one of the `n_p` coordinates of pattern `p` leads to
    /// the composition with one `p` fewer and one `p ⊕ e_j` more, with
    /// weight `−γ √(n_p (n_q + 1) k_n / k_m)` (`k`: compositions per
    /// class).
    fn build(&mut self) -> Result<()> {
        let classes = self.num_states();
        let shift = match self.generator {
            Generator::Laplacian => self.gamma * (self.d * self.particles) as f64,
            Generator::Adjacency => 0.0,
        };
        let mut triplets = Vec::with_capacity(classes * (1 + self.patterns * self.particles));
        let mut m = vec![0u16; self.patterns];
        for c in 0..classes {
            let n = self.class_counts(c).to_vec();
            let mut diagonal = shift;
            for i in 0..self.particles {
                for j in i + 1..self.particles {
                    match self.distance(&n, i, j) {
                        0 => diagonal += self.on_site,
                        1 => diagonal += self.neighbour,
                        _ => {}
                    }
                }
            }
            if diagonal != 0.0 {
                triplets.push((c, c, diagonal));
            }
            let k_n = self.images[c] as f64;
            for p in 0..self.patterns {
                if n[p] == 0 {
                    continue;
                }
                for j in 0..self.particles {
                    let q = p ^ (1 << j);
                    m.copy_from_slice(&n);
                    m[p] -= 1;
                    m[q] += 1;
                    let word = self.of[self.compositions.rank(&m)];
                    if word & !SIGN == DEAD {
                        continue;
                    }
                    let target = (word & !SIGN) as usize;
                    let sign = if word & SIGN == 0 { 1.0 } else { -1.0 };
                    let k_m = self.images[target] as f64;
                    let weight =
                        (n[p] as f64 * (n[q] as f64 + 1.0) * k_n / k_m).sqrt() * -self.gamma;
                    triplets.push((target, c, sign * weight));
                }
            }
        }
        self.hamiltonian = CsrMatrix::from_triplets(classes, &triplets)?;
        self.spectral = OnceLock::new();
        Ok(())
    }
}
