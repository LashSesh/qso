//! Boson sampling: exact output probabilities and exact sampling.

use super::permanent::glynn;
use super::{MAX_PHOTONS, check_modes, unitarity_error};
use crate::error::{QsoError, Result};
use crate::linalg::CMatrix;
use crate::parallel;
use crate::rng::sample_cumulative;
use crate::{C64, Rng};

/// Largest number of output patterns [`occupation_patterns`] enumerates.
pub const MAX_PATTERNS: usize = 2_000_000;

/// Largest number of occupation numbers (patterns or samples × modes) a
/// single call returns, to keep results within a few hundred megabytes.
pub const MAX_OUTPUT_ENTRIES: usize = 1 << 26;

/// Shots per deterministic work unit in [`BosonSampler::sample`].
const SHOT_CHUNK: usize = 64;

/// Exact output distribution of a [`BosonSampler`].
#[derive(Clone, Debug, PartialEq)]
pub struct Distribution {
    /// Every output occupation pattern (lexicographically descending).
    pub patterns: Vec<Vec<usize>>,
    /// Probability of each pattern.
    pub probabilities: Vec<f64>,
}

/// `n` single photons sent through an `m`-mode interferometer.
///
/// Output probabilities are computed exactly from permanents, both for
/// indistinguishable photons (quantum interference) and for distinguishable
/// photons (the classical reference). Samples are drawn exactly with the
/// algorithm of Clifford & Clifford (2018) — not by enumerating the output
/// distribution — at a cost of `O(n²·2ⁿ + m·n²)` per sample.
#[derive(Clone, Debug)]
pub struct BosonSampler {
    unitary: CMatrix,
    input: Vec<usize>,
    /// Input mode of every photon (ascending, repeated per occupation).
    photon_modes: Vec<usize>,
}

impl BosonSampler {
    /// Interferometer `unitary` (`m × m`, unitary to 1e-9) with input
    /// occupation `input` (`input[j]` photons in mode `j`).
    pub fn new(unitary: CMatrix, input: &[usize]) -> Result<Self> {
        let m = unitary.nrows();
        check_modes(m)?;
        if unitary.ncols() != m {
            return Err(QsoError::invalid(
                "unitary",
                format!("must be square, got {} × {}", m, unitary.ncols()),
            ));
        }
        if unitary
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(QsoError::invalid("unitary", "entries must be finite"));
        }
        let err = unitarity_error(&unitary);
        if err > 1e-9 {
            return Err(QsoError::invalid(
                "unitary",
                format!("matrix is not unitary (max |U†U − I| = {err:.2e})"),
            ));
        }
        if input.len() != m {
            return Err(QsoError::DimensionMismatch {
                expected: m,
                actual: input.len(),
            });
        }
        let n = checked_total(input).filter(|n| (1..=MAX_PHOTONS).contains(n));
        if n.is_none() {
            return Err(QsoError::invalid(
                "input",
                format!("number of photons must lie in 1..={MAX_PHOTONS}"),
            ));
        }
        let photon_modes: Vec<usize> = input
            .iter()
            .enumerate()
            .flat_map(|(j, &s)| std::iter::repeat_n(j, s))
            .collect();
        Ok(Self {
            unitary,
            input: input.to_vec(),
            photon_modes,
        })
    }

    /// Number of modes `m`.
    pub fn n_modes(&self) -> usize {
        self.unitary.nrows()
    }

    /// Number of photons `n`.
    pub fn n_photons(&self) -> usize {
        self.photon_modes.len()
    }

    /// Input occupation.
    pub fn input(&self) -> &[usize] {
        &self.input
    }

    /// The interferometer.
    pub fn unitary(&self) -> &CMatrix {
        &self.unitary
    }

    /// Transition amplitude `perm(U_{T,S}) / √(Π s_j! Π t_i!)` to the output
    /// occupation `output`.
    pub fn amplitude(&self, output: &[usize]) -> Result<C64> {
        let rows = self.output_rows(output)?;
        let sub = self.submatrix(&rows, |z| z);
        let norm = factorial_product(&self.input) * factorial_product(output);
        Ok(glynn(&sub, rows.len()) / norm.sqrt())
    }

    /// Probability of `output` for indistinguishable photons.
    pub fn probability(&self, output: &[usize]) -> Result<f64> {
        Ok(self.amplitude(output)?.norm_sqr())
    }

    /// Probability of `output` for distinguishable photons,
    /// `perm(|U_{T,S}|²) / Π t_i!`.
    pub fn probability_distinguishable(&self, output: &[usize]) -> Result<f64> {
        let rows = self.output_rows(output)?;
        let sub: Vec<f64> = self.submatrix(&rows, |z| z.norm_sqr());
        Ok(glynn(&sub, rows.len()) / factorial_product(output))
    }

    /// Exact output distribution for indistinguishable photons (all
    /// `C(m+n−1, n)` patterns; see [`count_patterns`]).
    pub fn distribution(&self) -> Result<Distribution> {
        self.tabulate(|s, t| s.probability(t))
    }

    /// Exact output distribution for distinguishable photons.
    pub fn distribution_distinguishable(&self) -> Result<Distribution> {
        self.tabulate(|s, t| s.probability_distinguishable(t))
    }

    fn tabulate(&self, p: impl Fn(&Self, &[usize]) -> Result<f64> + Sync) -> Result<Distribution> {
        let patterns = occupation_patterns(self.n_photons(), self.n_modes())?;
        let probabilities = parallel::map_tasks(patterns.len(), |k| p(self, &patterns[k]))
            .into_iter()
            .collect::<Result<Vec<f64>>>()?;
        Ok(Distribution {
            patterns,
            probabilities,
        })
    }

    /// `shots` exact samples for indistinguishable photons, as output
    /// occupations (Clifford & Clifford 2018, with the Laplace expansion of
    /// their algorithm B).
    ///
    /// Requires at most one photon per input mode, the standard boson
    /// sampling setting. Results depend only on the state of `rng`, not on
    /// the number of threads.
    pub fn sample(&self, shots: usize, rng: &mut Rng) -> Result<Vec<Vec<usize>>> {
        if self.input.iter().any(|&s| s > 1) {
            return Err(QsoError::invalid(
                "input",
                "exact sampling needs at most one photon per input mode",
            ));
        }
        self.check_shots(shots)?;
        Ok(self.sample_with(shots, rng, |s, rng| s.sample_one(rng)))
    }

    /// `shots` samples for distinguishable photons: every photon leaves its
    /// input mode `j` through output mode `i` with probability `|U_ij|²`,
    /// independently of the others.
    pub fn sample_distinguishable(&self, shots: usize, rng: &mut Rng) -> Result<Vec<Vec<usize>>> {
        self.check_shots(shots)?;
        let m = self.n_modes();
        let cumulative: Vec<Vec<f64>> = self
            .photon_modes
            .iter()
            .map(|&j| {
                let mut acc = 0.0;
                (0..m)
                    .map(|i| {
                        acc += self.unitary[(i, j)].norm_sqr();
                        acc
                    })
                    .collect()
            })
            .collect();
        Ok(self.sample_with(shots, rng, |s, rng| {
            let mut out = vec![0; s.n_modes()];
            for c in &cumulative {
                out[sample_cumulative(c, rng)] += 1;
            }
            out
        }))
    }

    fn check_shots(&self, shots: usize) -> Result<()> {
        match shots.checked_mul(self.n_modes()) {
            Some(entries) if entries <= MAX_OUTPUT_ENTRIES => Ok(()),
            _ => Err(QsoError::invalid(
                "shots",
                format!(
                    "{shots} shots of {} modes exceed the supported {MAX_OUTPUT_ENTRIES} \
                     occupation numbers per call",
                    self.n_modes()
                ),
            )),
        }
    }

    /// Run `one` for every shot. Shots are split into fixed chunks with one
    /// forked random stream each, so the result is independent of threads.
    fn sample_with(
        &self,
        shots: usize,
        rng: &mut Rng,
        one: impl Fn(&Self, &mut Rng) -> Vec<usize> + Sync,
    ) -> Vec<Vec<usize>> {
        let base = Rng::seed_from_u64(rng.next_u64());
        let chunks = shots.div_ceil(SHOT_CHUNK);
        parallel::map_tasks(chunks, |c| {
            let mut stream = base.fork(c as u64);
            let count = SHOT_CHUNK.min(shots - c * SHOT_CHUNK);
            (0..count)
                .map(|_| one(self, &mut stream))
                .collect::<Vec<_>>()
        })
        .into_iter()
        .flatten()
        .collect()
    }

    /// One exact sample. After a uniformly random permutation of the photons
    /// (columns), output mode `r_k` of photon `k` is drawn with weight
    /// `|perm(U[r_1..r_{k−1}, i; c_1..c_k])|²`; the permanent is expanded
    /// along its last row so that the `k` minors are computed once per step.
    fn sample_one(&self, rng: &mut Rng) -> Vec<usize> {
        let (m, n) = (self.n_modes(), self.n_photons());
        let mut cols = self.photon_modes.clone();
        rng.shuffle(&mut cols);
        let mut rows: Vec<usize> = Vec::with_capacity(n);
        let mut cumulative = vec![0.0; m];
        for k in 1..=n {
            let minors: Vec<C64> = (0..k)
                .map(|skip| {
                    let sub: Vec<C64> = rows
                        .iter()
                        .flat_map(|&r| {
                            cols[..k]
                                .iter()
                                .enumerate()
                                .filter(move |&(l, _)| l != skip)
                                .map(move |(_, &c)| (r, c))
                        })
                        .map(|(r, c)| self.unitary[(r, c)])
                        .collect();
                    glynn(&sub, k - 1)
                })
                .collect();
            let mut acc = 0.0;
            for (i, slot) in cumulative.iter_mut().enumerate() {
                let amp: C64 = cols[..k]
                    .iter()
                    .zip(&minors)
                    .map(|(&c, &minor)| self.unitary[(i, c)] * minor)
                    .sum();
                acc += amp.norm_sqr();
                *slot = acc;
            }
            rows.push(sample_cumulative(&cumulative, rng));
        }
        let mut out = vec![0; m];
        for r in rows {
            out[r] += 1;
        }
        out
    }

    /// Output modes repeated by occupation, after validating `output`.
    fn output_rows(&self, output: &[usize]) -> Result<Vec<usize>> {
        if output.len() != self.n_modes() {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_modes(),
                actual: output.len(),
            });
        }
        if checked_total(output) != Some(self.n_photons()) {
            return Err(QsoError::invalid(
                "output",
                format!(
                    "pattern must hold as many photons as the input ({})",
                    self.n_photons()
                ),
            ));
        }
        Ok(output
            .iter()
            .enumerate()
            .flat_map(|(i, &t)| std::iter::repeat_n(i, t))
            .collect())
    }

    /// Row-major `U_{T,S}` with `f` applied to every entry.
    fn submatrix<T>(&self, rows: &[usize], f: impl Fn(C64) -> T) -> Vec<T> {
        rows.iter()
            .flat_map(|&r| self.photon_modes.iter().map(move |&c| (r, c)))
            .map(|(r, c)| f(self.unitary[(r, c)]))
            .collect()
    }
}

fn factorial_product(occupation: &[usize]) -> f64 {
    occupation
        .iter()
        .map(|&k| (1..=k).map(|x| x as f64).product::<f64>())
        .product()
}

/// Total photon number of an occupation, or `None` on overflow.
fn checked_total(occupation: &[usize]) -> Option<usize> {
    occupation.iter().try_fold(0usize, |a, &t| a.checked_add(t))
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Number of ways to place `n` photons in `m` modes, `C(m + n − 1, n)`;
/// `u128::MAX` if the exact value does not fit into a `u128`.
pub fn count_patterns(n_photons: usize, n_modes: usize) -> u128 {
    if n_modes == 0 {
        return u128::from(n_photons == 0);
    }
    // C(top, k) with the smaller of the two symmetric choices of k, built
    // up multiplicatively: c_k = c_{k−1}·(top − k + 1)/k is a binomial
    // coefficient at every step. Dividing by the gcd first keeps the
    // product exact until the result itself overflows.
    let top = n_modes as u128 - 1 + n_photons as u128;
    let k_max = (n_photons as u128).min(n_modes as u128 - 1);
    let mut c: u128 = 1;
    for k in 1..=k_max {
        let g = gcd(c, k);
        let factor = (top - k + 1) / (k / g);
        match (c / g).checked_mul(factor) {
            Some(next) => c = next,
            None => return u128::MAX,
        }
    }
    c
}

/// All occupation patterns of `n_photons` photons in `n_modes` modes, in
/// lexicographically descending order (`[n, 0, …]` first).
pub fn occupation_patterns(n_photons: usize, n_modes: usize) -> Result<Vec<Vec<usize>>> {
    check_modes(n_modes)?;
    let count = count_patterns(n_photons, n_modes);
    let entries = count.saturating_mul(n_modes as u128);
    if count > MAX_PATTERNS as u128 || entries > MAX_OUTPUT_ENTRIES as u128 {
        return Err(QsoError::invalid(
            "patterns",
            format!(
                "{n_photons} photons in {n_modes} modes give {count} patterns, more than the \
                 supported {MAX_PATTERNS} (or {MAX_OUTPUT_ENTRIES} occupation numbers)"
            ),
        ));
    }
    let mut out = Vec::with_capacity(count as usize);
    let mut current = vec![0; n_modes];
    fill(&mut current, 0, n_photons, &mut out);
    Ok(out)
}

fn fill(current: &mut [usize], mode: usize, remaining: usize, out: &mut Vec<Vec<usize>>) {
    if mode + 1 == current.len() {
        current[mode] = remaining;
        out.push(current.to_vec());
        return;
    }
    for k in (0..=remaining).rev() {
        current[mode] = k;
        fill(current, mode + 1, remaining - k, out);
    }
    current[mode] = 0;
}

#[cfg(test)]
mod tests {
    use super::super::{beam_splitter, haar_unitary};
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn hong_ou_mandel_dip() {
        // Two photons on a 50:50 beam splitter never leave through
        // different ports (Hong, Ou & Mandel 1987); distinguishable photons
        // do so half of the time.
        let bs = beam_splitter(std::f64::consts::FRAC_PI_4, 0.0);
        let s = BosonSampler::new(bs, &[1, 1]).unwrap();
        assert!(s.probability(&[1, 1]).unwrap() < 1e-30);
        assert!((s.probability(&[2, 0]).unwrap() - 0.5).abs() < 1e-15);
        assert!((s.probability_distinguishable(&[1, 1]).unwrap() - 0.5).abs() < 1e-15);
        let samples = s.sample(1000, &mut Rng::seed_from_u64(1)).unwrap();
        assert!(samples.iter().all(|t| t[0] != 1));
    }

    #[test]
    fn distributions_are_normalised() {
        let mut rng = Rng::seed_from_u64(2);
        let u = haar_unitary(6, &mut rng).unwrap();
        for input in [[1, 1, 1, 0, 0, 0], [2, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 4]] {
            let s = BosonSampler::new(u.clone(), &input).unwrap();
            let q = s.distribution().unwrap();
            let c = s.distribution_distinguishable().unwrap();
            assert_eq!(q.patterns.len() as u128, count_patterns(s.n_photons(), 6));
            let total_q: f64 = q.probabilities.iter().sum();
            let total_c: f64 = c.probabilities.iter().sum();
            assert!((total_q - 1.0).abs() < 1e-12, "{input:?}: {total_q}");
            assert!((total_c - 1.0).abs() < 1e-12, "{input:?}: {total_c}");
        }
    }

    #[test]
    fn full_bunching_is_enhanced_by_n_factorial() {
        // All n photons in one output mode: perm of a rank-one matrix, so
        // P_boson / P_distinguishable = n! for every interferometer.
        let mut rng = Rng::seed_from_u64(4);
        let u = haar_unitary(7, &mut rng).unwrap();
        let s = BosonSampler::new(u, &[1, 1, 1, 1, 0, 0, 0]).unwrap();
        let mut bunched = vec![0; 7];
        bunched[3] = 4;
        let ratio =
            s.probability(&bunched).unwrap() / s.probability_distinguishable(&bunched).unwrap();
        assert!((ratio - 24.0).abs() < 1e-9, "{ratio}");
    }

    #[test]
    fn single_photon_follows_the_unitary_column() {
        let mut rng = Rng::seed_from_u64(6);
        let u = haar_unitary(5, &mut rng).unwrap();
        let s = BosonSampler::new(u.clone(), &[0, 0, 1, 0, 0]).unwrap();
        for i in 0..5 {
            let mut t = vec![0; 5];
            t[i] = 1;
            assert!((s.probability(&t).unwrap() - u[(i, 2)].norm_sqr()).abs() < 1e-14);
        }
    }

    #[test]
    fn exact_sampler_reproduces_the_distribution() {
        // χ² goodness of fit of 40 000 exact samples against the computed
        // distribution (3 photons in 5 modes: 35 patterns).
        let mut rng = Rng::seed_from_u64(8);
        let u = haar_unitary(5, &mut rng).unwrap();
        for (sampler_is_quantum, seed) in [(true, 10), (false, 11)] {
            let s = BosonSampler::new(u.clone(), &[1, 0, 1, 1, 0]).unwrap();
            let shots = 40_000;
            let mut rng = Rng::seed_from_u64(seed);
            let (dist, samples) = if sampler_is_quantum {
                (
                    s.distribution().unwrap(),
                    s.sample(shots, &mut rng).unwrap(),
                )
            } else {
                (
                    s.distribution_distinguishable().unwrap(),
                    s.sample_distinguishable(shots, &mut rng).unwrap(),
                )
            };
            let mut counts: HashMap<Vec<usize>, usize> = HashMap::new();
            for t in samples {
                *counts.entry(t).or_default() += 1;
            }
            let mut chi2 = 0.0;
            let mut dof = 0;
            for (t, &p) in dist.patterns.iter().zip(&dist.probabilities) {
                let expected = p * shots as f64;
                if expected > 5.0 {
                    let observed = *counts.get(t).unwrap_or(&0) as f64;
                    chi2 += (observed - expected).powi(2) / expected;
                    dof += 1;
                }
            }
            // 99.9 % quantile of χ² with ≤ 34 degrees of freedom is < 66.
            assert!(dof > 20 && chi2 < 66.0, "χ² = {chi2} with {dof} cells");
        }
    }

    #[test]
    fn sampling_is_reproducible_and_validated() {
        let u = haar_unitary(4, &mut Rng::seed_from_u64(12)).unwrap();
        let s = BosonSampler::new(u.clone(), &[1, 1, 0, 0]).unwrap();
        let a = s.sample(300, &mut Rng::seed_from_u64(13)).unwrap();
        let b = s.sample(300, &mut Rng::seed_from_u64(13)).unwrap();
        assert_eq!(a, b);
        assert!(a.iter().all(|t| t.iter().sum::<usize>() == 2));
        // Collision input is rejected by the sampler, not by probabilities.
        let c = BosonSampler::new(u, &[2, 0, 0, 0]).unwrap();
        assert!(c.sample(1, &mut Rng::seed_from_u64(1)).is_err());
        assert!(c.probability(&[0, 2, 0, 0]).is_ok());
    }

    #[test]
    fn rejects_invalid_input() {
        let u = haar_unitary(3, &mut Rng::seed_from_u64(1)).unwrap();
        assert!(BosonSampler::new(u.clone(), &[0, 0, 0]).is_err());
        assert!(BosonSampler::new(u.clone(), &[1, 0]).is_err());
        let mut not_unitary = u.clone();
        not_unitary[(0, 0)] *= 2.0;
        assert!(BosonSampler::new(not_unitary, &[1, 0, 0]).is_err());
        let s = BosonSampler::new(u, &[1, 1, 0]).unwrap();
        assert!(s.probability(&[1, 0, 0]).is_err());
        assert!(s.probability(&[1, 1]).is_err());
        assert!(occupation_patterns(30, 40).is_err());
    }

    #[test]
    fn rejects_hostile_sizes() {
        let u = haar_unitary(3, &mut Rng::seed_from_u64(1)).unwrap();
        // Photon totals that overflow are rejected, not wrapped around.
        assert!(BosonSampler::new(u.clone(), &[usize::MAX, 2, 0]).is_err());
        assert!(BosonSampler::new(u.clone(), &[MAX_PHOTONS + 1, 0, 0]).is_err());
        let s = BosonSampler::new(u, &[1, 1, 0]).unwrap();
        assert!(s.probability(&[usize::MAX, 3, 0]).is_err());
        assert!(s.probability_distinguishable(&[usize::MAX, 3, 0]).is_err());
        // Output sizes are bounded before anything is allocated.
        let shots = MAX_OUTPUT_ENTRIES / 3 + 1;
        assert!(s.sample(shots, &mut Rng::seed_from_u64(1)).is_err());
        assert!(
            s.sample_distinguishable(usize::MAX, &mut Rng::seed_from_u64(1))
                .is_err()
        );
        // C(301, 2)·300 = 13.5 million entries; C(1025, 2)·1024 = 537 million.
        assert!(occupation_patterns(2, 300).is_ok());
        assert!(occupation_patterns(2, 1024).is_err());
    }

    #[test]
    fn pattern_counts_are_exact_or_saturate() {
        for m in 0..12 {
            for n in 0..12 {
                let expected = if m == 0 {
                    usize::from(n == 0)
                } else {
                    occupation_patterns(n, m).unwrap().len()
                };
                assert_eq!(count_patterns(n, m), expected as u128, "n={n} m={m}");
            }
        }
        assert_eq!(count_patterns(1, 1024), 1024);
        assert_eq!(count_patterns(usize::MAX, 1), 1);
        assert_eq!(count_patterns(usize::MAX, 2), 1 << 64);
        // C(119, 60) and C(131, 65) fit; C(132, 66) does not.
        assert_eq!(
            count_patterns(60, 60),
            48_307_454_420_181_661_301_946_569_760_686_328
        );
        assert_eq!(
            count_patterns(65, 67),
            188_694_833_082_770_476_622_296_176_145_946_360_850
        );
        assert_eq!(count_patterns(66, 67), u128::MAX);
        assert_eq!(count_patterns(30, 1024), u128::MAX);
        assert_eq!(count_patterns(usize::MAX, usize::MAX), u128::MAX);
    }

    #[test]
    fn pattern_enumeration() {
        let p = occupation_patterns(2, 3).unwrap();
        assert_eq!(
            p,
            vec![
                vec![2, 0, 0],
                vec![1, 1, 0],
                vec![1, 0, 1],
                vec![0, 2, 0],
                vec![0, 1, 1],
                vec![0, 0, 2],
            ]
        );
        assert_eq!(count_patterns(4, 16), 3876);
        assert_eq!(count_patterns(0, 5), 1);
    }
}
