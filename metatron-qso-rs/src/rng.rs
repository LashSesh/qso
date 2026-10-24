//! Deterministic, portable pseudo-random numbers.
//!
//! Every stochastic routine in the crate (measurement sampling, random
//! initial parameters, SPSA perturbations, random graphs, Haar-random states)
//! takes an explicit [`Rng`]. The generator is xoshiro256++ seeded through
//! SplitMix64, implemented here rather than pulled from a dependency so that
//! a given seed produces the *same* numbers on every platform and across
//! dependency upgrades. That is what makes published results reproducible.
//!
//! This generator is not cryptographically secure.

/// xoshiro256++ generator (Blackman & Vigna, 2019).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Create a generator from a 64-bit seed.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Self { s }
    }

    /// Derive an independent generator for sub-task `stream` (e.g. one per
    /// random restart) without consuming numbers from `self`.
    pub fn fork(&self, stream: u64) -> Self {
        let mut sm =
            self.s[0] ^ self.s[3].rotate_left(17) ^ stream.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        Self {
            s: [
                splitmix64(&mut sm),
                splitmix64(&mut sm),
                splitmix64(&mut sm),
                splitmix64(&mut sm),
            ],
        }
    }

    /// Next raw 64-bit output.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform `f64` in `[0, 1)` with 53 bits of precision.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform `f64` in `[low, high)`.
    #[inline]
    pub fn uniform_range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.uniform()
    }

    /// Uniform integer in `0..n` (Lemire's nearly-divisionless method).
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "Rng::below requires n > 0");
        let mut m = (self.next_u64() as u128) * (n as u128);
        let mut low = m as u64;
        if low < n {
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                m = (self.next_u64() as u128) * (n as u128);
                low = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Standard normal deviate (Box–Muller).
    pub fn normal(&mut self) -> f64 {
        // 1 - uniform() lies in (0, 1], so the logarithm is finite.
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Rademacher variable: ±1 with equal probability.
    #[inline]
    pub fn sign(&mut self) -> f64 {
        if self.next_u64() >> 63 == 0 {
            1.0
        } else {
            -1.0
        }
    }

    /// In-place Fisher–Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

/// Sample an index from an unnormalised cumulative distribution.
///
/// `cumulative` must be non-decreasing with a positive last element.
pub(crate) fn sample_cumulative(cumulative: &[f64], rng: &mut Rng) -> usize {
    let total = *cumulative.last().expect("non-empty distribution");
    let r = rng.uniform() * total;
    // First index whose cumulative weight exceeds r.
    let idx = cumulative.partition_point(|&c| c <= r);
    idx.min(cumulative.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_output_is_stable() {
        // Pin the stream so accidental changes to the generator are caught:
        // published results depend on it. Values cross-checked against an
        // independent Python implementation of SplitMix64 + xoshiro256++.
        let mut rng = Rng::seed_from_u64(42);
        let first: Vec<u64> = (0..3).map(|_| rng.next_u64()).collect();
        assert_eq!(
            first,
            [0xd0764d4f4476689f, 0x519e4174576f3791, 0xfbe07cfb0c24ed8c]
        );
    }

    #[test]
    fn uniform_moments() {
        let mut rng = Rng::seed_from_u64(7);
        let n = 200_000;
        let xs: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 5e-3);
        assert!((var - 1.0 / 12.0).abs() < 2e-3);
        assert!(xs.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn normal_moments() {
        let mut rng = Rng::seed_from_u64(11);
        let n = 200_000;
        let xs: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 1e-2);
        assert!((var - 1.0).abs() < 2e-2);
    }

    #[test]
    fn below_is_unbiased_and_in_range() {
        let mut rng = Rng::seed_from_u64(3);
        let mut counts = [0usize; 6];
        for _ in 0..60_000 {
            counts[rng.below(6) as usize] += 1;
        }
        for c in counts {
            assert!((c as f64 - 10_000.0).abs() < 400.0, "{counts:?}");
        }
    }

    #[test]
    fn forks_are_distinct_and_deterministic() {
        let base = Rng::seed_from_u64(5);
        let mut a = base.fork(1);
        let mut b = base.fork(2);
        let mut a2 = base.fork(1);
        assert_ne!(a.next_u64(), b.next_u64());
        let _ = a2.next_u64();
        assert_eq!(a.next_u64(), a2.next_u64());
    }

    #[test]
    fn cumulative_sampling_respects_weights() {
        let mut rng = Rng::seed_from_u64(9);
        let cumulative = [0.1, 0.1, 0.4, 1.0];
        let mut counts = [0usize; 4];
        for _ in 0..100_000 {
            counts[sample_cumulative(&cumulative, &mut rng)] += 1;
        }
        assert_eq!(counts[1], 0);
        assert!((counts[0] as f64 / 1e5 - 0.1).abs() < 0.01);
        assert!((counts[2] as f64 / 1e5 - 0.3).abs() < 0.01);
        assert!((counts[3] as f64 / 1e5 - 0.6).abs() < 0.01);
    }
}
