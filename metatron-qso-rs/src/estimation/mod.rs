//! Statistical estimates from samples: values with standard errors and
//! confidence intervals.
//!
//! Every stochastic estimate in QSO (trajectory averages, shot-based
//! expectation values, error-mitigated values, classical shadows) is an
//! [`Estimate`]: the value, its standard error and the number of samples
//! behind it.
//!
//! ```
//! use metatron_qso::estimation::Estimate;
//!
//! let e = Estimate::from_samples(&[0.9, 1.1, 1.0, 0.95, 1.05]);
//! let (lo, hi) = e.confidence_interval(0.95);
//! assert!(lo < 1.0 && 1.0 < hi);
//! ```

use serde::{Deserialize, Serialize};

/// A value with its standard error.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Estimate {
    pub value: f64,
    /// Standard error of `value` (0 for exact values).
    pub std_error: f64,
    /// Number of samples behind the estimate (0 for exact values).
    pub samples: usize,
}

impl Estimate {
    /// An exact value (zero standard error).
    pub fn exact(value: f64) -> Self {
        Self {
            value,
            std_error: 0.0,
            samples: 0,
        }
    }

    /// Mean of independent samples with the standard error `s/√n` (sample
    /// standard deviation `s`; zero for fewer than two samples).
    pub fn from_samples(samples: &[f64]) -> Self {
        let n = samples.len();
        if n == 0 {
            return Self {
                value: f64::NAN,
                std_error: f64::NAN,
                samples: 0,
            };
        }
        let mean = samples.iter().sum::<f64>() / n as f64;
        let std_error = if n > 1 {
            let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            (var / n as f64).sqrt()
        } else {
            0.0
        };
        Self {
            value: mean,
            std_error,
            samples: n,
        }
    }

    /// Two-sided confidence interval at `level` (e.g. 0.95) from the normal
    /// approximation: `value ± z·std_error`.
    ///
    /// # Panics
    /// If `level` is not in `(0, 1)`.
    pub fn confidence_interval(&self, level: f64) -> (f64, f64) {
        assert!(
            level > 0.0 && level < 1.0,
            "confidence level {level} outside (0, 1)"
        );
        let z = normal_quantile(0.5 + level / 2.0);
        (
            self.value - z * self.std_error,
            self.value + z * self.std_error,
        )
    }

    /// Sum of independent estimates (standard errors add in quadrature).
    pub fn sum(estimates: &[Estimate]) -> Self {
        Self {
            value: estimates.iter().map(|e| e.value).sum(),
            std_error: estimates
                .iter()
                .map(|e| e.std_error * e.std_error)
                .sum::<f64>()
                .sqrt(),
            samples: estimates.iter().map(|e| e.samples).sum(),
        }
    }
}

/// Quantile function of the standard normal distribution (Acklam's
/// rational approximation, relative error below 1.2e-9).
///
/// # Panics
/// If `p` is not in `(0, 1)`.
pub fn normal_quantile(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "probability {p} outside (0, 1)");
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    const LOW: f64 = 0.02425;
    if p < LOW {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - LOW {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_quantiles() {
        assert!((normal_quantile(0.975) - 1.959_963_984_540_054).abs() < 1e-8);
        assert!((normal_quantile(0.5)).abs() < 1e-12);
        assert!((normal_quantile(0.001) + 3.090_232_306_167_813_5).abs() < 1e-8);
        assert!((normal_quantile(0.999_99) - 4.264_890_793_922_602).abs() < 1e-7);
    }

    #[test]
    fn sample_statistics() {
        let e = Estimate::from_samples(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(e.value, 2.5);
        // s² = 5/3, s/√4
        assert!((e.std_error - (5.0f64 / 3.0).sqrt() / 2.0).abs() < 1e-15);
        let (lo, hi) = e.confidence_interval(0.95);
        assert!((hi - lo - 2.0 * 1.959_963_984_540_054 * e.std_error).abs() < 1e-8);
        let total = Estimate::sum(&[e, Estimate::exact(1.0)]);
        assert_eq!(total.value, 3.5);
        assert_eq!(total.std_error, e.std_error);
    }
}
