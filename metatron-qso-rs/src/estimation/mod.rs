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

pub mod shadows;

pub use shadows::ClassicalShadow;

use serde::{Deserialize, Serialize};

use crate::backend::{Backend, measurement_groups};
use crate::error::{QsoError, Result};
use crate::mitigation::ReadoutCalibration;
use crate::mitigation::readout::ParityWeights;
use crate::quantum::{Circuit, Counts, Pauli, PauliSum};
use crate::rng::Rng;

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
        let z = -normal_quantile((1.0 - level) / 2.0);
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

/// How [`estimate`] spreads its shots over the measurement groups.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ShotAllocation {
    /// The same number of shots for every group.
    Uniform,
    /// Shots proportional to `Σ|c|` of a group's terms, a bound on the
    /// standard deviation of one shot.
    Weighted,
    /// A fraction `pilot` of the shots spread uniformly to measure the
    /// standard deviations; the rest proportional to them (Neyman
    /// allocation, which minimises the variance of the total). The pilot
    /// shots only choose the allocation and do not enter the estimate.
    Adaptive { pilot: f64 },
}

/// Options of [`estimate`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EstimatorOptions {
    /// Total number of shots over all measurement groups.
    pub shots: usize,
    pub allocation: ShotAllocation,
    /// Readout calibration for mitigated parities (`None`: raw counts).
    pub readout: Option<ReadoutCalibration>,
    pub seed: u64,
}

impl EstimatorOptions {
    /// `shots` shots in total, weighted allocation, no mitigation, seed 0.
    pub fn new(shots: usize) -> Self {
        Self {
            shots,
            allocation: ShotAllocation::Weighted,
            readout: None,
            seed: 0,
        }
    }

    pub fn with_allocation(mut self, allocation: ShotAllocation) -> Self {
        self.allocation = allocation;
        self
    }

    /// Mitigate readout errors with `calibration`.
    pub fn with_readout_mitigation(mut self, calibration: ReadoutCalibration) -> Self {
        self.readout = Some(calibration);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// The estimate of one measurement group.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupEstimate {
    /// Measurement basis: Pauli per measured qubit.
    pub basis: Vec<(usize, Pauli)>,
    /// Number of terms measured in this basis.
    pub terms: usize,
    pub shots: usize,
    /// Sum of the group's terms (coefficients included).
    pub estimate: Estimate,
}

/// An observable's estimate and its parts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservableEstimate {
    /// Constant term plus all groups; standard errors add in quadrature.
    pub estimate: Estimate,
    pub groups: Vec<GroupEstimate>,
}

/// Split `total` into parts proportional to `weights` (largest
/// remainders), giving every part at least `min`.
fn allocate(total: usize, weights: &[f64], min: usize) -> Vec<usize> {
    let g = weights.len();
    let base = min * g;
    let rest = total.saturating_sub(base);
    let sum: f64 = weights.iter().sum();
    let shares: Vec<f64> = if sum > 0.0 && sum.is_finite() {
        weights.iter().map(|w| rest as f64 * w / sum).collect()
    } else {
        vec![rest as f64 / g as f64; g]
    };
    let mut parts: Vec<usize> = shares.iter().map(|s| s.floor() as usize).collect();
    let mut left = rest - parts.iter().sum::<usize>();
    let mut order: Vec<usize> = (0..g).collect();
    order.sort_by(|&a, &b| {
        (shares[b] - shares[b].floor()).total_cmp(&(shares[a] - shares[a].floor()))
    });
    for &i in order.iter().cycle() {
        if left == 0 {
            break;
        }
        parts[i] += 1;
        left -= 1;
    }
    parts.iter().map(|p| p + min).collect()
}

/// Per-shot value of a group: `Σ_t w_t(b)`.
struct GroupWeights {
    terms: Vec<(f64, usize, Option<ParityWeights>)>,
}

impl GroupWeights {
    fn value(&self, b: usize) -> f64 {
        self.terms
            .iter()
            .map(|(c, mask, w)| match w {
                Some(w) => c * w.weight(b),
                None if (b & mask).count_ones().is_multiple_of(2) => *c,
                None => -c,
            })
            .sum()
    }

    /// Largest magnitude of a per-shot value.
    fn bound(&self) -> f64 {
        self.terms
            .iter()
            .map(|(c, _, w)| {
                c.abs()
                    * match w {
                        None => 1.0,
                        Some(ParityWeights::Factors(f)) => {
                            f.iter().map(|(_, [a, b])| a.abs().max(b.abs())).product()
                        }
                        Some(ParityWeights::Dense(d)) => {
                            d.iter().fold(0.0, |m: f64, x| m.max(x.abs()))
                        }
                    }
            })
            .sum()
    }

    /// Mean and standard error over the shots in `counts`. When all shots
    /// agree (or there is one shot) the sample variance says nothing about
    /// the spread; the variance is then bounded below by that of a value
    /// spanning the whole range with probability `1/(n + 2)` (add-one
    /// smoothing), so no estimate claims to be exact.
    fn estimate(&self, counts: &Counts) -> Estimate {
        let n: usize = counts.values().sum();
        let values: Vec<(f64, f64)> = counts
            .iter()
            .map(|(&b, &k)| (self.value(b), k as f64))
            .collect();
        let mean = values.iter().map(|(v, k)| v * k).sum::<f64>() / n as f64;
        let mut var = if n > 1 {
            values
                .iter()
                .map(|(v, k)| k * (v - mean).powi(2))
                .sum::<f64>()
                / (n - 1) as f64
        } else {
            0.0
        };
        if var <= 0.0 {
            let p = 1.0 / (n as f64 + 2.0);
            var = (2.0 * self.bound()).powi(2) * p * (1.0 - p);
        }
        Estimate {
            value: mean,
            std_error: (var / n as f64).sqrt(),
            samples: n,
        }
    }
}

/// Estimate `⟨H⟩` for `circuit(theta)` on `backend`: terms are grouped
/// into qubit-wise commuting sets, each set is measured in its rotated
/// basis with its share of the shots, and every estimate carries its
/// standard error (from the spread of the per-shot values). With a readout
/// calibration the parities are mitigated, and the standard error includes
/// the cost of mitigation.
pub fn estimate(
    backend: &dyn Backend,
    circuit: &Circuit,
    theta: &[f64],
    observable: &PauliSum,
    options: &EstimatorOptions,
) -> Result<ObservableEstimate> {
    let n = circuit.n_qubits();
    if observable.min_qubits() > n {
        return Err(QsoError::QubitOutOfRange {
            qubit: observable.min_qubits() - 1,
            n_qubits: n,
        });
    }
    if let Some(cal) = &options.readout
        && cal.n_qubits() != n
    {
        return Err(QsoError::DimensionMismatch {
            expected: n,
            actual: cal.n_qubits(),
        });
    }
    let groups = measurement_groups(observable);
    let g = groups.len();
    let constant = Estimate::exact(observable.constant());
    if g == 0 {
        return Ok(ObservableEstimate {
            estimate: constant,
            groups: Vec::new(),
        });
    }
    if options.shots < 2 * g {
        return Err(QsoError::invalid(
            "shots",
            format!("need at least two shots per measurement group ({g} groups)"),
        ));
    }
    let mut rotated = Vec::with_capacity(g);
    let mut weights = Vec::with_capacity(g);
    for (basis, members) in &groups {
        let mut c = circuit.clone();
        for (&q, &p) in basis {
            match p {
                Pauli::X => {
                    c.h(q);
                }
                Pauli::Y => {
                    c.sdg(q).h(q);
                }
                _ => {}
            }
        }
        rotated.push(c);
        weights.push(GroupWeights {
            terms: members
                .iter()
                .map(|&(coef, s)| {
                    let mask = (s.x_mask() | s.z_mask()) as usize;
                    let w = options.readout.as_ref().map(|cal| cal.parity_weights(mask));
                    (coef, mask, w)
                })
                .collect(),
        });
    }
    let base = Rng::seed_from_u64(options.seed);
    let mut counts: Vec<Counts> = vec![Counts::new(); g];
    let run = |k: usize, shots: &[usize], counts: &mut Vec<Counts>| -> Result<()> {
        for (j, &s) in shots.iter().enumerate() {
            if s == 0 {
                continue;
            }
            let seed = base.fork((k * g + j) as u64).next_u64();
            for (b, m) in backend.run(&rotated[j], theta, s, seed)? {
                *counts[j].entry(b).or_insert(0) += m;
            }
        }
        Ok(())
    };
    let bound = |members: &Vec<(f64, crate::quantum::PauliString)>| {
        members.iter().map(|(c, _)| c.abs()).sum::<f64>()
    };
    match options.allocation {
        ShotAllocation::Uniform => run(0, &allocate(options.shots, &vec![1.0; g], 2), &mut counts)?,
        ShotAllocation::Weighted => {
            let w: Vec<f64> = groups.iter().map(|(_, m)| bound(m)).collect();
            run(0, &allocate(options.shots, &w, 2), &mut counts)?;
        }
        ShotAllocation::Adaptive { pilot } => {
            if !(pilot > 0.0 && pilot < 1.0) {
                return Err(QsoError::invalid(
                    "pilot",
                    "pilot fraction must lie in (0, 1)",
                ));
            }
            // The pilot only chooses the allocation; the estimate uses the
            // main shots alone. Pooling both would make the number of main
            // shots depend on the pilot outcome and bias the result.
            let pilot_shots =
                ((pilot * options.shots as f64).round() as usize).clamp(2 * g, options.shots);
            if options.shots - pilot_shots < 2 * g {
                // Too few shots for two stages: weighted allocation.
                let w: Vec<f64> = groups.iter().map(|(_, m)| bound(m)).collect();
                run(0, &allocate(options.shots, &w, 2), &mut counts)?;
            } else {
                let mut pilot_counts: Vec<Counts> = vec![Counts::new(); g];
                run(
                    0,
                    &allocate(pilot_shots, &vec![1.0; g], 2),
                    &mut pilot_counts,
                )?;
                // Standard deviations, floored at the bound over √n so that
                // a pilot whose shots all agree still gets main shots.
                let sigma: Vec<f64> = (0..g)
                    .map(|j| {
                        let e = weights[j].estimate(&pilot_counts[j]);
                        let n = e.samples as f64;
                        (e.std_error * n.sqrt()).max(weights[j].bound() / n.sqrt())
                    })
                    .collect();
                run(
                    1,
                    &allocate(options.shots - pilot_shots, &sigma, 2),
                    &mut counts,
                )?;
            }
        }
    }
    let parts: Vec<GroupEstimate> = groups
        .iter()
        .zip(&weights)
        .zip(&counts)
        .map(|(((basis, members), w), c)| GroupEstimate {
            basis: basis.iter().map(|(&q, &p)| (q, p)).collect(),
            terms: members.len(),
            shots: c.values().sum(),
            estimate: w.estimate(c),
        })
        .collect();
    let mut all: Vec<Estimate> = parts.iter().map(|p| p.estimate).collect();
    all.push(constant);
    Ok(ObservableEstimate {
        estimate: Estimate::sum(&all),
        groups: parts,
    })
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

    use crate::algorithms::models;
    use crate::backend::{DeviceBackend, StatevectorBackend};
    use crate::noise::{DeviceModel, QubitProperties};
    use crate::quantum::StateVector;

    #[test]
    fn allocation_rounds_to_the_total() {
        assert_eq!(allocate(10, &[1.0, 1.0, 1.0], 1), vec![4, 3, 3]);
        assert_eq!(allocate(10, &[0.0, 1.0], 1), vec![1, 9]);
        assert_eq!(allocate(7, &[0.0, 0.0], 0).iter().sum::<usize>(), 7);
        let parts = allocate(1000, &[5.0, 0.2, 1.3], 2);
        assert_eq!(parts.iter().sum::<usize>(), 1000);
        assert!(parts[0] > parts[2] && parts[2] > parts[1]);
    }

    #[test]
    fn confidence_intervals_cover_the_exact_value() {
        let h2 = models::h2_sto3g();
        let mut c = Circuit::new(2);
        c.ry(0, 0.4).ry(1, -1.1).cx(0, 1).ry(1, 0.3);
        let exact = h2
            .hamiltonian
            .expectation(&StateVector::from_circuit(&c, &[]).unwrap())
            .unwrap();
        let repeats = 300;
        let (mut covered, mut values, mut errors) = (0, Vec::new(), Vec::new());
        for seed in 0..repeats {
            let opts = EstimatorOptions::new(2000)
                .with_allocation(ShotAllocation::Adaptive { pilot: 0.2 })
                .with_seed(seed);
            let e = estimate(&StatevectorBackend, &c, &[], &h2.hamiltonian, &opts)
                .unwrap()
                .estimate;
            let (lo, hi) = e.confidence_interval(0.95);
            covered += usize::from(lo <= exact && exact <= hi);
            values.push(e.value);
            errors.push(e.std_error);
            assert!(e.samples > 1500 && e.samples < 2000);
        }
        let coverage = covered as f64 / repeats as f64;
        assert!((0.91..=0.99).contains(&coverage), "coverage {coverage}");
        // Reported standard errors match the spread of the estimates.
        let spread = Estimate::from_samples(&values).std_error * (repeats as f64).sqrt();
        let reported = errors.iter().sum::<f64>() / repeats as f64;
        assert!(
            (spread / reported - 1.0).abs() < 0.15,
            "{spread} vs {reported}"
        );
    }

    #[test]
    fn neyman_allocation_reduces_the_error() {
        // Four groups; on |++⟩ only 5 Z0 Z1 fluctuates (σ = 5), the others
        // hardly (σ ≤ 0.1). Uniform allocation wastes three quarters of the
        // shots: its error is √(4·25)/5.2 ≈ 1.9 times the Neyman error.
        let h: PauliSum = "5 Z0 Z1 + 0.1 X0 X1 + 0.1 Y0 Y1 + 0.1 X0 Y1"
            .parse()
            .unwrap();
        assert_eq!(measurement_groups(&h).len(), 4);
        let mut c = Circuit::new(2);
        c.h(0).h(1);
        let mean_error = |allocation| {
            (0..20)
                .map(|seed| {
                    let opts = EstimatorOptions::new(4000)
                        .with_allocation(allocation)
                        .with_seed(seed);
                    estimate(&StatevectorBackend, &c, &[], &h, &opts)
                        .unwrap()
                        .estimate
                        .std_error
                })
                .sum::<f64>()
                / 20.0
        };
        let uniform = mean_error(ShotAllocation::Uniform);
        let weighted = mean_error(ShotAllocation::Weighted);
        let adaptive = mean_error(ShotAllocation::Adaptive { pilot: 0.1 });
        assert!(uniform > 1.6 * adaptive, "{uniform} vs {adaptive}");
        assert!(uniform > 1.6 * weighted, "{uniform} vs {weighted}");
    }

    #[test]
    fn readout_mitigated_estimates_are_unbiased() {
        let errors = [(0.04, 0.09), (0.07, 0.03)];
        let mut device = DeviceModel::new("readout");
        for (q, &(a, b)) in errors.iter().enumerate() {
            device = device.with_qubit(q, QubitProperties::new().with_readout(a, b));
        }
        let backend = DeviceBackend::density_matrix(device).unwrap();
        let mut c = Circuit::new(2);
        c.ry(0, 0.5).cx(0, 1).rx(1, 0.3);
        let h: PauliSum = "Z0 Z1 + 0.5 Z0 - 0.7 X1 + 0.2".parse().unwrap();
        let exact = h
            .expectation(&StateVector::from_circuit(&c, &[]).unwrap())
            .unwrap();
        let raw = estimate(
            &backend,
            &c,
            &[],
            &h,
            &EstimatorOptions::new(400_000).with_seed(1),
        )
        .unwrap()
        .estimate;
        assert!((raw.value - exact).abs() > 10.0 * raw.std_error);
        let cal = ReadoutCalibration::tensored(&backend, 2, 200_000, 9).unwrap();
        let opts = EstimatorOptions::new(400_000)
            .with_seed(1)
            .with_readout_mitigation(cal);
        let mitigated = estimate(&backend, &c, &[], &h, &opts).unwrap().estimate;
        // The calibration's own error adds a little bias.
        assert!(
            (mitigated.value - exact).abs() < 4.0 * mitigated.std_error + 0.005,
            "{} vs {exact}",
            mitigated.value
        );
        assert!(mitigated.std_error > raw.std_error);
    }

    #[test]
    fn adaptive_allocation_is_unbiased_for_nearly_deterministic_groups() {
        // P(Z0 = −1) ≈ 0.01: most pilots see no spread in the Z group.
        let h: PauliSum = "Z0 + X0".parse().unwrap();
        let mut c = Circuit::new(1);
        c.ry(0, 0.2);
        let exact = h
            .expectation(&StateVector::from_circuit(&c, &[]).unwrap())
            .unwrap();
        let repeats = 2000;
        let mut values = Vec::with_capacity(repeats);
        for seed in 0..repeats as u64 {
            let opts = EstimatorOptions::new(1000)
                .with_allocation(ShotAllocation::Adaptive { pilot: 0.1 })
                .with_seed(seed);
            let r = estimate(&StatevectorBackend, &c, &[], &h, &opts).unwrap();
            assert!(r.groups.iter().all(|g| g.estimate.std_error > 0.0));
            values.push(r.estimate.value);
        }
        let mean = Estimate::from_samples(&values);
        assert!(
            (mean.value - exact).abs() < 4.0 * mean.std_error,
            "{mean:?} vs {exact}"
        );
    }

    #[test]
    fn extreme_levels_and_tiny_samples() {
        let e = Estimate {
            value: 1.0,
            std_error: 0.1,
            samples: 10,
        };
        let (lo, hi) = e.confidence_interval(1.0 - 1e-17_f64.max(f64::EPSILON / 2.0));
        assert!(lo < 1.0 && hi > 1.0 && lo.is_finite());
        // Two shots that agree still carry an error.
        let h: PauliSum = "Z0 + X0".parse().unwrap();
        let r = estimate(
            &StatevectorBackend,
            &Circuit::new(1),
            &[],
            &h,
            &EstimatorOptions::new(4),
        )
        .unwrap();
        assert!(r.groups.iter().all(|g| g.estimate.std_error > 0.0));
    }

    #[test]
    fn estimation_errors() {
        let h: PauliSum = "Z0 + X0 + 1.5".parse().unwrap();
        let c = Circuit::new(2);
        assert!(estimate(&StatevectorBackend, &c, &[], &h, &EstimatorOptions::new(1)).is_err());
        let constant: PauliSum = "2.5".parse().unwrap();
        let e = estimate(
            &StatevectorBackend,
            &c,
            &[],
            &constant,
            &EstimatorOptions::new(0),
        )
        .unwrap();
        assert_eq!(e.estimate, Estimate::exact(2.5));
        let bad =
            EstimatorOptions::new(100).with_allocation(ShotAllocation::Adaptive { pilot: 1.5 });
        assert!(estimate(&StatevectorBackend, &c, &[], &h, &bad).is_err());
        let cal = ReadoutCalibration::from_errors(&[(0.1, 0.1)]).unwrap();
        let wrong = EstimatorOptions::new(100).with_readout_mitigation(cal);
        assert!(estimate(&StatevectorBackend, &c, &[], &h, &wrong).is_err());
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
