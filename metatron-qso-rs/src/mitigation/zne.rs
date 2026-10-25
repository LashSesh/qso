//! Zero-noise extrapolation (ZNE).
//!
//! The circuit is run at several noise levels and the results are
//! extrapolated to zero noise. Noise is amplified by *unitary folding*
//! (Giurgica-Tiron et al. 2020): replacing `G` by `G G† G` leaves the ideal
//! circuit unchanged and triples the noise of `G`.
//!
//! * [`Folding::Global`] folds the whole circuit, `C (C† C)^k`, and folds
//!   the last gates once more for fractional scale factors.
//! * [`Folding::Random`] folds individual gates chosen at random.
//!
//! Odd scale factors (1, 3, 5, …) fold every gate equally often and so
//! amplify every gate's noise by the same factor; on a test circuit with
//! depolarising noise Richardson extrapolation from 1, 3, 5 reduces the
//! error by a factor of about 400. Fractional factors fold only some gates
//! and amplify the noise by the gate-count ratio only when all gates are
//! about equally noisy; with noisy two-qubit and quiet one-qubit gates they
//! can make the estimate worse.
//!
//! Extrapolation propagates the standard errors of the measured values to
//! the zero-noise estimate (the estimate is a linear combination of the
//! values, or, for the exponential fit, linearised around it).
//!
//! ```
//! use metatron_qso::estimation::Estimate;
//! use metatron_qso::mitigation::zne::{Extrapolation, Folding, zne};
//! use metatron_qso::noise::{DensityMatrix, DeviceModel, GateRule};
//! use metatron_qso::quantum::{Circuit, PauliSum};
//!
//! let device = DeviceModel::new("dep").with_rule(GateRule::any().with_depolarizing(0.02));
//! let mut c = Circuit::new(2);
//! c.h(0).cx(0, 1).rz(1, 0.3).cx(0, 1).h(0);
//! let z: PauliSum = "Z0".parse().unwrap();
//! let run = |folded: &Circuit| {
//!     let mut rho = DensityMatrix::zero(2)?;
//!     rho.apply_device(folded, &[], &device)?;
//!     Ok(Estimate::exact(rho.expectation(&z)?))
//! };
//! let noisy = run(&c).unwrap().value;
//! let result = zne(&c, &[1.0, 3.0, 5.0], Folding::Global, Extrapolation::Richardson, run).unwrap();
//! let ideal = 0.3f64.cos();
//! assert!((result.estimate.value - ideal).abs() < 0.1 * (noisy - ideal).abs());
//! ```

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::estimation::Estimate;
use crate::quantum::{Circuit, Instruction};
use crate::rng::Rng;

/// How circuits are folded to amplify their noise.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Folding {
    /// `C (C† C)^k`, plus the last gates folded once more for fractional
    /// scale factors.
    Global,
    /// Gates folded individually (`G → G G† G`), chosen at random.
    Random { seed: u64 },
}

/// How values at several noise scales are extrapolated to zero.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Extrapolation {
    /// Least-squares straight line.
    Linear,
    /// The polynomial through all points (degree = points − 1).
    Richardson,
    /// Least-squares polynomial of the given degree.
    Polynomial { degree: usize },
    /// `E(λ) = a + b·e^{−cλ}` with known asymptote `a` (0 for Pauli
    /// observables under depolarising noise), fitted linearly to
    /// `ln|E − a|`.
    Exponential { asymptote: f64 },
}

/// The result of [`zne`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZneResult {
    /// Zero-noise estimate with propagated standard error.
    pub estimate: Estimate,
    /// Realised scale factors (folding rounds to whole gates).
    pub scales: Vec<f64>,
    /// Values measured at those scales.
    pub values: Vec<Estimate>,
}

/// Largest folded circuit (gates).
pub const MAX_FOLDED_GATES: usize = 1 << 22;

fn check_scale(scale: f64) -> Result<()> {
    if !(scale >= 1.0 && scale.is_finite()) {
        return Err(QsoError::invalid(
            "scale",
            format!("scale factors must be finite and at least 1, got {scale}"),
        ));
    }
    Ok(())
}

fn check_size(d: usize, scale: f64) -> Result<()> {
    if d as f64 * scale > MAX_FOLDED_GATES as f64 {
        return Err(QsoError::invalid(
            "scale",
            format!("folding {d} gates by {scale} exceeds {MAX_FOLDED_GATES} gates"),
        ));
    }
    Ok(())
}

fn empty_like(circuit: &Circuit) -> Circuit {
    let mut c = Circuit::new(circuit.n_qubits());
    c.reserve_parameters(circuit.num_parameters());
    c
}

fn push(c: &mut Circuit, inst: Instruction) {
    c.try_push(inst)
        .expect("instructions of a valid circuit stay valid");
}

/// Fold `circuit` to amplify its noise by about `scale` (≥ 1). Returns the
/// folded circuit and the realised scale `gates(folded) / gates(circuit)`.
pub fn fold(circuit: &Circuit, scale: f64, folding: Folding) -> Result<(Circuit, f64)> {
    check_scale(scale)?;
    let d = circuit.len();
    if d == 0 {
        return Ok((circuit.clone(), 1.0));
    }
    check_size(d, scale)?;
    let gates = circuit.instructions();
    let mut out = empty_like(circuit);
    match folding {
        Folding::Global => {
            let k = ((scale - 1.0) / 2.0).floor() as usize;
            let m = (((scale - 1.0 - 2.0 * k as f64) / 2.0) * d as f64).round() as usize;
            let m = m.min(d);
            for inst in gates {
                push(&mut out, inst.clone());
            }
            for _ in 0..k {
                for inst in gates.iter().rev() {
                    push(&mut out, inst.inverse());
                }
                for inst in gates {
                    push(&mut out, inst.clone());
                }
            }
            let tail = &gates[d - m..];
            for inst in tail.iter().rev() {
                push(&mut out, inst.inverse());
            }
            for inst in tail {
                push(&mut out, inst.clone());
            }
        }
        Folding::Random { seed } => {
            let folds = (((scale - 1.0) * d as f64) / 2.0).round() as usize;
            let mut times = vec![folds / d; d];
            let mut order: Vec<usize> = (0..d).collect();
            Rng::seed_from_u64(seed).shuffle(&mut order);
            for &i in &order[..folds % d] {
                times[i] += 1;
            }
            for (inst, &t) in gates.iter().zip(&times) {
                push(&mut out, inst.clone());
                for _ in 0..t {
                    push(&mut out, inst.inverse());
                    push(&mut out, inst.clone());
                }
            }
        }
    }
    let realised = out.len() as f64 / d as f64;
    Ok((out, realised))
}

/// Coefficients `γ` of the least-squares polynomial of `degree` evaluated
/// at zero: `p(0) = Σ γ_i y_i`.
fn zero_coefficients(scales: &[f64], degree: usize) -> Result<Vec<f64>> {
    let n = scales.len();
    if degree >= n {
        return Err(QsoError::invalid(
            "scales",
            format!(
                "a degree-{degree} fit needs at least {} scale factors",
                degree + 1
            ),
        ));
    }
    let x = DMatrix::from_fn(n, degree + 1, |i, j| scales[i].powi(j as i32));
    let xtx = x.transpose() * &x;
    let inv = xtx
        .try_inverse()
        .ok_or_else(|| QsoError::invalid("scales", "scale factors must be distinct for the fit"))?;
    let rows = inv * x.transpose();
    Ok(rows.row(0).iter().copied().collect())
}

fn combine(gamma: &[f64], values: &[Estimate]) -> Estimate {
    Estimate {
        value: gamma.iter().zip(values).map(|(g, v)| g * v.value).sum(),
        std_error: gamma
            .iter()
            .zip(values)
            .map(|(g, v)| (g * v.std_error).powi(2))
            .sum::<f64>()
            .sqrt(),
        samples: values.iter().map(|v| v.samples).sum(),
    }
}

/// Extrapolate `(scale, value)` pairs to scale 0.
pub fn extrapolate(scales: &[f64], values: &[Estimate], method: Extrapolation) -> Result<Estimate> {
    if scales.len() != values.len() || scales.is_empty() {
        return Err(QsoError::invalid(
            "values",
            "need one value per scale factor, and at least one",
        ));
    }
    match method {
        Extrapolation::Linear => Ok(combine(&zero_coefficients(scales, 1)?, values)),
        Extrapolation::Polynomial { degree } => {
            Ok(combine(&zero_coefficients(scales, degree)?, values))
        }
        Extrapolation::Richardson => {
            // Lagrange interpolation at 0: γ_i = Π_{j≠i} λ_j / (λ_j − λ_i).
            let mut gamma = Vec::with_capacity(scales.len());
            for (i, &li) in scales.iter().enumerate() {
                let mut g = 1.0;
                for (j, &lj) in scales.iter().enumerate() {
                    if i != j {
                        if lj == li {
                            return Err(QsoError::invalid(
                                "scales",
                                "Richardson extrapolation needs distinct scale factors",
                            ));
                        }
                        g *= lj / (lj - li);
                    }
                }
                gamma.push(g);
            }
            Ok(combine(&gamma, values))
        }
        Extrapolation::Exponential { asymptote } => {
            let shifted: Vec<f64> = values.iter().map(|v| v.value - asymptote).collect();
            let sign = shifted[0].signum();
            if shifted.iter().any(|&s| s == 0.0 || s.signum() != sign) {
                return Err(QsoError::invalid(
                    "values",
                    "exponential extrapolation needs all values on one side of the asymptote",
                ));
            }
            let logs: Vec<Estimate> = values
                .iter()
                .zip(&shifted)
                .map(|(v, s)| Estimate {
                    value: s.abs().ln(),
                    std_error: v.std_error / s.abs(),
                    samples: v.samples,
                })
                .collect();
            let alpha = combine(&zero_coefficients(scales, 1)?, &logs);
            let amplitude = alpha.value.exp();
            Ok(Estimate {
                value: asymptote + sign * amplitude,
                std_error: amplitude * alpha.std_error,
                samples: alpha.samples,
            })
        }
    }
}

/// Zero-noise extrapolation: fold `circuit` to each scale, evaluate it
/// with `execute` (which returns an [`Estimate`], exact or from shots) and
/// extrapolate to zero noise.
pub fn zne<F>(
    circuit: &Circuit,
    scales: &[f64],
    folding: Folding,
    extrapolation: Extrapolation,
    mut execute: F,
) -> Result<ZneResult>
where
    F: FnMut(&Circuit) -> Result<Estimate>,
{
    for &s in scales {
        check_scale(s)?;
        check_size(circuit.len(), s)?;
    }
    if circuit.is_empty() {
        // Nothing to fold: the value at scale 1 is the zero-noise value.
        let value = execute(circuit)?;
        return Ok(ZneResult {
            estimate: value,
            scales: vec![1.0],
            values: vec![value],
        });
    }
    let mut realised = Vec::with_capacity(scales.len());
    let mut values = Vec::with_capacity(scales.len());
    for &s in scales {
        let (folded, actual) = fold(circuit, s, folding)?;
        realised.push(actual);
        values.push(execute(&folded)?);
    }
    for (i, a) in realised.iter().enumerate() {
        if realised[..i].contains(a) {
            return Err(QsoError::invalid(
                "scales",
                format!(
                    "the realised scale factors {realised:?} coincide (folding rounds to whole \
                     gates); use more distinct, preferably odd, scale factors"
                ),
            ));
        }
    }
    let estimate = extrapolate(&realised, &values, extrapolation)?;
    Ok(ZneResult {
        estimate,
        scales: realised,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DeviceBackend;
    use crate::estimation::{EstimatorOptions, estimate};
    use crate::noise::{DensityMatrix, DeviceModel, GateRule};
    use crate::quantum::{PauliSum, StateVector};

    fn circuit() -> Circuit {
        let mut c = Circuit::new(3);
        let t = c.new_parameter();
        c.h(0)
            .ry(1, t)
            .cx(0, 1)
            .rz(2, 0.4)
            .cx(1, 2)
            .rx(0, 0.3)
            .cx(0, 2)
            .ry(2, -0.8);
        c
    }

    #[test]
    fn folding_keeps_the_ideal_circuit() {
        let c = circuit();
        let theta = [0.9];
        let ideal = StateVector::from_circuit(&c, &theta).unwrap();
        for folding in [Folding::Global, Folding::Random { seed: 4 }] {
            for scale in [1.0, 1.5, 2.0, 3.0, 4.25, 7.0] {
                let (f, actual) = fold(&c, scale, folding).unwrap();
                assert!(
                    (actual - scale).abs() <= 1.0 / c.len() as f64 + 1e-12,
                    "{scale}: {actual}"
                );
                let psi = StateVector::from_circuit(&f, &theta).unwrap();
                assert!((psi.fidelity(&ideal).unwrap() - 1.0).abs() < 1e-12);
            }
        }
        // Empty circuits and coinciding realised scales.
        let empty = Circuit::new(1);
        let r = zne(
            &empty,
            &[1.0, 3.0],
            Folding::Global,
            Extrapolation::Richardson,
            |_| Ok(Estimate::exact(0.7)),
        )
        .unwrap();
        assert_eq!((r.estimate.value, r.scales.clone()), (0.7, vec![1.0]));
        let mut one = Circuit::new(1);
        one.h(0);
        let err = zne(
            &one,
            &[1.0, 2.0, 3.0],
            Folding::Global,
            Extrapolation::Richardson,
            |_| Ok(Estimate::exact(0.5)),
        );
        assert!(err.unwrap_err().to_string().contains("coincide"));
        let (f3, s3) = fold(&c, 3.0, Folding::Global).unwrap();
        assert_eq!((f3.len(), s3), (3 * c.len(), 3.0));
        assert!(fold(&c, 0.5, Folding::Global).is_err());
    }

    #[test]
    fn extrapolations_are_exact_for_their_models() {
        let scales = [1.0, 2.0, 3.0, 5.0];
        let exact = |f: &dyn Fn(f64) -> f64| -> Vec<Estimate> {
            scales.iter().map(|&s| Estimate::exact(f(s))).collect()
        };
        let line = exact(&|s| 0.9 - 0.1 * s);
        assert!(
            (extrapolate(&scales, &line, Extrapolation::Linear)
                .unwrap()
                .value
                - 0.9)
                .abs()
                < 1e-14
        );
        let cubic = exact(&|s| 1.0 - 0.2 * s + 0.03 * s * s - 0.002 * s.powi(3));
        let r = extrapolate(&scales, &cubic, Extrapolation::Richardson).unwrap();
        assert!((r.value - 1.0).abs() < 1e-12);
        let q = extrapolate(&scales, &cubic, Extrapolation::Polynomial { degree: 3 }).unwrap();
        assert!((q.value - 1.0).abs() < 1e-12);
        let decay = exact(&|s| 0.1 + 0.8 * (-0.3 * s).exp());
        let e = extrapolate(
            &scales,
            &decay,
            Extrapolation::Exponential { asymptote: 0.1 },
        )
        .unwrap();
        assert!((e.value - 0.9).abs() < 1e-12);
        // Standard errors propagate linearly: Richardson on (1, 3) gives
        // 1.5·E(1) − 0.5·E(3).
        let v = [
            Estimate {
                value: 0.8,
                std_error: 0.01,
                samples: 100,
            },
            Estimate {
                value: 0.6,
                std_error: 0.02,
                samples: 100,
            },
        ];
        let r = extrapolate(&[1.0, 3.0], &v, Extrapolation::Richardson).unwrap();
        assert!((r.value - 0.9).abs() < 1e-15);
        assert!((r.std_error - (0.015f64.powi(2) + 0.01f64.powi(2)).sqrt()).abs() < 1e-15);
        assert!(extrapolate(&[1.0, 1.0], &v, Extrapolation::Richardson).is_err());
        assert!(extrapolate(&[1.0], &v[..1], Extrapolation::Linear).is_err());
        let mixed = [Estimate::exact(0.5), Estimate::exact(-0.5)];
        assert!(
            extrapolate(
                &[1.0, 3.0],
                &mixed,
                Extrapolation::Exponential { asymptote: 0.0 }
            )
            .is_err()
        );
    }

    #[test]
    fn zne_recovers_noiseless_values() {
        let c = circuit();
        let theta = [0.9];
        let h: PauliSum = "Z0 Z1 + 0.5 X2 - 0.3 Z1".parse().unwrap();
        let ideal = h
            .expectation(&StateVector::from_circuit(&c, &theta).unwrap())
            .unwrap();
        let device = DeviceModel::new("dep")
            .with_rule(GateRule::any().with_num_qubits(2).with_depolarizing(0.01))
            .with_rule(GateRule::any().with_depolarizing(0.002));
        let exact = |f: &Circuit| {
            let mut rho = DensityMatrix::zero(3)?;
            rho.apply_device(f, &theta, &device)?;
            Ok(Estimate::exact(rho.expectation(&h)?))
        };
        let noisy = exact(&c).unwrap().value;
        // Odd scale factors fold every gate equally often, so every gate's
        // noise is amplified by the same factor.
        for (folding, scales, method, gain) in [
            (
                Folding::Global,
                &[1.0, 3.0, 5.0][..],
                Extrapolation::Linear,
                5.0,
            ),
            (
                Folding::Global,
                &[1.0, 3.0, 5.0],
                Extrapolation::Richardson,
                100.0,
            ),
            (
                Folding::Global,
                &[1.0, 3.0, 5.0, 7.0],
                Extrapolation::Richardson,
                1000.0,
            ),
            (
                Folding::Global,
                &[1.0, 3.0, 5.0, 7.0],
                Extrapolation::Polynomial { degree: 2 },
                50.0,
            ),
            (
                Folding::Global,
                &[1.0, 3.0, 5.0],
                Extrapolation::Exponential { asymptote: 0.0 },
                100.0,
            ),
            (
                Folding::Random { seed: 1 },
                &[1.0, 3.0, 5.0],
                Extrapolation::Richardson,
                100.0,
            ),
        ] {
            let r = zne(&c, scales, folding, method, exact).unwrap();
            let error = (r.estimate.value - ideal).abs();
            assert!(
                error * gain < (noisy - ideal).abs(),
                "{folding:?} {method:?}: {error} vs {}",
                (noisy - ideal).abs()
            );
        }

        // With shots, the mitigated value lies within its error bars.
        let backend = DeviceBackend::density_matrix(device.clone()).unwrap();
        let mut seed = 0;
        let sampled = |f: &Circuit| {
            seed += 1;
            Ok(estimate(
                &backend,
                f,
                &theta,
                &h,
                &EstimatorOptions::new(400_000).with_seed(seed),
            )?
            .estimate)
        };
        let r = zne(
            &c,
            &[1.0, 3.0, 5.0],
            Folding::Global,
            Extrapolation::Richardson,
            sampled,
        )
        .unwrap();
        assert!(r.estimate.std_error > 0.0);
        assert!(
            (r.estimate.value - ideal).abs() < 4.0 * r.estimate.std_error,
            "{} ± {} vs {ideal}",
            r.estimate.value,
            r.estimate.std_error
        );
        assert_eq!(r.scales, vec![1.0, 3.0, 5.0]);
    }
}
