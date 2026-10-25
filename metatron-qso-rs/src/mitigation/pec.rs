//! Probabilistic error cancellation (Temme, Bravyi and Gambetta 2017).
//!
//! The inverse of a Pauli channel `N(ρ) = Σ_P p_P PρP` is not a channel,
//! but it is a *quasi-probabilistic* mixture of Paulis:
//! `N⁻¹ = Σ_P η_P P·P` with real `η_P` summing to one, some negative. After
//! every noisy operation PEC inserts a Pauli drawn with probability
//! `|η_P|/γ` (`γ = Σ|η_P|`) and multiplies the result by `γ·sign(η_P)`. The
//! average over such circuits is the noiseless expectation value, exactly;
//! the price is a variance larger by the square of the total overhead
//! `Γ = Π γ`, which grows exponentially with the number of noisy gates.
//!
//! PEC here needs the noise as Pauli channels (Pauli twirling turns general
//! noise into its Pauli twirl, [`Channel::twirled`](crate::noise::Channel::twirled)).
//! The inserted Paulis are applied without noise, as when they are merged
//! into neighbouring single-qubit gates.
//!
//! ```
//! use metatron_qso::mitigation::pec::pec_expectation;
//! use metatron_qso::noise::{DeviceModel, GateRule};
//! use metatron_qso::quantum::{Circuit, PauliSum};
//!
//! let device = DeviceModel::new("dep").with_rule(GateRule::for_gate("cx").with_depolarizing(0.05));
//! let mut c = Circuit::new(2);
//! c.h(0).cx(0, 1);
//! let zz: PauliSum = "Z0 Z1".parse().unwrap();
//! let r = pec_expectation(&c, &[], &device, &zz, 4000, 1).unwrap();
//! assert!((r.estimate.value - 1.0).abs() < 4.0 * r.estimate.std_error);
//! assert!(r.overhead > 1.0);
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::estimation::Estimate;
use crate::noise::DensityMatrix;
use crate::noise::DeviceModel;
use crate::noise::device::{CompiledChannel, Op, Program, rotation_kind};
use crate::noise::trajectories::run_ops;
use crate::parallel::map_tasks;
use crate::quantum::{Circuit, PauliSum};
use crate::rng::{Rng, sample_cumulative};

/// Registers up to this size evaluate every sampled circuit exactly with
/// density matrices; larger ones run one trajectory per sample.
pub const PEC_DENSITY_QUBITS: usize = 10;

/// The result of [`pec_expectation`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PecResult {
    /// Error-cancelled estimate (unbiased) with its standard error.
    pub estimate: Estimate,
    /// Sampling overhead `Γ = Π γ`: the standard error is about `Γ` times
    /// that of an unmitigated estimate with the same number of samples.
    pub overhead: f64,
    /// Number of noisy operations that were inverted.
    pub noisy_operations: usize,
}

/// Whether the Pauli strings with base-4 indices `a` and `b` anticommute.
fn anticommute(mut a: usize, mut b: usize) -> bool {
    let mut odd = false;
    while a > 0 || b > 0 {
        let (x, y) = (a & 3, b & 3);
        odd ^= x != 0 && y != 0 && x != y;
        a >>= 2;
        b >>= 2;
    }
    odd
}

/// Quasi-probabilities `η` of the inverse of the Pauli channel with
/// probabilities `p` (base-4 indices, length `4^k`):
/// `η_P = 4^{−k} Σ_Q χ(P,Q)/f_Q` with fidelities `f_Q = Σ_P p_P χ(P,Q)`
/// and `χ = ±1` for commuting/anticommuting strings.
pub fn inverse_pauli_quasi_probabilities(p: &[f64]) -> Result<Vec<f64>> {
    let m = p.len();
    if m != 4 && m != 16 {
        return Err(QsoError::invalid(
            "probabilities",
            "Pauli channels on one or two qubits have 4 or 16 probabilities",
        ));
    }
    let chi = |a: usize, b: usize| if anticommute(a, b) { -1.0 } else { 1.0 };
    let f: Vec<f64> = (0..m)
        .map(|q| (0..m).map(|pp| p[pp] * chi(pp, q)).sum())
        .collect();
    if f.iter().any(|&x| x.abs() <= 1e-12) {
        return Err(QsoError::invalid(
            "probabilities",
            "the channel destroys information (a Pauli fidelity is 0) and cannot be inverted",
        ));
    }
    Ok((0..m)
        .map(|pp| (0..m).map(|q| chi(pp, q) / f[q]).sum::<f64>() / m as f64)
        .collect())
}

/// Sampling data of one noisy operation.
struct Inverse {
    quasi: Vec<f64>,
    cumulative: Vec<f64>,
    gamma: f64,
}

/// The program, its noisy operations and their inverses.
struct Plan {
    program: Program,
    /// `(op index, inverse)` for every channel operation.
    noisy: Vec<(usize, usize)>,
    inverses: Vec<Inverse>,
}

impl Plan {
    fn new(circuit: &Circuit, theta: &[f64], device: &DeviceModel) -> Result<Self> {
        device.validate()?;
        // Over-rotations change the gates themselves, not a channel after
        // them, so PEC cannot see them.
        if circuit.instructions().iter().any(|inst| {
            rotation_kind(inst.kind)
                && device
                    .rule_for(inst)
                    .is_some_and(|r| r.over_rotation != 0.0)
        }) {
            return Err(QsoError::invalid(
                "device",
                "PEC cannot cancel over-rotations (coherent errors in the gates themselves)",
            ));
        }
        let program = Program::compile(circuit, theta, device)?;
        let mut by_channel: Vec<Option<usize>> = vec![None; program.channels.len()];
        let mut inverses = Vec::new();
        let mut noisy = Vec::new();
        for (i, op) in program.ops.iter().enumerate() {
            if let Op::Channel { channel, .. } = op {
                let slot = match by_channel[*channel] {
                    Some(s) => s,
                    None => {
                        let ch = &program.channels[*channel].channel;
                        let p = ch.pauli_probabilities().ok_or_else(|| {
                            QsoError::invalid(
                                "device",
                                format!(
                                    "PEC needs Pauli noise, but {} is not a Pauli channel \
                                     (T1/T2 relaxation and coherent errors are not; describe \
                                     the twirled noise with Pauli channels)",
                                    ch.name()
                                ),
                            )
                        })?;
                        let quasi = inverse_pauli_quasi_probabilities(&p)?;
                        let gamma: f64 = quasi.iter().map(|x| x.abs()).sum();
                        let mut acc = 0.0;
                        let cumulative = quasi
                            .iter()
                            .map(|x| {
                                acc += x.abs() / gamma;
                                acc
                            })
                            .collect();
                        inverses.push(Inverse {
                            quasi,
                            cumulative,
                            gamma,
                        });
                        by_channel[*channel] = Some(inverses.len() - 1);
                        inverses.len() - 1
                    }
                };
                noisy.push((i, slot));
            }
        }
        Ok(Self {
            program,
            noisy,
            inverses,
        })
    }

    fn overhead(&self) -> f64 {
        self.noisy
            .iter()
            .map(|&(_, s)| self.inverses[s].gamma)
            .product()
    }

    /// The operations with Pauli `choices[j]` inserted after noisy
    /// operation `j`.
    fn corrected(&self, choices: &[usize]) -> Vec<Op> {
        let n = self.program.n_qubits;
        let mut ops = Vec::with_capacity(self.program.ops.len() + choices.len());
        let mut next = 0;
        for (i, op) in self.program.ops.iter().enumerate() {
            ops.push(op.clone());
            if next < self.noisy.len() && self.noisy[next].0 == i {
                let choice = choices[next];
                next += 1;
                if choice == 0 {
                    continue;
                }
                let Op::Channel { qubits, .. } = op else {
                    unreachable!("noisy operations are channels")
                };
                let k = qubits.len();
                let mut c = Circuit::new(n);
                for (pos, &q) in qubits.iter().enumerate() {
                    match (choice >> (2 * (k - 1 - pos))) & 3 {
                        1 => c.x(q),
                        2 => c.y(q),
                        3 => c.z(q),
                        _ => &mut c,
                    };
                }
                ops.push(Op::Gates(c));
            }
        }
        ops
    }

    /// `⟨H⟩` after `ops` (exact for small registers, one trajectory with
    /// `rng` otherwise).
    fn evaluate(&self, ops: &[Op], observable: &PauliSum, rng: Rng) -> f64 {
        let n = self.program.n_qubits;
        let channels: &[CompiledChannel] = &self.program.channels;
        if n <= PEC_DENSITY_QUBITS {
            let mut rho = DensityMatrix::zero(n).expect("small register");
            rho.run_ops(ops, channels);
            rho.expectation(observable).expect("sizes checked")
        } else {
            let (state, _) = run_ops(n, ops, channels, rng);
            observable.expectation(&state).expect("sizes checked")
        }
    }
}

/// Error-cancelled `⟨H⟩` of `circuit(theta)` under the Pauli noise of
/// `device`, averaged over `samples` randomly corrected circuits.
pub fn pec_expectation(
    circuit: &Circuit,
    theta: &[f64],
    device: &DeviceModel,
    observable: &PauliSum,
    samples: usize,
    seed: u64,
) -> Result<PecResult> {
    if samples == 0 {
        return Err(QsoError::invalid("samples", "need at least one sample"));
    }
    if observable.min_qubits() > circuit.n_qubits() {
        return Err(QsoError::QubitOutOfRange {
            qubit: observable.min_qubits() - 1,
            n_qubits: circuit.n_qubits(),
        });
    }
    let plan = Plan::new(circuit, theta, device)?;
    let overhead = plan.overhead();
    let base = Rng::seed_from_u64(seed);
    let values = map_tasks(samples, |s| {
        let mut rng = base.fork(s as u64);
        let mut sign = 1.0;
        let choices: Vec<usize> = plan
            .noisy
            .iter()
            .map(|&(_, slot)| {
                let inv = &plan.inverses[slot];
                let j = sample_cumulative(&inv.cumulative, &mut rng);
                if inv.quasi[j] < 0.0 {
                    sign = -sign;
                }
                j
            })
            .collect();
        let ops = plan.corrected(&choices);
        overhead * sign * plan.evaluate(&ops, observable, rng)
    });
    Ok(PecResult {
        estimate: Estimate::from_samples(&values),
        overhead,
        noisy_operations: plan.noisy.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::{ChannelSpec, GateRule, QubitProperties};
    use crate::quantum::StateVector;

    /// The exact PEC expectation: the sum over all correction choices.
    fn pec_exact(plan: &Plan, observable: &PauliSum) -> f64 {
        let sizes: Vec<usize> = plan
            .noisy
            .iter()
            .map(|&(_, s)| plan.inverses[s].quasi.len())
            .collect();
        let total: usize = sizes.iter().product();
        let mut sum = 0.0;
        for mut index in 0..total {
            let mut weight = 1.0;
            let choices: Vec<usize> = plan
                .noisy
                .iter()
                .zip(&sizes)
                .map(|(&(_, s), &m)| {
                    let j = index % m;
                    index /= m;
                    weight *= plan.inverses[s].quasi[j];
                    j
                })
                .collect();
            sum += weight
                * plan.evaluate(&plan.corrected(&choices), observable, Rng::seed_from_u64(0));
        }
        sum
    }

    fn device() -> DeviceModel {
        DeviceModel::new("pauli")
            .with_rule(
                GateRule::for_gate("cx")
                    .with_depolarizing(0.04)
                    .with_channel(ChannelSpec::Pauli([("ZZ".to_string(), 0.03)].into())),
            )
            .with_rule(GateRule::any().with_channel(ChannelSpec::Pauli(
                [("X".to_string(), 0.02), ("Z".to_string(), 0.01)].into(),
            )))
    }

    #[test]
    fn quasi_probabilities_invert_pauli_channels() {
        let p = 0.03;
        let dep = [1.0 - p, p / 3.0, p / 3.0, p / 3.0];
        let eta = inverse_pauli_quasi_probabilities(&dep).unwrap();
        let f = 1.0 - 4.0 * p / 3.0;
        let gamma: f64 = eta.iter().map(|x| x.abs()).sum();
        assert!((gamma - (3.0 / f - 1.0) / 2.0).abs() < 1e-14);
        assert!((eta.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        // Composing with the channel gives fidelity 1 for every Pauli.
        let two: Vec<f64> = (0..16)
            .map(|i| [0.9, 0.02, 0.03, 0.01][i % 4] * [0.95, 0.01, 0.01, 0.03][i / 4])
            .collect();
        let eta = inverse_pauli_quasi_probabilities(&two).unwrap();
        for q in 0..16 {
            let fq: f64 = (0..16)
                .map(|pp| two[pp] * if anticommute(pp, q) { -1.0 } else { 1.0 })
                .sum();
            let gq: f64 = (0..16)
                .map(|pp| eta[pp] * if anticommute(pp, q) { -1.0 } else { 1.0 })
                .sum();
            assert!((fq * gq - 1.0).abs() < 1e-12);
        }
        assert!(inverse_pauli_quasi_probabilities(&[0.25; 4]).is_err());
        // Negative fidelities are invertible: a certain bit flip, strong
        // dephasing.
        assert!(inverse_pauli_quasi_probabilities(&[0.0, 1.0, 0.0, 0.0]).is_ok());
        assert!(inverse_pauli_quasi_probabilities(&[0.3, 0.0, 0.0, 0.7]).is_ok());
    }

    #[test]
    fn pec_is_exactly_unbiased() {
        let mut c = Circuit::new(2);
        c.ry(0, 0.7).cx(0, 1).rz(1, 0.4).h(1);
        let h: PauliSum = "Z0 Z1 + 0.6 X1 - 0.3 Y0".parse().unwrap();
        let ideal = h
            .expectation(&StateVector::from_circuit(&c, &[]).unwrap())
            .unwrap();
        let plan = Plan::new(&c, &[], &device()).unwrap();
        // ry, rz, h: one channel each; cx: depolarising and ZZ on the pair,
        // merged into one channel.
        assert_eq!(plan.noisy.len(), 3 + 1);
        let noisy = plan.evaluate(&plan.program.ops, &h, Rng::seed_from_u64(0));
        assert!((noisy - ideal).abs() > 0.01);
        assert!((pec_exact(&plan, &h) - ideal).abs() < 1e-12);
    }

    #[test]
    fn sampled_pec_lies_within_its_error_bars() {
        let mut c = Circuit::new(3);
        c.h(0).cx(0, 1).ry(2, 0.5).cx(1, 2).rx(0, 0.3);
        let h: PauliSum = "Z0 Z2 + X0 X1 X2".parse().unwrap();
        let ideal = h
            .expectation(&StateVector::from_circuit(&c, &[]).unwrap())
            .unwrap();
        let r = pec_expectation(&c, &[], &device(), &h, 20_000, 3).unwrap();
        assert!(
            (r.estimate.value - ideal).abs() < 4.0 * r.estimate.std_error,
            "{:?} vs {ideal}",
            r.estimate
        );
        assert_eq!(r.noisy_operations, 3 + 2);
        let plan = Plan::new(&c, &[], &device()).unwrap();
        assert!((r.overhead - plan.overhead()).abs() < 1e-15 && r.overhead > 1.1);
        // Trajectories for larger registers give the same (unbiased) mean.
        let mut big = Circuit::new(11);
        big.h(0).cx(0, 1).ry(2, 0.5).cx(1, 2);
        let zz: PauliSum = "Z0 Z1".parse().unwrap();
        let r = pec_expectation(&big, &[], &device(), &zz, 4000, 4).unwrap();
        assert!((r.estimate.value - 1.0).abs() < 4.0 * r.estimate.std_error);
    }

    #[test]
    fn non_pauli_noise_is_rejected() {
        let device = DeviceModel::new("t1")
            .with_default_qubit(QubitProperties::new().with_t1_t2(10.0, 8.0))
            .with_rule(GateRule::any().with_duration(1.0));
        let mut c = Circuit::new(1);
        c.h(0);
        let z: PauliSum = "Z0".parse().unwrap();
        assert!(pec_expectation(&c, &[], &device, &z, 10, 0).is_err());
    }
}
