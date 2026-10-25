//! Quantum trajectories (Monte-Carlo wave functions).
//!
//! A noisy circuit on n qubits needs a 4ⁿ-entry density matrix, but each
//! *trajectory* needs only a 2ⁿ-entry state vector: every channel
//! `ρ ↦ Σ_i K_i ρ K_i†` is replaced by applying one Kraus operator, chosen
//! with probability `‖K_i ψ‖²`, and renormalising. The average of `|ψ⟩⟨ψ|`
//! over trajectories is the density matrix, so averages of observables and
//! of measurement outcomes converge to the density-matrix results with the
//! usual `1/√N` statistical error. Channels that are mixtures of unitaries
//! (Pauli, depolarising and coherent errors) are sampled without looking at
//! the state.
//!
//! Every trajectory has its own random stream derived from the seed, so
//! results do not depend on the number of threads.
//!
//! ```
//! use metatron_qso::noise::{DeviceModel, GateRule, TrajectorySimulator};
//! use metatron_qso::quantum::{Circuit, PauliSum};
//!
//! let device = DeviceModel::new("depolarising").with_rule(GateRule::any().with_depolarizing(0.02));
//! let mut c = Circuit::new(2);
//! c.h(0).cx(0, 1);
//! let zz: PauliSum = "Z0 Z1".parse().unwrap();
//! let sim = TrajectorySimulator::new(device, 2000, 7).unwrap();
//! let e = sim.expectation(&c, &[], &zz).unwrap();
//! assert!(e.value < 1.0 && e.value > 0.9 && e.std_error < 0.01);
//! ```

use num_complex::Complex64 as C64;

use super::device::{CompiledChannel, DeviceModel, Op, Program, Unraveling};
use super::{DensityMatrix, MAX_DENSITY_QUBITS};
use crate::error::{QsoError, Result};
use crate::estimation::Estimate;
use crate::parallel::{map_tasks, sum_ranges};
use crate::quantum::kernels::{apply_1q, apply_2q, insert_zeros};
use crate::quantum::{Circuit, Counts, PauliSum, StateVector, plan};
use crate::rng::{Rng, sample_cumulative};

const ZERO: C64 = C64::new(0.0, 0.0);

/// Registers from this size run their trajectories one after the other
/// (each state vector multi-threaded) instead of several at once.
const SEQUENTIAL_QUBITS: usize = 20;

/// Trajectories processed per batch when results are accumulated in order.
const BATCH: usize = 64;

/// Simulates noisy circuits by averaging quantum trajectories.
#[derive(Clone, Debug)]
pub struct TrajectorySimulator {
    device: DeviceModel,
    trajectories: usize,
    seed: u64,
}

/// Sum of reduced-density-matrix contributions (up to 4×4).
struct Reduced([C64; 16]);

impl std::iter::Sum for Reduced {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Reduced([ZERO; 16]), |mut a, b| {
            for (x, y) in a.0.iter_mut().zip(b.0) {
                *x += y;
            }
            a
        })
    }
}

/// `ρ[a][b] = Σ ψ_a ψ_b*` over the other qubits, for one qubit or an
/// ordered pair (first = more significant local bit). Row-major `d × d`.
fn reduced_matrix(amps: &[C64], qubits: &[usize]) -> Reduced {
    let k = qubits.len();
    let d = 1usize << k;
    let mut sorted = qubits.to_vec();
    sorted.sort_unstable();
    // Offsets of the local basis states, local index 2·x_first + x_second.
    let offsets: Vec<usize> = (0..d)
        .map(|l| {
            (0..k).fold(0, |acc, j| {
                if (l >> (k - 1 - j)) & 1 == 1 {
                    acc | (1 << qubits[j])
                } else {
                    acc
                }
            })
        })
        .collect();
    let count = (amps.len() >> k) as u64;
    sum_ranges(count, |start, end| {
        let mut acc = [ZERO; 16];
        for kk in start..end {
            let base = insert_zeros(kk as usize, &sorted);
            let mut v = [ZERO; 4];
            for (l, off) in offsets.iter().enumerate() {
                v[l] = amps[base | off];
            }
            for a in 0..d {
                for b in 0..d {
                    acc[a * d + b] += v[a] * v[b].conj();
                }
            }
        }
        Reduced(acc)
    })
}

/// Apply a `d × d` matrix to one qubit or an ordered pair.
fn apply_local(amps: &mut [C64], qubits: &[usize], m: &[C64]) {
    match qubits {
        [q] => apply_1q(amps, *q, 0, &[m[0], m[1], m[2], m[3]]),
        [a, b] => {
            let m: [C64; 16] = m.try_into().expect("4x4 matrix");
            apply_2q(amps, *a, *b, 0, &m);
        }
        _ => unreachable!("channels act on one or two qubits"),
    }
}

/// Apply one Kraus operator of `ch`, chosen with the Born probabilities.
pub(crate) fn apply_stochastic(
    amps: &mut [C64],
    ch: &CompiledChannel,
    qubits: &[usize],
    rng: &mut Rng,
) {
    match &ch.unraveling {
        Unraveling::Mixture {
            cumulative,
            unitaries,
        } => {
            let i = sample_cumulative(cumulative, rng);
            if let Some(u) = &unitaries[i] {
                apply_local(amps, qubits, u);
            }
        }
        Unraveling::General { kraus, gram } => {
            let d = 1usize << qubits.len();
            let rho = reduced_matrix(amps, qubits).0;
            // ‖Kψ‖² = Σ_{r,c} (K†K)[r][c] ρ[c][r].
            let probs: Vec<f64> = gram
                .iter()
                .map(|g| {
                    let mut p = ZERO;
                    for r in 0..d {
                        for c in 0..d {
                            p += g[r * d + c] * rho[c * d + r];
                        }
                    }
                    p.re.max(0.0)
                })
                .collect();
            let mut cumulative = Vec::with_capacity(probs.len());
            let mut acc = 0.0;
            for p in &probs {
                acc += p;
                cumulative.push(acc);
            }
            let i = sample_cumulative(&cumulative, rng);
            let scale = 1.0 / probs[i].sqrt();
            let m: Vec<C64> = kraus[i].iter().map(|z| z * scale).collect();
            apply_local(amps, qubits, &m);
        }
    }
}

/// Run one trajectory of `program`; returns the final state and the
/// trajectory's random stream (for sampling from it).
pub(crate) fn run_trajectory(program: &Program, rng: Rng) -> (StateVector, Rng) {
    run_ops(program.n_qubits, &program.ops, &program.channels, rng)
}

/// Run compiled operations on `|0…0⟩` as one trajectory.
pub(crate) fn run_ops(
    n_qubits: usize,
    ops: &[Op],
    channels: &[CompiledChannel],
    mut rng: Rng,
) -> (StateVector, Rng) {
    let mut state = StateVector::zero(n_qubits);
    let amps = state.amplitudes_mut();
    for op in ops {
        match op {
            Op::Gates(c) => plan::run(amps, c, &[]),
            Op::Channel { channel, qubits } => {
                apply_stochastic(amps, &channels[*channel], qubits, &mut rng)
            }
        }
    }
    (state, rng)
}

/// Flip measured bits with the per-qubit readout errors.
pub(crate) fn apply_readout(b: usize, readout: &[(f64, f64)], rng: &mut Rng) -> usize {
    let mut b = b;
    for (q, &(p01, p10)) in readout.iter().enumerate() {
        if p01 == 0.0 && p10 == 0.0 {
            continue;
        }
        let flip = if (b >> q) & 1 == 0 { p01 } else { p10 };
        if rng.uniform() < flip {
            b ^= 1 << q;
        }
    }
    b
}

impl TrajectorySimulator {
    /// A simulator averaging `trajectories` trajectories with random
    /// streams derived from `seed`.
    pub fn new(device: DeviceModel, trajectories: usize, seed: u64) -> Result<Self> {
        device.validate()?;
        if trajectories == 0 {
            return Err(QsoError::invalid(
                "trajectories",
                "need at least one trajectory",
            ));
        }
        Ok(Self {
            device,
            trajectories,
            seed,
        })
    }

    pub fn device(&self) -> &DeviceModel {
        &self.device
    }

    pub fn trajectories(&self) -> usize {
        self.trajectories
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    fn program(&self, circuit: &Circuit, theta: &[f64]) -> Result<Program> {
        Program::compile(circuit, theta, &self.device)
    }

    /// Map trajectories `0..count` to results (`f(k, state, rng)`), in
    /// parallel over trajectories for small registers.
    fn map<T: Send>(
        &self,
        program: &Program,
        count: usize,
        f: impl Fn(usize, StateVector, Rng) -> T + Sync + Send,
    ) -> Vec<T> {
        let base = Rng::seed_from_u64(self.seed);
        let one = |k: usize| {
            let (state, rng) = run_trajectory(program, base.fork(k as u64));
            f(k, state, rng)
        };
        if program.n_qubits >= SEQUENTIAL_QUBITS {
            (0..count).map(one).collect()
        } else {
            map_tasks(count, one)
        }
    }

    /// Averages of several observables over the same trajectories.
    pub fn expectations(
        &self,
        circuit: &Circuit,
        theta: &[f64],
        observables: &[PauliSum],
    ) -> Result<Vec<Estimate>> {
        for h in observables {
            if h.min_qubits() > circuit.n_qubits() {
                return Err(QsoError::QubitOutOfRange {
                    qubit: h.min_qubits() - 1,
                    n_qubits: circuit.n_qubits(),
                });
            }
        }
        let program = self.program(circuit, theta)?;
        let values = self.map(&program, self.trajectories, |_, state, _| {
            observables
                .iter()
                .map(|h| h.expectation(&state).expect("sizes checked"))
                .collect::<Vec<f64>>()
        });
        Ok((0..observables.len())
            .map(|j| Estimate::from_samples(&values.iter().map(|v| v[j]).collect::<Vec<_>>()))
            .collect())
    }

    /// Average of `⟨ψ|H|ψ⟩` over the trajectories, with its standard error.
    pub fn expectation(
        &self,
        circuit: &Circuit,
        theta: &[f64],
        observable: &PauliSum,
    ) -> Result<Estimate> {
        Ok(self.expectations(circuit, theta, std::slice::from_ref(observable))?[0])
    }

    /// Measure all qubits `shots` times, with the device's readout errors.
    /// Shots are spread evenly over the trajectories (one shot per
    /// trajectory when `shots` does not exceed their number, which gives
    /// independent samples of the noisy distribution).
    pub fn sample(&self, circuit: &Circuit, theta: &[f64], shots: usize) -> Result<Counts> {
        let program = self.program(circuit, theta)?;
        if shots == 0 {
            return Ok(Counts::new());
        }
        let runs = self.trajectories.min(shots);
        let parts = self.map(&program, runs, |k, state, mut rng| {
            let mine = shots / runs + usize::from(k < shots % runs);
            let mut out = Counts::new();
            for (b, n) in state.sample(mine, &mut rng) {
                for _ in 0..n {
                    let read = apply_readout(b, &program.readout, &mut rng);
                    *out.entry(read).or_insert(0) += 1;
                }
            }
            out
        });
        let mut total = Counts::new();
        for part in parts {
            for (b, n) in part {
                *total.entry(b).or_insert(0) += n;
            }
        }
        Ok(total)
    }

    /// The density matrix estimated as the average of `|ψ⟩⟨ψ|` over the
    /// trajectories (registers of at most
    /// [`MAX_DENSITY_QUBITS`] qubits).
    pub fn density_matrix(&self, circuit: &Circuit, theta: &[f64]) -> Result<DensityMatrix> {
        let n = circuit.n_qubits();
        if n > MAX_DENSITY_QUBITS {
            return Err(QsoError::TooManyQubits {
                what: "DensityMatrix",
                max: MAX_DENSITY_QUBITS,
                requested: n,
            });
        }
        let program = self.program(circuit, theta)?;
        let dim = 1usize << n;
        let mut data = vec![ZERO; dim * dim];
        let base = Rng::seed_from_u64(self.seed);
        for start in (0..self.trajectories).step_by(BATCH) {
            let count = BATCH.min(self.trajectories - start);
            let states = map_tasks(count, |j| {
                run_trajectory(&program, base.fork((start + j) as u64)).0
            });
            for state in states {
                let psi = state.amplitudes();
                for c in 0..dim {
                    let pc = psi[c].conj();
                    for r in 0..dim {
                        data[r + (c << n)] += psi[r] * pc;
                    }
                }
            }
        }
        let scale = 1.0 / self.trajectories as f64;
        for z in &mut data {
            *z *= scale;
        }
        DensityMatrix::from_data(n, data)
    }

    /// Final state of trajectory `k` (as used by the other methods).
    pub fn trajectory_state(
        &self,
        circuit: &Circuit,
        theta: &[f64],
        k: usize,
    ) -> Result<StateVector> {
        let program = self.program(circuit, theta)?;
        let base = Rng::seed_from_u64(self.seed);
        Ok(run_trajectory(&program, base.fork(k as u64)).0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::normal_quantile;
    use crate::noise::{ChannelSpec, GateRule, QubitProperties};

    /// Upper `p`-quantile of χ² with `df` degrees of freedom
    /// (Wilson–Hilferty).
    fn chi_square_quantile(df: f64, p: f64) -> f64 {
        let z = normal_quantile(p);
        df * (1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt()).powi(3)
    }

    fn device() -> DeviceModel {
        DeviceModel::new("mixed")
            .with_default_qubit(
                QubitProperties::new()
                    .with_t1_t2(40.0, 30.0)
                    .with_readout(0.03, 0.06),
            )
            .with_rule(
                GateRule::any()
                    .with_num_qubits(2)
                    .with_duration(1.0)
                    .with_depolarizing(0.02)
                    .with_channel(ChannelSpec::Pauli([("ZZ".to_string(), 0.01)].into()))
                    .with_channel(ChannelSpec::Rotation {
                        pauli: "ZZ".into(),
                        angle: 0.05,
                    }),
            )
            .with_rule(
                GateRule::any()
                    .with_duration(0.2)
                    .with_over_rotation(0.02)
                    .with_channel(ChannelSpec::AmplitudeDamping(0.01)),
            )
    }

    fn layered(n: usize, layers: usize) -> Circuit {
        let mut c = Circuit::new(n);
        for l in 0..layers {
            for q in 0..n {
                c.ry(q, 0.3 + 0.17 * (q + l) as f64).rz(q, 0.1 * q as f64);
            }
            for q in (l % 2..n - 1).step_by(2) {
                c.cx(q, q + 1);
            }
        }
        c
    }

    /// Exact outcome distribution including readout errors.
    fn noisy_distribution(rho: &DensityMatrix, readout: &[(f64, f64)]) -> Vec<f64> {
        let mut p = rho.probabilities();
        for (q, &(p01, p10)) in readout.iter().enumerate() {
            let bit = 1 << q;
            for b in 0..p.len() {
                if b & bit == 0 {
                    let (a, c) = (p[b], p[b | bit]);
                    p[b] = (1.0 - p01) * a + p10 * c;
                    p[b | bit] = p01 * a + (1.0 - p10) * c;
                }
            }
        }
        p
    }

    /// χ² statistic and degrees of freedom, merging bins with expected
    /// counts below 5.
    fn chi_square(counts: &Counts, expected: &[f64], shots: usize) -> (f64, f64) {
        let (mut stat, mut bins) = (0.0, 0usize);
        let (mut small_obs, mut small_exp) = (0.0, 0.0);
        for (b, &p) in expected.iter().enumerate() {
            let e = p * shots as f64;
            let o = *counts.get(&b).unwrap_or(&0) as f64;
            if e < 5.0 {
                small_obs += o;
                small_exp += e;
            } else {
                stat += (o - e).powi(2) / e;
                bins += 1;
            }
        }
        if small_exp > 0.0 {
            stat += (small_obs - small_exp).powi(2) / small_exp;
            bins += 1;
        }
        (stat, (bins - 1) as f64)
    }

    #[test]
    fn trajectory_samples_follow_the_density_matrix() {
        for (n, layers, shots) in [(3, 3, 20_000), (6, 3, 20_000), (10, 2, 20_000)] {
            let c = layered(n, layers);
            let dev = device();
            let mut rho = DensityMatrix::zero(n).unwrap();
            rho.apply_device(&c, &[], &dev).unwrap();
            let expected = noisy_distribution(&rho, &dev.readout_errors(n));
            let sim = TrajectorySimulator::new(dev, shots, 11 + n as u64).unwrap();
            let counts = sim.sample(&c, &[], shots).unwrap();
            assert_eq!(counts.values().sum::<usize>(), shots);
            let (stat, df) = chi_square(&counts, &expected, shots);
            let limit = chi_square_quantile(df, 0.999);
            assert!(
                stat < limit,
                "n={n}: χ² = {stat:.1} with {df} df, limit {limit:.1}"
            );
            // The test has power: the distribution without readout errors
            // is rejected.
            let wrong = noisy_distribution(&rho, &[]);
            let (stat, df) = chi_square(&counts, &wrong, shots);
            assert!(stat > chi_square_quantile(df, 0.999), "n={n}: {stat:.1}");
        }
    }

    #[test]
    fn averages_converge_to_density_matrix_values() {
        let n = 4;
        let c = layered(n, 3);
        let dev = device();
        let mut rho = DensityMatrix::zero(n).unwrap();
        rho.apply_device(&c, &[], &dev).unwrap();
        let observables: Vec<PauliSum> = ["Z0 Z1", "X2", "0.5 Y1 Y3 - Z3"]
            .iter()
            .map(|s| s.parse().unwrap())
            .collect();
        let sim = TrajectorySimulator::new(dev.clone(), 4000, 5).unwrap();
        let estimates = sim.expectations(&c, &[], &observables).unwrap();
        for (h, e) in observables.iter().zip(&estimates) {
            let exact = rho.expectation(h).unwrap();
            assert!(e.std_error > 0.0 && e.samples == 4000);
            assert!(
                (e.value - exact).abs() < 4.0 * e.std_error,
                "{} vs {exact} ± {}",
                e.value,
                e.std_error
            );
        }
        // The averaged |ψ⟩⟨ψ| approaches ρ entry by entry.
        let avg = sim.density_matrix(&c, &[]).unwrap();
        let diff = (avg.to_matrix() - rho.to_matrix())
            .iter()
            .map(|z| z.norm())
            .fold(0.0, f64::max);
        assert!(diff < 0.03, "{diff}");
        assert!((avg.trace() - 1.0).abs() < 1e-12);
        // Same seed, same numbers.
        let again = sim.expectations(&c, &[], &observables).unwrap();
        assert_eq!(estimates, again);
    }

    #[test]
    fn noiseless_trajectories_are_the_pure_state() {
        let c = layered(5, 2);
        let sim = TrajectorySimulator::new(DeviceModel::new("ideal"), 3, 1).unwrap();
        let psi = StateVector::from_circuit(&c, &[]).unwrap();
        for k in 0..3 {
            let s = sim.trajectory_state(&c, &[], k).unwrap();
            assert!((s.fidelity(&psi).unwrap() - 1.0).abs() < 1e-12);
        }
        let z: PauliSum = "Z0 Z4".parse().unwrap();
        let e = sim.expectation(&c, &[], &z).unwrap();
        assert!((e.value - z.expectation(&psi).unwrap()).abs() < 1e-12);
        assert!(e.std_error < 1e-12);
    }

    #[test]
    fn large_registers_run() {
        // 22 qubits: beyond any density matrix, still a few trajectories.
        let n = 22;
        let mut c = Circuit::new(n);
        c.h(0);
        for q in 0..n - 1 {
            c.cx(q, q + 1);
        }
        let dev = DeviceModel::new("ghz")
            .with_rule(GateRule::any().with_num_qubits(2).with_depolarizing(0.01));
        let sim = TrajectorySimulator::new(dev, 4, 3).unwrap();
        let zz: PauliSum = format!("Z0 Z{}", n - 1).parse().unwrap();
        let e = sim.expectation(&c, &[], &zz).unwrap();
        assert!(e.value.abs() <= 1.0 + 1e-12);
        assert!(sim.density_matrix(&c, &[]).is_err());
        assert_eq!(sim.sample(&c, &[], 10).unwrap().values().sum::<usize>(), 10);
    }
}
