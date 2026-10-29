//! Quantum trajectories (Monte-Carlo wave functions).
//!
//! A noisy circuit on n qubits needs a 4ⁿ-entry density matrix, but each
//! *trajectory* needs only a 2ⁿ-entry state vector: every channel
//! `ρ ↦ Σ_i K_i ρ K_i†` is replaced by applying one Kraus operator, chosen
//! with probability `‖K_i ψ‖²`, and renormalising. The average of `|ψ⟩⟨ψ|`
//! over trajectories is the density matrix, so averages of observables and
//! of measurement outcomes converge to the density-matrix results with the
//! usual `1/√N` statistical error.
//!
//! The simulation uses the waiting-time form of the method. Channels that
//! are mixtures of unitaries (Pauli, depolarising, dephasing and coherent
//! errors) are sampled up front, as their choice does not depend on the
//! state. Other channels (amplitude damping, relaxation) apply their
//! dominant Kraus operator without renormalising; the squared norm is then
//! the probability that no jump has happened, and a jump happens where it
//! falls below a random threshold. Stretches without jumps run as one
//! fused, cache-blocked circuit, and jumps are located by bisection. One
//! trajectory, compilation included, costs about three noiseless runs at
//! 16–20 qubits; many trajectories share one compilation and cost about
//! 1.2–1.3 noiseless runs each (see `docs/BENCHMARKS.md`).
//!
//! Every trajectory has its own random stream derived from the seed, so
//! results do not depend on the number of threads.
//!
//! Noise correlated in time (quasi-static detunings and over-rotations,
//! telegraph noise; see [`DeviceModel`]) is drawn per trajectory before it
//! runs: telegraph processes have exponential waiting times between jumps
//! and are integrated exactly over every gate and wait of their qubit.
//! Averages over trajectories are then exact in expectation, as for
//! channels; density matrices cannot represent such noise.
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

use super::device::{
    CompiledChannel, DeviceModel, Op, Program, SlowNoise, Spread, Telegraph, Unraveling,
};
use super::{DensityMatrix, MAX_DENSITY_QUBITS};
use crate::error::{QsoError, Result};
use crate::estimation::Estimate;
use crate::parallel::{map_tasks, sum_ranges};
use std::sync::Arc;

use crate::quantum::kernels::{apply_instruction, insert_zeros};
use crate::quantum::{Circuit, Counts, GateKind, Instruction, Param, PauliSum, StateVector, plan};
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

/// Sum of reduced-density-matrix contributions (up to 16×16).
struct Reduced(Vec<C64>);

impl std::iter::Sum for Reduced {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Reduced(Vec::new()), |mut a, b| {
            if a.0.is_empty() {
                return b;
            }
            for (x, y) in a.0.iter_mut().zip(b.0) {
                *x += y;
            }
            a
        })
    }
}

/// `ρ[a][b] = Σ ψ_a ψ_b*` over the other qubits, for one to four ordered
/// qubits (first = most significant local bit). Row-major `d × d`.
fn reduced_matrix(amps: &[C64], qubits: &[usize]) -> Reduced {
    if let [q] = *qubits {
        let stride = 1usize << q;
        return sum_ranges((amps.len() / 2) as u64, |start, end| {
            let (mut p0, mut p1, mut coh) = (0.0, 0.0, ZERO);
            for k in start as usize..end as usize {
                let i0 = ((k >> q) << (q + 1)) | (k & (stride - 1));
                let (a, b) = (amps[i0], amps[i0 | stride]);
                p0 += a.norm_sqr();
                p1 += b.norm_sqr();
                coh += a * b.conj();
            }
            Reduced(vec![C64::new(p0, 0.0), coh, coh.conj(), C64::new(p1, 0.0)])
        });
    }
    let k = qubits.len();
    let d = 1usize << k;
    let mut sorted = qubits.to_vec();
    sorted.sort_unstable();
    // Offsets of the local basis states; the first qubit is the most
    // significant bit of the local index.
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
        let mut acc = vec![ZERO; d * d];
        for kk in start..end {
            let base = insert_zeros(kk as usize, &sorted);
            let mut v = [ZERO; 16];
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

/// Kraus weights `‖K_i ψ‖²` (absolute, for unnormalised states).
fn kraus_weights(amps: &[C64], gram: &[Vec<C64>], qubits: &[usize]) -> Vec<f64> {
    let d = 1usize << qubits.len();
    let rho = reduced_matrix(amps, qubits).0;
    // ‖Kψ‖² = Σ_{r,c} (K†K)[r][c] ρ[c][r].
    gram.iter()
        .map(|g| {
            let mut p = ZERO;
            for r in 0..d {
                for c in 0..d {
                    p += g[r * d + c] * rho[c * d + r];
                }
            }
            p.re.max(0.0)
        })
        .collect()
}

/// A local operator as a (possibly non-unitary) instruction.
fn local_instruction(qubits: &[usize], matrix: &[C64]) -> Instruction {
    Instruction {
        kind: GateKind::Unitary,
        targets: qubits.to_vec(),
        controls: Vec::new(),
        params: Vec::new(),
        matrix: Some(Arc::new(matrix.to_vec())),
    }
}

/// One step of a trajectory.
#[derive(Debug)]
enum Step {
    /// An ideal gate (angles bound).
    Gate(Instruction),
    /// A unitary mixture: one instruction per component (`None`: identity);
    /// `slot` indexes the pre-sampled choices.
    Mixture {
        options: Vec<Option<Instruction>>,
        cumulative: Vec<f64>,
        slot: usize,
    },
    /// A general channel: its no-jump operator as an instruction.
    Jump {
        no_jump: Instruction,
        channel: usize,
        qubits: Vec<usize>,
    },
    /// The turn of `qubit` by its slow noise; `slot` indexes the phases
    /// sampled for the trajectory.
    Phase {
        qubit: usize,
        from: f64,
        to: f64,
        slot: usize,
    },
    /// A rotation whose angles a trajectory scales by `base + η[rule]`.
    SpreadGate {
        inst: Instruction,
        base: f64,
        rule: usize,
    },
}

/// The random choices of one trajectory made before it runs.
struct Sample {
    /// Component of every unitary mixture.
    choices: Vec<usize>,
    /// Turn of every phase step.
    phases: Vec<f64>,
    /// Over-rotation spread of every rule.
    etas: Vec<f64>,
}

/// A telegraph process along one trajectory, evaluated in time order.
struct TelegraphProcess {
    amplitude: f64,
    rate: f64,
    sign: f64,
    /// Time up to which the process is known, and of the next jump.
    time: f64,
    next: f64,
}

impl TelegraphProcess {
    fn new(t: &Telegraph, rng: &mut Rng) -> Self {
        let sign = if rng.uniform() < 0.5 { 1.0 } else { -1.0 };
        let mut p = Self {
            amplitude: t.amplitude,
            rate: t.rate,
            sign,
            time: 0.0,
            next: 0.0,
        };
        p.next = p.waiting_time(rng);
        p
    }

    /// Exponential waiting time with mean `1/rate` (infinite at rate 0).
    fn waiting_time(&self, rng: &mut Rng) -> f64 {
        if self.rate > 0.0 {
            -(1.0 - rng.uniform()).ln() / self.rate
        } else {
            f64::INFINITY
        }
    }

    /// `∫ν(t) dt` from `from` to `to` (`from ≥` the times asked before).
    fn integral(&mut self, from: f64, to: f64, rng: &mut Rng) -> f64 {
        // Jumps before `from` only change the sign.
        while self.next <= from {
            self.sign = -self.sign;
            self.next += self.waiting_time(rng);
        }
        self.time = from;
        let mut acc = 0.0;
        while self.next < to {
            acc += self.sign * (self.next - self.time);
            self.time = self.next;
            self.sign = -self.sign;
            self.next += self.waiting_time(rng);
        }
        acc += self.sign * (to - self.time);
        self.time = to;
        self.amplitude * acc
    }
}

/// A compiled program prepared for trajectories (shared by all of them).
///
/// Trajectories use the waiting-time form of the Monte-Carlo wave-function
/// method. Unitary mixtures are sampled up front, as they do not depend on
/// the state. General channels apply their dominant ("no-jump") Kraus
/// operator without renormalising, so the squared norm of the state is the
/// probability that no jump has happened yet; a jump happens where it
/// falls below a uniform random threshold. Runs of steps without a jump
/// therefore execute as one blocked, fused circuit, and a jump is located
/// by bisection.
#[derive(Debug)]
pub(crate) struct Skeleton<'p> {
    n_qubits: usize,
    steps: Vec<Step>,
    /// `jumps[i]`: number of jump steps before step `i`.
    jumps: Vec<usize>,
    channels: &'p [CompiledChannel],
    /// Slow noise of every qubit and over-rotation spread of every rule
    /// (empty for Markovian programs).
    slow: Vec<SlowNoise>,
    spreads: Vec<Option<Spread>>,
    phase_slots: usize,
}

/// Steps simulated one at a time when locating a jump.
const STEPWISE: usize = 8;

/// Steps in the first block after the start or a jump; blocks then double
/// (a galloping search), so locating a jump costs time proportional to its
/// distance from the previous one.
const FIRST_BLOCK: usize = 256;

impl<'p> Skeleton<'p> {
    /// The skeleton of a compiled program, with its slow noise.
    pub(crate) fn for_program(program: &'p Program) -> Self {
        let mut s = Self::new(program.n_qubits, &program.ops, &program.channels);
        s.slow = program.slow.clone();
        s.spreads = program.spreads.clone();
        s
    }

    pub(crate) fn new(n_qubits: usize, ops: &[Op], channels: &'p [CompiledChannel]) -> Self {
        let mut steps = Vec::new();
        let mut slot = 0;
        let mut phase_slots = 0;
        for op in ops {
            match op {
                Op::Gates(c) => steps.extend(c.instructions().iter().cloned().map(Step::Gate)),
                Op::Channel { channel, qubits } => match &channels[*channel].unraveling {
                    Unraveling::Mixture {
                        cumulative,
                        unitaries,
                    } => {
                        steps.push(Step::Mixture {
                            options: unitaries
                                .iter()
                                .map(|u| u.as_ref().map(|u| local_instruction(qubits, u)))
                                .collect(),
                            cumulative: cumulative.clone(),
                            slot,
                        });
                        slot += 1;
                    }
                    Unraveling::General { kraus, no_jump, .. } => steps.push(Step::Jump {
                        no_jump: local_instruction(qubits, &kraus[*no_jump]),
                        channel: *channel,
                        qubits: qubits.clone(),
                    }),
                },
                Op::Phase { qubit, from, to } => {
                    steps.push(Step::Phase {
                        qubit: *qubit,
                        from: *from,
                        to: *to,
                        slot: phase_slots,
                    });
                    phase_slots += 1;
                }
                Op::SpreadGate { inst, base, rule } => steps.push(Step::SpreadGate {
                    inst: inst.clone(),
                    base: *base,
                    rule: *rule,
                }),
            }
        }
        let mut jumps = Vec::with_capacity(steps.len() + 1);
        let mut count = 0;
        jumps.push(0);
        for step in &steps {
            count += usize::from(matches!(step, Step::Jump { .. }));
            jumps.push(count);
        }
        Self {
            n_qubits,
            steps,
            jumps,
            channels,
            slow: Vec::new(),
            spreads: Vec::new(),
            phase_slots,
        }
    }

    /// Draw the slow noise of one trajectory: the over-rotation spread of
    /// every rule, the detuning of every qubit, and the telegraph processes,
    /// integrated over the phase steps in time order. Nothing is drawn for
    /// Markovian programs.
    pub(crate) fn slow_sample(&self, rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
        let spread = self
            .steps
            .iter()
            .any(|s| matches!(s, Step::SpreadGate { .. }));
        let etas = if spread {
            self.spreads
                .iter()
                .map(|s| s.as_ref().map_or(0.0, |s| s.sample(rng)))
                .collect()
        } else {
            Vec::new()
        };
        if self.phase_slots == 0 {
            return (Vec::new(), etas);
        }
        let detunings: Vec<f64> = self
            .slow
            .iter()
            .map(|s| s.detuning.as_ref().map_or(0.0, |d| d.sample(rng)))
            .collect();
        let mut telegraph: Vec<Option<TelegraphProcess>> = self
            .slow
            .iter()
            .map(|s| s.telegraph.as_ref().map(|t| TelegraphProcess::new(t, rng)))
            .collect();
        let mut phases = vec![0.0; self.phase_slots];
        for step in &self.steps {
            if let Step::Phase {
                qubit,
                from,
                to,
                slot,
            } = step
            {
                let mut phi = detunings[*qubit] * (to - from);
                if let Some(p) = &mut telegraph[*qubit] {
                    phi += p.integral(*from, *to, rng);
                }
                phases[*slot] = phi;
            }
        }
        (phases, etas)
    }

    /// Whether steps `a..b` contain a jump step.
    fn has_jump(&self, a: usize, b: usize) -> bool {
        self.jumps[b] > self.jumps[a]
    }

    /// Apply steps `a..b` without jumps as one blocked circuit.
    fn apply_block(&self, amps: &mut [C64], a: usize, b: usize, sample: &Sample) {
        let mut c = Circuit::new(self.n_qubits);
        for step in &self.steps[a..b] {
            match step {
                Step::Gate(inst) => c.push_trusted(inst.clone()),
                Step::Mixture { options, slot, .. } => {
                    if let Some(inst) = &options[sample.choices[*slot]] {
                        c.push_trusted(inst.clone());
                    }
                }
                Step::Jump { no_jump, .. } => c.push_trusted(no_jump.clone()),
                Step::Phase { qubit, slot, .. } => {
                    let phi = sample.phases[*slot];
                    if phi != 0.0 {
                        c.push_trusted(Instruction {
                            kind: GateKind::RZ,
                            targets: vec![*qubit],
                            controls: Vec::new(),
                            params: vec![Param::Value(phi)],
                            matrix: None,
                        });
                    }
                }
                Step::SpreadGate { inst, base, rule } => {
                    let factor = base + sample.etas[*rule];
                    let mut scaled = inst.clone();
                    for p in &mut scaled.params {
                        if let Param::Value(v) = p {
                            *v *= factor;
                        }
                    }
                    c.push_trusted(scaled);
                }
            }
        }
        if !c.is_empty() {
            plan::run(amps, &c, &[]);
        }
    }

    /// The general channel of a jump step.
    fn general(&self, channel: usize) -> (&[Vec<C64>], &[Vec<C64>], usize) {
        match &self.channels[channel].unraveling {
            Unraveling::General {
                kraus,
                gram,
                no_jump,
            } => (kraus, gram, *no_jump),
            Unraveling::Mixture { .. } => unreachable!("jump steps are general channels"),
        }
    }

    /// From `state` at step `a` (squared norm at least `threshold`, and below
    /// it after step `b`), find the jump: returns its step with `state` just
    /// before it, or (when only rounding pushed the norm down) `Err(b)` with
    /// `state` after step `b`.
    fn locate(
        &self,
        state: &mut StateVector,
        mut a: usize,
        mut b: usize,
        threshold: f64,
        sample: &Sample,
    ) -> std::result::Result<usize, usize> {
        while b - a > STEPWISE {
            let mid = a + (b - a) / 2;
            let checkpoint = state.clone();
            self.apply_block(state.amplitudes_mut(), a, mid, sample);
            if state.norm().powi(2) >= threshold {
                a = mid;
            } else {
                *state = checkpoint;
                b = mid;
            }
        }
        for j in a..b {
            if let Step::Jump {
                no_jump,
                channel,
                qubits,
            } = &self.steps[j]
            {
                let (_, gram, k0) = self.general(*channel);
                let stay = kraus_weights(state.amplitudes(), &gram[k0..=k0], qubits)[0];
                if stay < threshold {
                    return Ok(j);
                }
                apply_instruction(state.amplitudes_mut(), no_jump, &[], false, 0);
            } else {
                self.apply_block(state.amplitudes_mut(), j, j + 1, sample);
            }
        }
        Err(b)
    }

    /// Apply the jump of step `j`: one of the operators other than the
    /// no-jump one, chosen with the Born weights; the state is normalised.
    fn jump(&self, state: &mut StateVector, j: usize, rng: &mut Rng) {
        let Step::Jump {
            channel, qubits, ..
        } = &self.steps[j]
        else {
            unreachable!("jumps happen at jump steps")
        };
        let (kraus, gram, no_jump) = self.general(*channel);
        let weights = kraus_weights(state.amplitudes(), gram, qubits);
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut acc = 0.0;
        for (i, w) in weights.iter().enumerate() {
            if i != no_jump {
                acc += w;
            }
            cumulative.push(acc);
        }
        // Without weight on the jumps (only through rounding) keep the
        // no-jump branch.
        let i = if acc > 0.0 {
            sample_cumulative(&cumulative, rng)
        } else {
            no_jump
        };
        let scale = 1.0 / weights[i].sqrt();
        let m: Vec<C64> = kraus[i].iter().map(|z| z * scale).collect();
        apply_instruction(
            state.amplitudes_mut(),
            &local_instruction(qubits, &m),
            &[],
            false,
            0,
        );
    }

    /// Run one trajectory with the random stream `rng`; returns the
    /// normalised final state and the stream.
    pub(crate) fn run(&self, mut rng: Rng) -> (StateVector, Rng) {
        let choices: Vec<usize> = self
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Mixture { cumulative, .. } => Some(sample_cumulative(cumulative, &mut rng)),
                _ => None,
            })
            .collect();
        let (phases, etas) = self.slow_sample(&mut rng);
        let sample = Sample {
            choices,
            phases,
            etas,
        };
        // Threshold in (0, 1] for the squared norm.
        let threshold = 1.0 - rng.uniform();
        let state = self.run_from(&sample, threshold, &mut rng);
        (state, rng)
    }

    /// The trajectory for given choices and first threshold.
    fn run_from(&self, sample: &Sample, mut threshold: f64, rng: &mut Rng) -> StateVector {
        let mut state = StateVector::zero(self.n_qubits);
        let end = self.steps.len();
        let mut start = 0;
        let mut size = FIRST_BLOCK;
        while start < end {
            if !self.has_jump(start, end) {
                self.apply_block(state.amplitudes_mut(), start, end, sample);
                break;
            }
            let stop = (start + size).min(end);
            let checkpoint = state.clone();
            self.apply_block(state.amplitudes_mut(), start, stop, sample);
            if state.norm().powi(2) >= threshold {
                start = stop;
                size *= 2;
                continue;
            }
            state = checkpoint;
            match self.locate(&mut state, start, stop, threshold, sample) {
                Ok(j) => {
                    self.jump(&mut state, j, rng);
                    threshold = 1.0 - rng.uniform();
                    start = j + 1;
                    size = FIRST_BLOCK;
                }
                Err(b) => {
                    // Rounding pushed the norm below the threshold without
                    // a jump: continue from there with the threshold lowered.
                    threshold = threshold.min(state.norm().powi(2));
                    start = b;
                }
            }
        }
        let norm = state.norm();
        for z in state.amplitudes_mut() {
            *z /= norm;
        }
        state
    }
}

/// Run compiled operations on `|0…0⟩` as one trajectory; returns the final
/// state and the trajectory's random stream.
pub(crate) fn run_ops(
    n_qubits: usize,
    ops: &[Op],
    channels: &[CompiledChannel],
    rng: Rng,
) -> (StateVector, Rng) {
    Skeleton::new(n_qubits, ops, channels).run(rng)
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
        let skeleton = Skeleton::for_program(program);
        let one = |k: usize| {
            let (state, rng) = skeleton.run(base.fork(k as u64));
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
        let skeleton = Skeleton::for_program(&program);
        for start in (0..self.trajectories).step_by(BATCH) {
            let count = BATCH.min(self.trajectories - start);
            let states = map_tasks(count, |j| skeleton.run(base.fork((start + j) as u64)).0);
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
        if k >= self.trajectories {
            return Err(QsoError::invalid(
                "k",
                format!("trajectory {k} of {}", self.trajectories),
            ));
        }
        let program = self.program(circuit, theta)?;
        let base = Rng::seed_from_u64(self.seed);
        let skeleton = Skeleton::for_program(&program);
        Ok(skeleton.run(base.fork(k as u64)).0)
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
    fn rounding_below_the_threshold_finishes_the_circuit() {
        // A custom gate that is unitary within tolerance but shrinks the
        // norm by 4e-10, and a threshold just above the result: the norm
        // drops below it at a gate, not at a jump step (the damping acts
        // on a qubit in |0⟩ and loses nothing). The trajectory must still
        // run every gate.
        let mut c = Circuit::new(3);
        c.x(0);
        let shrink = vec![C64::new(1.0, 0.0), ZERO, ZERO, C64::new(1.0 - 4e-10, 0.0)];
        c.unitary(shrink, &[0], &[]).unwrap();
        for _ in 0..41 {
            c.x(1);
        }
        c.s(2).h(0);
        let dev = DeviceModel::new("damping")
            .with_rule(GateRule::for_gate("s").with_channel(ChannelSpec::AmplitudeDamping(1e-3)));
        let program = Program::compile(&c, &[], &dev).unwrap();
        let skeleton = Skeleton::new(3, &program.ops, &program.channels);
        let sample = Sample {
            choices: Vec::new(),
            phases: Vec::new(),
            etas: Vec::new(),
        };
        let state = skeleton.run_from(&sample, 1.0 - 5e-10, &mut Rng::seed_from_u64(0));
        let psi = StateVector::from_circuit(&c, &[]).unwrap();
        assert!((state.fidelity(&psi).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn many_jumps_per_trajectory() {
        // Strong relaxation and damping: several jumps per trajectory, so
        // the bisection that locates them is exercised throughout.
        let dev = DeviceModel::new("strong")
            .with_default_qubit(QubitProperties::new().with_t1_t2(4.0, 3.0))
            .with_rule(
                GateRule::any()
                    .with_duration(0.5)
                    .with_channel(ChannelSpec::AmplitudeDamping(0.08))
                    .with_channel(ChannelSpec::Rotation {
                        pauli: "X".into(),
                        angle: 0.1,
                    }),
            );
        let n = 4;
        let c = layered(n, 6);
        let mut rho = DensityMatrix::zero(n).unwrap();
        rho.apply_device(&c, &[], &dev).unwrap();
        let expected = noisy_distribution(&rho, &[]);
        let shots = 20_000;
        let sim = TrajectorySimulator::new(dev, shots, 21).unwrap();
        let counts = sim.sample(&c, &[], shots).unwrap();
        let (stat, df) = chi_square(&counts, &expected, shots);
        assert!(
            stat < chi_square_quantile(df, 0.999),
            "χ² = {stat:.1} with {df} df"
        );
        assert!(
            rho.purity() < 0.5,
            "the noise is strong: purity {}",
            rho.purity()
        );
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
