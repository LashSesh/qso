//! Expectation values of large noisy circuits by Pauli propagation.
//!
//! An observable `O = Σ c_s P_s` is carried backwards through a circuit in
//! the Heisenberg picture, `O ↦ G† O G` for a gate and `O ↦ E†(O)` for a
//! channel, as a table of Pauli strings on any number of qubits (two bit
//! masks of `⌈n/64⌉` words) with real coefficients, as in sparse Pauli
//! dynamics (Begušić, Gray and Chan, Sci. Adv. 10, eadk4321 (2024)). The
//! value `Tr(ρ O(0))` is then read in `|0…0⟩` or in a product state.
//!
//! * Clifford gates map a string to one string with the signed rules of
//!   the tableau ([`stabilizer`](crate::stabilizer)).
//! * A rotation `exp(−iθG/2)` maps a string `P` that anticommutes with `G`
//!   to `cos θ·P + sin θ·(iGP)`; other gates on one to four qubits act by
//!   their Pauli transfer matrices `T_ab = 2^{−k} Re Tr(P_a U† P_b U)`.
//! * Channels on one to four qubits act by the transfer matrices of their
//!   adjoints `T_ab = 2^{−k} Re Tr(E(P_a) P_b)` (amplitude damping maps `Z`
//!   to `(1 − γ)Z + γI`); a sparse Pauli–Lindblad layer with rates `λ_k`
//!   scaled by `α` multiplies a string by `exp(−2α Σ λ_k)` over the
//!   generators that anticommute with it.
//!
//! Entries of a table with magnitude at most `10⁻¹⁴` are dropped and
//! counted in the truncation bound, so rotations at multiples of π/2 act
//! as signed permutations.
//!
//! Without truncation the value is exact up to rounding; holding more
//! than [`PropagationOptions::max_terms`] strings is an error. With
//! [`PropagationOptions::min_coefficient`] or
//! [`PropagationOptions::max_weight`], strings are dropped and
//! [`PropagationResult::truncation_bound`] bounds the error rigorously:
//! adjoint channels and unitaries do not increase the operator norm, so a
//! dropped `c·P` moves the value by at most `|c|`, times `e^{2|α|Σλ_k}` for
//! every Pauli–Lindblad layer with `α < 0` between it and the start. The
//! bound adds magnitudes and can lie far above the actual error.
//!
//! Device models run through [`device_layers`]; on this path, and only
//! here, they may hold up to [`MAX_PROPAGATION_QUBITS`] per-qubit entries
//! ([`device_from_json`]).
//!
//! # Example
//!
//! A GHZ state of 100 qubits, then sparse Pauli–Lindblad noise on its last
//! qubit, amplified (`α = 2`) and inverted again (`α = −2`):
//!
//! ```
//! use metatron_qso::propagation::{Layer, PauliOperator, PropagationOptions, propagate};
//! use metatron_qso::quantum::Circuit;
//!
//! let n = 100;
//! let mut ghz = Circuit::new(n);
//! ghz.h(0);
//! for k in 0..n - 1 {
//!     ghz.cx(k, k + 1);
//! }
//! let rates = PauliOperator::parse(n, "0.01*X99 + 0.02*Y99")?;
//! let obs = PauliOperator::parse(n, "Z0 Z99")?;
//! let opts = PropagationOptions::default();
//! let noisy = [Layer::gates(ghz.clone()), Layer::pauli_lindblad(rates.clone(), 2.0)];
//! let r = propagate(&noisy, &obs, None, &opts)?;
//! assert!((r.value - (-0.12f64).exp()).abs() < 1e-15); // X and Y flip Z99
//! let undone = [
//!     Layer::gates(ghz),
//!     Layer::pauli_lindblad(rates.clone(), 2.0),
//!     Layer::pauli_lindblad(rates, -2.0),
//! ];
//! assert!((propagate(&undone, &obs, None, &opts)?.value - 1.0).abs() < 1e-15);
//! # Ok::<(), metatron_qso::QsoError>(())
//! ```

mod compile;
mod operator;
mod table;
mod transfer;

#[cfg(test)]
mod tests;

pub use operator::PauliOperator;

use serde::{Deserialize, Serialize};

use crate::error::{QsoError, Result};
use crate::estimation::Estimate;
use crate::noise::device::{Op as DeviceOp, Program as DeviceProgram};
use crate::noise::{Channel, DeviceModel};
use crate::quantum::{Circuit, Operation, Program};

/// Largest register of Pauli propagation (as
/// [`MAX_STABILIZER_QUBITS`](crate::stabilizer::MAX_STABILIZER_QUBITS)).
pub const MAX_PROPAGATION_QUBITS: usize = 1 << 14;

/// Hard ceiling of [`PropagationOptions::max_terms`].
pub const MAX_PROPAGATION_TERMS: usize = 1 << 30;

/// Default of [`PropagationOptions::max_terms`].
pub const DEFAULT_MAX_TERMS: usize = 1 << 24;

/// Most qubits of a gate or channel (as
/// [`MAX_CHANNEL_QUBITS`](crate::noise::MAX_CHANNEL_QUBITS)): its transfer
/// matrix has `4^k × 4^k` entries.
pub const MAX_TRANSFER_QUBITS: usize = 4;

/// One step of a noisy circuit, in time order (layers are propagated last
/// to first).
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum Layer {
    /// Ideal gates on register qubits `0..circuit.n_qubits()`, parameters
    /// bound to `theta`.
    Gates {
        /// The gates; each acts on at most [`MAX_TRANSFER_QUBITS`] qubits
        /// (controls included).
        circuit: Circuit,
        /// The circuit's parameters.
        theta: Vec<f64>,
    },
    /// A channel on one to four register qubits.
    Channel {
        /// The channel.
        channel: Channel,
        /// The register qubits it acts on, in the order of its Kraus
        /// operators (the first one most significant).
        qubits: Vec<usize>,
    },
    /// The sparse Pauli–Lindblad map `ρ ↦ Π_k (w_k ρ + (1 − w_k) P_k ρ P_k)`,
    /// `w_k = (1 + e^{−2αλ_k})/2`, with rates `Σ λ_k P_k` on the register,
    /// every rate times `factor = α`: `α ≥ 0` amplifies (probabilistic
    /// error amplification), `α = −1` after the device's own noise gives
    /// the value probabilistic error cancellation converges to.
    PauliLindblad {
        /// The rates `λ_k ≥ 0` as the coefficients of their generators
        /// (no identity term).
        rates: PauliOperator,
        /// The factor `α` of every rate (finite).
        factor: f64,
    },
}

impl Layer {
    /// Ideal gates without parameters.
    pub fn gates(circuit: Circuit) -> Self {
        Layer::Gates {
            circuit,
            theta: Vec::new(),
        }
    }

    /// `channel` on `qubits`.
    pub fn channel(channel: Channel, qubits: &[usize]) -> Self {
        Layer::Channel {
            channel,
            qubits: qubits.to_vec(),
        }
    }

    /// The Pauli–Lindblad layer with `rates` scaled by `factor`.
    pub fn pauli_lindblad(rates: PauliOperator, factor: f64) -> Self {
        Layer::PauliLindblad { rates, factor }
    }
}

/// Truncation and limits of [`propagate`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PropagationOptions {
    /// Strings with `|c|` below this are dropped and counted in the
    /// truncation bound. Finite and at least 0; default 0 (exact).
    pub min_coefficient: f64,
    /// Strings with more non-identity factors are dropped and counted.
    /// Default `None` (no limit).
    pub max_weight: Option<usize>,
    /// Most strings held at once (the table and, in a splitting level, the
    /// images of the batch of strings being expanded, before equal ones
    /// are merged); more is an error, not a truncation.
    /// In `1..=`[`MAX_PROPAGATION_TERMS`]; default [`DEFAULT_MAX_TERMS`].
    pub max_terms: usize,
    /// Return the Heisenberg observable at the start of the circuit.
    /// Default `false`.
    pub keep_operator: bool,
}

impl Default for PropagationOptions {
    fn default() -> Self {
        PropagationOptions {
            min_coefficient: 0.0,
            max_weight: None,
            max_terms: DEFAULT_MAX_TERMS,
            keep_operator: false,
        }
    }
}

impl PropagationOptions {
    /// These options with [`min_coefficient`](Self::min_coefficient) `c`.
    pub fn with_min_coefficient(mut self, c: f64) -> Self {
        self.min_coefficient = c;
        self
    }

    /// These options with [`max_weight`](Self::max_weight) `Some(w)`.
    pub fn with_max_weight(mut self, w: usize) -> Self {
        self.max_weight = Some(w);
        self
    }

    /// These options with [`max_terms`](Self::max_terms) `n`.
    pub fn with_max_terms(mut self, n: usize) -> Self {
        self.max_terms = n;
        self
    }

    /// These options with [`keep_operator`](Self::keep_operator) `keep`.
    pub fn with_operator(mut self, keep: bool) -> Self {
        self.keep_operator = keep;
        self
    }

    /// Check the ranges documented on the fields.
    pub(crate) fn validate(&self) -> Result<()> {
        if !(self.min_coefficient.is_finite() && self.min_coefficient >= 0.0) {
            return Err(QsoError::invalid(
                "min_coefficient",
                "the smallest coefficient kept must be finite and at least 0",
            ));
        }
        if !(1..=MAX_PROPAGATION_TERMS).contains(&self.max_terms) {
            return Err(QsoError::invalid(
                "max_terms",
                format!("the most strings held must lie in 1..={MAX_PROPAGATION_TERMS}"),
            ));
        }
        Ok(())
    }
}

/// The value of [`propagate`] and how it was reached.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct PropagationResult {
    /// The expectation value.
    pub value: f64,
    /// Rigorous bound on `|exact − value|` in every product (or mixed)
    /// input state, up to floating-point rounding: the dropped and
    /// snapped coefficients, each times the overhead `e^{2|α|Σλ_k}` of the
    /// Pauli–Lindblad layers with `α < 0` between it and the start.
    pub truncation_bound: f64,
    /// Strings at the start of the circuit.
    pub terms: usize,
    /// Most strings held in one table.
    pub peak_terms: usize,
    /// Strings dropped by `min_coefficient`, `max_weight` or cancellation.
    pub dropped_terms: usize,
    /// The Heisenberg observable at the start of the circuit (register
    /// labels), with [`PropagationOptions::keep_operator`].
    pub operator: Option<PauliOperator>,
}

impl PropagationResult {
    /// The value as an exact [`Estimate`] (zero standard error; the
    /// truncation bound is not a standard error).
    pub fn estimate(&self) -> Estimate {
        Estimate::exact(self.value)
    }
}

/// Check the Bloch vectors of a product state on `n` qubits.
fn check_state(state: &[[f64; 3]], n: usize) -> Result<()> {
    if state.len() != n {
        return Err(QsoError::DimensionMismatch {
            expected: n,
            actual: state.len(),
        });
    }
    for (q, r) in state.iter().enumerate() {
        let norm2: f64 = r.iter().map(|x| x * x).sum();
        // A unit vector written in decimals may exceed 1 by rounding.
        if !(r.iter().all(|x| x.is_finite()) && norm2 <= 1.0 + 1e-12) {
            return Err(QsoError::invalid(
                "state",
                format!("the Bloch vector of qubit {q} must be finite with length at most 1"),
            ));
        }
    }
    Ok(())
}

/// `⟨observable⟩` after `layers` on `|0…0⟩` (`state = None`) or on the
/// product state with Bloch vectors `state[q] = (⟨X⟩, ⟨Y⟩, ⟨Z⟩)` of
/// qubit `q` (`|r| ≤ 1`, mixed allowed).
///
/// The observable is carried from the last layer to the first. Ops are
/// scheduled in levels of ops on disjoint qubits, cut to the light cone of
/// the observable (from the letters each qubit can carry, counting the
/// images of table entries snapped into the bound too: an op that
/// leaves every string it can meet unchanged is removed, a Pauli–Lindblad
/// generator is cut to the factors some possible letter anticommutes
/// with), and applied level by level:
/// runs of Clifford gates, Pauli channels and Pauli–Lindblad layers
/// rewrite the table in place, any other level splits every string at
/// once and merges equal images. Strings are dropped by
/// [`PropagationOptions::min_coefficient`] and
/// [`PropagationOptions::max_weight`] at the start and before every
/// splitting level; [`PropagationResult::truncation_bound`] bounds the
/// error this makes. The result is bit-identical for every number of
/// threads. Overflow is an error, never an infinite or NaN result.
///
/// # Errors
///
/// [`QsoError::DimensionMismatch`] for a layer wider than the
/// observable's register, a channel on the wrong number of qubits, rates
/// on another register or a state of the wrong length;
/// [`QsoError::QubitOutOfRange`], [`QsoError::DuplicateQubit`] and
/// [`QsoError::ParameterCount`] for bad layers; [`QsoError::InvalidArgument`]
/// for a gate on more than [`MAX_TRANSFER_QUBITS`] qubits, invalid rates,
/// factors, Bloch vectors or options, an amplification `Γ` beyond floating
/// point (`"factor"`), an observable whose 1-norm times `Γ` lies beyond
/// floating point (`"factor"`, or `"observable"` without inverted layers),
/// coefficients, a value or a bound that overflow on the way
/// (`"observable"`), or more than [`PropagationOptions::max_terms`]
/// strings held (`"max_terms"`).
pub fn propagate(
    layers: &[Layer],
    observable: &PauliOperator,
    state: Option<&[[f64; 3]]>,
    options: &PropagationOptions,
) -> Result<PropagationResult> {
    options.validate()?;
    let n = observable.n_qubits();
    if let Some(state) = state {
        check_state(state, n)?;
    }
    let mut cache = transfer::TableCache::default();
    let ops = compile::compile_layers(layers, n, &mut cache)?;
    let plan = compile::plan(ops, observable)?;
    let bloch: Option<Vec<[f64; 3]>> =
        state.map(|state| plan.labels.iter().map(|&q| state[q]).collect());
    let out = table::run(&plan, bloch.as_deref(), options)?;
    let terms = out.coefs.iter().filter(|&&c| c != 0.0).count();
    let operator = if options.keep_operator {
        Some(register_operator(&plan, n, &out.keys, &out.coefs)?)
    } else {
        None
    };
    Ok(PropagationResult {
        // An empty sum is +0, not −0.
        value: out.value + 0.0,
        truncation_bound: out.bound,
        terms,
        peak_terms: out.peak,
        dropped_terms: out.dropped,
        operator,
    })
}

/// The table on compact labels as an operator on the register.
fn register_operator(
    plan: &compile::Plan,
    n: usize,
    keys: &[u64],
    coefs: &[f64],
) -> Result<PauliOperator> {
    let (w, wr) = (plan.words, n.div_ceil(64));
    let mut out = vec![0u64; coefs.len() * 2 * wr];
    for (key, reg) in keys.chunks_exact(2 * w).zip(out.chunks_exact_mut(2 * wr)) {
        for word in 0..w {
            let mut bits = key[word] | key[w + word];
            while bits != 0 {
                let b = bits.trailing_zeros();
                bits &= bits - 1;
                let q = plan.labels[64 * word + b as usize];
                reg[q / 64] |= ((key[word] >> b) & 1) << (q % 64);
                reg[wr + q / 64] |= ((key[w + word] >> b) & 1) << (q % 64);
            }
        }
    }
    PauliOperator::from_rows(n, out, coefs.to_vec())
}

/// The layers of `circuit(theta)` run on a Markovian `device`: ideal gate
/// segments and the device's channels (gate rules, idle relaxation with
/// `T1`/`T2`, spectators, fixed over-rotations), as
/// [`DensityMatrix::apply_device`](crate::noise::DensityMatrix::apply_device)
/// applies them, with up to [`MAX_PROPAGATION_QUBITS`] per-qubit entries
/// (the simulators take 34).
///
/// Readout errors are not applied, as in
/// [`DensityMatrix::expectation`](crate::noise::DensityMatrix::expectation).
/// Each call compiles one circuit: the device noise of a circuit in
/// several pieces is that of the pieces run one after another.
///
/// # Errors
///
/// [`QsoError::InvalidArgument`] for an invalid device or one with noise
/// correlated in time (quasi-static or telegraph detunings, spreads of
/// over-rotations: no product of channels), and the errors of compiling
/// the circuit against the device ([`QsoError::ParameterCount`],
/// [`QsoError::TooManyQubits`] for a circuit wider than the device).
pub fn device_layers(circuit: &Circuit, theta: &[f64], device: &DeviceModel) -> Result<Vec<Layer>> {
    device.validate_entries(MAX_PROPAGATION_QUBITS)?;
    if !device.is_markovian() {
        return Err(QsoError::invalid(
            "device",
            "slow noise (quasi-static or telegraph detunings, over-rotation spreads) is \
             correlated in time and no product of channels",
        ));
    }
    let program = DeviceProgram::compile(circuit, theta, device)?;
    program
        .ops
        .iter()
        .map(|op| match op {
            DeviceOp::Gates(c) => Ok(Layer::gates(c.clone())),
            DeviceOp::Channel { channel, qubits } => Ok(Layer::Channel {
                channel: program.channels[*channel].channel.clone(),
                qubits: qubits.clone(),
            }),
            DeviceOp::Phase { .. } | DeviceOp::SpreadGate { .. } => Err(QsoError::invalid(
                "device",
                "slow noise is correlated in time and no product of channels",
            )),
        })
        .collect()
}

/// The gates of `program` (such as OpenQASM text read by
/// [`Program::from_qasm`]) for [`propagate`]. Measurements are allowed at
/// the end only, with no later gate on a measured qubit, and are left out:
/// the value is then that of the state before them (the same for
/// observables made of `Z` and `I` on the measured qubits).
///
/// # Errors
///
/// [`QsoError::InvalidArgument`] (`"program"`) for a measurement followed
/// by a gate on its qubit, a reset or a classical condition, which no
/// circuit of gates describes.
pub fn program_circuit(program: &Program) -> Result<Circuit> {
    let mut circuit = Circuit::new(program.n_qubits());
    let mut measured = vec![false; program.n_qubits()];
    for op in program.operations() {
        match op {
            Operation::Gate(inst) => {
                if let Some(q) = inst.qubits().find(|&q| measured[q]) {
                    return Err(QsoError::invalid(
                        "program",
                        format!(
                            "qubit {q} is measured before a later {}; Pauli propagation takes \
                             gates followed by final measurements only",
                            inst.name()
                        ),
                    ));
                }
                circuit.try_push(inst.clone())?;
            }
            Operation::Measure { qubit, .. } => measured[*qubit] = true,
            Operation::Reset { .. } | Operation::If { .. } => {
                return Err(QsoError::invalid(
                    "program",
                    "has resets or classical conditions; Pauli propagation takes gates \
                     followed by final measurements only",
                ));
            }
        }
    }
    Ok(circuit)
}

/// [`DeviceModel::from_json`] with the per-qubit entry limit of this path
/// ([`MAX_PROPAGATION_QUBITS`] instead of the simulators' 34).
///
/// # Errors
///
/// [`QsoError::Parse`] for malformed JSON and [`QsoError::InvalidArgument`]
/// for an invalid description.
pub fn device_from_json(text: &str) -> Result<DeviceModel> {
    let device: DeviceModel = serde_json::from_str(text)
        .map_err(|e| QsoError::Parse(format!("device description: {e}")))?;
    device.validate_entries(MAX_PROPAGATION_QUBITS)?;
    Ok(device)
}
