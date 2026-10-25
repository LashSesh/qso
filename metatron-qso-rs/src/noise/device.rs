//! Device-like noise models.
//!
//! A [`DeviceModel`] describes a device the way calibration data does:
//! per-qubit relaxation times and readout errors, and per-gate durations
//! and errors. Gates are matched against an ordered list of [`GateRule`]s
//! (the first matching rule applies), so a description can give a default
//! for all two-qubit gates and override it for one pair:
//!
//! ```
//! use metatron_qso::noise::{ChannelSpec, DeviceModel, GateRule, QubitProperties};
//!
//! let device = DeviceModel::new("toy")
//!     .with_default_qubit(QubitProperties::new().with_t1_t2(100.0, 80.0).with_readout(0.02, 0.04))
//!     .with_rule(GateRule::for_gate("cx").on_qubits(&[0, 1]).with_depolarizing(0.02).with_duration(0.3))
//!     .with_rule(GateRule::any().with_num_qubits(2).with_depolarizing(0.01).with_duration(0.3)
//!         .with_channel(ChannelSpec::Rotation { pauli: "ZZ".into(), angle: 0.01 }))
//!     .with_rule(GateRule::any().with_num_qubits(1).with_depolarizing(0.001).with_duration(0.035));
//! let json = device.to_json();
//! assert_eq!(DeviceModel::from_json(&json).unwrap(), device);
//! ```
//!
//! Compiling a circuit against a device (done by the density-matrix and
//! trajectory simulators) schedules every gate as early as its qubits allow.
//! A qubit that waits for others relaxes with its T₁ and T₂ for the waiting
//! time, and every qubit of a gate relaxes for the gate's duration. After
//! each gate come, in this order, the rule's depolarising error, its further
//! channels and the relaxation during the gate. Durations, T₁ and T₂ share
//! one unit of time, whichever the description uses.

use std::collections::{BTreeMap, HashMap};

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::{Channel, NoiseModel, check_probability};
use crate::error::{QsoError, Result};
use crate::quantum::{Circuit, GateKind, Instruction, Param};

/// A noise channel in a device description, in a form that reads well as
/// JSON (`{"depolarizing": 0.01}`, `{"pauli": {"ZZ": 0.02}}`,
/// `{"rotation": {"pauli": "Z", "angle": 0.01}}`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum ChannelSpec {
    /// [`Channel::depolarizing`].
    Depolarizing(f64),
    /// [`Channel::depolarizing_2q`].
    Depolarizing2q(f64),
    /// [`Channel::amplitude_damping`].
    AmplitudeDamping(f64),
    /// [`Channel::phase_damping`].
    PhaseDamping(f64),
    /// [`Channel::bit_flip`].
    BitFlip(f64),
    /// [`Channel::phase_flip`].
    PhaseFlip(f64),
    /// [`Channel::pauli`]: probabilities of non-identity Pauli strings.
    Pauli(BTreeMap<String, f64>),
    /// [`Channel::thermal_relaxation`].
    ThermalRelaxation { t1: f64, t2: f64, time: f64 },
    /// [`Channel::rotation`]: a coherent error `exp(−iθP/2)`.
    Rotation { pauli: String, angle: f64 },
    /// [`Channel::from_kraus`] with explicit operators.
    Kraus { name: String, kraus: Vec<Vec<C64>> },
}

impl ChannelSpec {
    /// The channel this specification describes.
    pub fn to_channel(&self) -> Result<Channel> {
        match self {
            ChannelSpec::Depolarizing(p) => Channel::depolarizing(*p),
            ChannelSpec::Depolarizing2q(p) => Channel::depolarizing_2q(*p),
            ChannelSpec::AmplitudeDamping(g) => Channel::amplitude_damping(*g),
            ChannelSpec::PhaseDamping(l) => Channel::phase_damping(*l),
            ChannelSpec::BitFlip(p) => Channel::bit_flip(*p),
            ChannelSpec::PhaseFlip(p) => Channel::phase_flip(*p),
            ChannelSpec::Pauli(map) => {
                let list: Vec<(&str, f64)> = map.iter().map(|(k, &v)| (k.as_str(), v)).collect();
                Channel::pauli(&list)
            }
            ChannelSpec::ThermalRelaxation { t1, t2, time } => {
                Channel::thermal_relaxation(*t1, *t2, *time)
            }
            ChannelSpec::Rotation { pauli, angle } => Channel::rotation(pauli, *angle),
            ChannelSpec::Kraus { name, kraus } => Channel::from_kraus(name.clone(), kraus.clone()),
        }
    }
}

impl From<&Channel> for ChannelSpec {
    fn from(channel: &Channel) -> Self {
        ChannelSpec::Kraus {
            name: channel.name().to_string(),
            kraus: channel.kraus_operators().to_vec(),
        }
    }
}

/// Properties of one qubit. In [`DeviceModel::qubits`], fields left unset
/// take the value of [`DeviceModel::default_qubit`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct QubitProperties {
    /// Energy relaxation time T₁ (`None`: no relaxation).
    pub t1: Option<f64>,
    /// Dephasing time T₂ ≤ 2·T₁ (`None`: 2·T₁, no pure dephasing; for a
    /// listed qubit, `None` first falls back to `default_qubit`).
    pub t2: Option<f64>,
    /// Readout errors `(P(read 1 | 0), P(read 0 | 1))`.
    pub readout: Option<(f64, f64)>,
}

impl QubitProperties {
    /// A noiseless qubit.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set T₁ and T₂.
    pub fn with_t1_t2(mut self, t1: f64, t2: f64) -> Self {
        self.t1 = Some(t1);
        self.t2 = Some(t2);
        self
    }

    /// Set the readout errors `(P(read 1 | 0), P(read 0 | 1))`.
    pub fn with_readout(mut self, p01: f64, p10: f64) -> Self {
        self.readout = Some((p01, p10));
        self
    }

    /// `(T₁, T₂)` if the qubit relaxes at all.
    fn relaxation(&self) -> Option<(f64, f64)> {
        match (self.t1, self.t2) {
            (None, None) => None,
            (Some(t1), None) if t1.is_infinite() => None,
            (Some(t1), None) => Some((t1, 2.0 * t1)),
            (None, Some(t2)) => Some((f64::INFINITY, t2)),
            (Some(t1), Some(t2)) => Some((t1, t2)),
        }
    }

    fn validate(&self, what: &str) -> Result<()> {
        if let Some((t1, t2)) = self.relaxation() {
            if !(t1 > 0.0 && t2 > 0.0) || t2.is_infinite() || t1.is_nan() {
                return Err(QsoError::invalid(
                    "t1/t2",
                    format!("{what}: T1 and T2 must be positive and T2 finite"),
                ));
            }
            if t2 > 2.0 * t1 * (1.0 + 1e-12) {
                return Err(QsoError::invalid(
                    "t2",
                    format!("{what}: physical qubits have T2 ≤ 2·T1"),
                ));
            }
        }
        if let Some((p01, p10)) = self.readout {
            check_probability("readout", p01)?;
            check_probability("readout", p10)?;
        }
        Ok(())
    }
}

/// Noise attached to the gates a rule matches.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct GateRule {
    /// Gate name as in [`Circuit::gate_counts`] (`"x"`, `"cx"`, `"crz"`);
    /// `None` matches every gate.
    pub gate: Option<String>,
    /// The gate's qubits, controls first (`None`: any qubits).
    pub qubits: Option<Vec<usize>>,
    /// Number of qubits the gate acts on, controls included (`None`: any).
    pub num_qubits: Option<usize>,
    /// Duration of the gate.
    pub duration: f64,
    /// Depolarising probability after the gate: one-qubit depolarising for
    /// one-qubit gates, two-qubit depolarising on consecutive pairs of the
    /// gate's qubits otherwise.
    pub depolarizing: Option<f64>,
    /// Relative error of rotation angles (`rx`, `ry`, `rz`, `p`, `rxx`,
    /// `ryy`, `rzz` and their controlled forms): θ ↦ θ·(1 + ε).
    pub over_rotation: f64,
    /// Further channels after the gate: one-qubit channels act on every
    /// qubit of the gate, two-qubit channels on consecutive pairs.
    pub channels: Vec<ChannelSpec>,
}

impl GateRule {
    /// A rule matching every gate.
    pub fn any() -> Self {
        Self::default()
    }

    /// A rule matching gates named `name` (`"cx"`, `"sx"`, …).
    pub fn for_gate(name: &str) -> Self {
        Self {
            gate: Some(name.to_ascii_lowercase()),
            ..Self::default()
        }
    }

    /// Restrict the rule to gates on exactly these qubits (controls first).
    pub fn on_qubits(mut self, qubits: &[usize]) -> Self {
        self.qubits = Some(qubits.to_vec());
        self
    }

    /// Restrict the rule to gates on `k` qubits.
    pub fn with_num_qubits(mut self, k: usize) -> Self {
        self.num_qubits = Some(k);
        self
    }

    pub fn with_duration(mut self, duration: f64) -> Self {
        self.duration = duration;
        self
    }

    pub fn with_depolarizing(mut self, p: f64) -> Self {
        self.depolarizing = Some(p);
        self
    }

    pub fn with_over_rotation(mut self, epsilon: f64) -> Self {
        self.over_rotation = epsilon;
        self
    }

    pub fn with_channel(mut self, channel: ChannelSpec) -> Self {
        self.channels.push(channel);
        self
    }

    /// Whether the rule applies to `inst`.
    pub fn matches(&self, inst: &Instruction) -> bool {
        let n = inst.controls.len() + inst.targets.len();
        self.num_qubits.is_none_or(|k| k == n)
            && self
                .gate
                .as_deref()
                .is_none_or(|g| g.eq_ignore_ascii_case(&gate_name(inst)))
            && self
                .qubits
                .as_deref()
                .is_none_or(|qs| qs.iter().copied().eq(inst.qubits()))
    }

    fn validate(&self, index: usize) -> Result<()> {
        let what = format!("gate rule {index}");
        if !(self.duration >= 0.0 && self.duration.is_finite()) {
            return Err(QsoError::invalid(
                "duration",
                format!("{what}: duration must be finite and non-negative"),
            ));
        }
        if !self.over_rotation.is_finite() {
            return Err(QsoError::invalid(
                "over_rotation",
                format!("{what}: must be finite"),
            ));
        }
        if let Some(p) = self.depolarizing {
            check_probability("depolarizing", p)?;
        }
        if let Some(qs) = &self.qubits
            && self.num_qubits.is_some_and(|k| k != qs.len())
        {
            return Err(QsoError::invalid(
                "qubits",
                format!("{what}: qubits and num_qubits disagree"),
            ));
        }
        for spec in &self.channels {
            let ch = spec.to_channel()?;
            let smallest = self
                .qubits
                .as_ref()
                .map(Vec::len)
                .or(self.num_qubits)
                .or_else(|| self.gate.as_deref().and_then(name_arity))
                .unwrap_or(2);
            if ch.n_qubits() == 2 && smallest < 2 {
                return Err(QsoError::invalid(
                    "channels",
                    format!("{what}: two-qubit channel on one-qubit gates"),
                ));
            }
        }
        Ok(())
    }
}

/// Number of qubits of a gate name such as `"cx"` or `"ccrz"` (`None` for
/// unknown names and custom unitaries).
fn name_arity(name: &str) -> Option<usize> {
    let lower = name.to_ascii_lowercase();
    let base = lower.trim_start_matches('c');
    let controls = lower.len() - base.len();
    let targets = match base {
        "swap" | "rxx" | "ryy" | "rzz" => 2,
        "unitary" | "" => return None,
        _ => 1,
    };
    Some(controls + targets)
}

/// Name of an instruction as in [`Circuit::gate_counts`].
pub(crate) fn gate_name(inst: &Instruction) -> String {
    format!("{}{}", "c".repeat(inst.controls.len()), inst.kind.name())
}

/// A device: qubit properties and gate rules (see the module documentation).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct DeviceModel {
    pub name: String,
    /// Number of qubits of the device (`None`: any circuit size).
    pub num_qubits: Option<usize>,
    /// Properties of qubits `0, 1, …`. Unset fields, and qubits beyond the
    /// list, take the values of `default_qubit`.
    pub qubits: Vec<QubitProperties>,
    /// Properties of qubits not listed in `qubits`.
    pub default_qubit: QubitProperties,
    /// Gate rules; the first rule matching a gate applies.
    pub gates: Vec<GateRule>,
    /// Whether qubits relax while they wait for other qubits' gates and
    /// until the end of the circuit (default `true`).
    pub idle_noise: bool,
}

impl Default for DeviceModel {
    fn default() -> Self {
        Self {
            name: String::new(),
            num_qubits: None,
            qubits: Vec::new(),
            default_qubit: QubitProperties::default(),
            gates: Vec::new(),
            idle_noise: true,
        }
    }
}

impl DeviceModel {
    /// A noiseless device.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Set the properties of qubit `q` (unset fields keep following
    /// `default_qubit`, whenever that is set).
    pub fn with_qubit(mut self, q: usize, properties: QubitProperties) -> Self {
        if self.qubits.len() <= q {
            self.qubits.resize(q + 1, QubitProperties::default());
        }
        self.qubits[q] = properties;
        self
    }

    /// Set the properties of every qubit not set individually.
    pub fn with_default_qubit(mut self, properties: QubitProperties) -> Self {
        self.default_qubit = properties;
        self
    }

    /// Append a gate rule (rules are tried in order).
    pub fn with_rule(mut self, rule: GateRule) -> Self {
        self.gates.push(rule);
        self
    }

    /// Restrict the device to `n` qubits.
    pub fn with_num_qubits(mut self, n: usize) -> Self {
        self.num_qubits = Some(n);
        self
    }

    /// Switch relaxation of waiting qubits on or off.
    pub fn with_idle_noise(mut self, on: bool) -> Self {
        self.idle_noise = on;
        self
    }

    /// Properties of qubit `q`, with unset fields taken from
    /// `default_qubit`.
    pub fn qubit(&self, q: usize) -> QubitProperties {
        let d = &self.default_qubit;
        match self.qubits.get(q) {
            None => d.clone(),
            Some(p) => QubitProperties {
                t1: p.t1.or(d.t1),
                t2: p.t2.or(d.t2),
                readout: p.readout.or(d.readout),
            },
        }
    }

    /// The first rule matching `inst`.
    pub fn rule_for(&self, inst: &Instruction) -> Option<&GateRule> {
        self.gates.iter().find(|r| r.matches(inst))
    }

    /// Readout errors of qubits `0..n`.
    pub fn readout_errors(&self, n: usize) -> Vec<(f64, f64)> {
        (0..n)
            .map(|q| self.qubit(q).readout.unwrap_or((0.0, 0.0)))
            .collect()
    }

    /// Check every value and channel.
    pub fn validate(&self) -> Result<()> {
        if self.num_qubits == Some(0) {
            return Err(QsoError::invalid(
                "num_qubits",
                "a device needs at least one qubit",
            ));
        }
        if self.qubits.len() > crate::quantum::MAX_QUBITS {
            return Err(QsoError::invalid(
                "qubits",
                format!("more than {} qubit entries", crate::quantum::MAX_QUBITS),
            ));
        }
        if let Some(n) = self.num_qubits
            && self.qubits.len() > n
        {
            return Err(QsoError::invalid(
                "qubits",
                "more qubit entries than the device has qubits",
            ));
        }
        self.default_qubit.validate("default qubit")?;
        for q in 0..self.qubits.len() {
            self.qubit(q).validate(&format!("qubit {q}"))?;
        }
        for (i, rule) in self.gates.iter().enumerate() {
            rule.validate(i)?;
        }
        Ok(())
    }

    /// Parse and validate a JSON description.
    pub fn from_json(text: &str) -> Result<Self> {
        let device: DeviceModel = serde_json::from_str(text)
            .map_err(|e| QsoError::Parse(format!("device description: {e}")))?;
        device.validate()?;
        Ok(device)
    }

    /// The description as pretty-printed JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("device models serialise")
    }
}

impl From<&NoiseModel> for DeviceModel {
    /// The same noise as [`NoiseModel`]: its one-qubit channels after
    /// one-qubit gates, its multi-qubit channels after the others, and one
    /// readout error for all qubits.
    fn from(model: &NoiseModel) -> Self {
        let specs = |chs: &[Channel]| chs.iter().map(ChannelSpec::from).collect::<Vec<_>>();
        let mut device = DeviceModel::new("noise model").with_idle_noise(false);
        device.default_qubit.readout = model.readout;
        device.gates = vec![
            GateRule {
                num_qubits: Some(1),
                channels: specs(&model.single_qubit),
                ..GateRule::default()
            },
            GateRule {
                channels: specs(&model.multi_qubit),
                ..GateRule::default()
            },
        ];
        device
    }
}

/// How a channel is sampled on a pure state.
#[derive(Clone, Debug)]
pub(crate) enum Unraveling {
    /// Every Kraus operator is `√w·U`: pick `U` with probability `w`.
    /// `None` marks a multiple of the identity (nothing to apply).
    Mixture {
        cumulative: Vec<f64>,
        unitaries: Vec<Option<Vec<C64>>>,
    },
    /// General operators: the probabilities depend on the state through
    /// `K†K`. `no_jump` is the operator with the largest average weight
    /// `Tr(K†K)`, applied without a jump in the waiting-time method.
    General {
        kraus: Vec<Vec<C64>>,
        gram: Vec<Vec<C64>>,
        no_jump: usize,
    },
}

/// A channel prepared for both simulators.
#[derive(Clone, Debug)]
pub(crate) struct CompiledChannel {
    pub channel: Channel,
    pub superoperator: Vec<C64>,
    pub unraveling: Unraveling,
}

fn gram(k: &[C64], d: usize) -> Vec<C64> {
    let mut g = vec![C64::new(0.0, 0.0); d * d];
    for r in 0..d {
        for c in 0..d {
            for m in 0..d {
                g[r * d + c] += k[m * d + r].conj() * k[m * d + c];
            }
        }
    }
    g
}

impl CompiledChannel {
    pub(crate) fn new(channel: Channel) -> Self {
        let d = 1usize << channel.n_qubits();
        let grams: Vec<Vec<C64>> = channel
            .kraus_operators()
            .iter()
            .map(|k| gram(k, d))
            .collect();
        // K†K = w·I for every operator: a mixture of unitaries.
        let scalar = |g: &[C64]| {
            let w = g[0].re;
            (0..d * d).all(|i| {
                let expected = if i % (d + 1) == 0 { w } else { 0.0 };
                (g[i] - C64::new(expected, 0.0)).norm() <= 1e-12
            })
        };
        let unraveling = if grams.iter().all(|g| scalar(g)) {
            let mut cumulative = Vec::new();
            let mut unitaries = Vec::new();
            let mut acc = 0.0;
            for (k, g) in channel.kraus_operators().iter().zip(&grams) {
                let w = g[0].re;
                if w <= 0.0 {
                    continue;
                }
                acc += w;
                cumulative.push(acc);
                let u: Vec<C64> = k.iter().map(|z| z / w.sqrt()).collect();
                // A multiple of the identity only changes the global phase.
                let identity = (0..d * d).all(|i| {
                    if i % (d + 1) == 0 {
                        (u[i] - u[0]).norm() <= 1e-12
                    } else {
                        u[i].norm() <= 1e-12
                    }
                });
                unitaries.push((!identity).then_some(u));
            }
            Unraveling::Mixture {
                cumulative,
                unitaries,
            }
        } else {
            let weight = |g: &[C64]| (0..d).map(|i| g[i * (d + 1)].re).sum::<f64>();
            let no_jump = (0..grams.len())
                .max_by(|&a, &b| weight(&grams[a]).total_cmp(&weight(&grams[b])))
                .expect("at least one Kraus operator");
            Unraveling::General {
                kraus: channel.kraus_operators().to_vec(),
                gram: grams,
                no_jump,
            }
        };
        Self {
            superoperator: channel.superoperator(),
            channel,
            unraveling,
        }
    }
}

/// One step of a compiled noisy circuit.
#[derive(Clone, Debug)]
pub(crate) enum Op {
    /// Ideal gates with bound angles.
    Gates(Circuit),
    /// A channel (index into [`Program::channels`]) on one or two qubits.
    Channel { channel: usize, qubits: Vec<usize> },
}

/// A circuit compiled against a device: ideal gate segments and channels.
#[derive(Clone, Debug)]
pub(crate) struct Program {
    pub n_qubits: usize,
    pub ops: Vec<Op>,
    pub channels: Vec<CompiledChannel>,
    pub readout: Vec<(f64, f64)>,
}

pub(crate) fn rotation_kind(kind: GateKind) -> bool {
    use GateKind::*;
    matches!(kind, RX | RY | RZ | P | RXX | RYY | RZZ)
}

/// Builder state while compiling.
struct Compiler<'d> {
    device: &'d DeviceModel,
    program: Program,
    /// Channels by a cache key, to share thermal relaxation channels.
    cache: HashMap<String, usize>,
    rule_channels: Vec<Option<Vec<usize>>>,
}

impl Compiler<'_> {
    fn add_channel(
        &mut self,
        key: String,
        make: impl FnOnce() -> Result<Channel>,
    ) -> Result<usize> {
        if let Some(&i) = self.cache.get(&key) {
            return Ok(i);
        }
        let ch = make()?;
        self.program.channels.push(CompiledChannel::new(ch));
        let i = self.program.channels.len() - 1;
        self.cache.insert(key, i);
        Ok(i)
    }

    /// Append a channel; a channel directly after another one on the same
    /// qubits is merged with it into one (fewer passes over the state).
    fn push_channel(&mut self, channel: usize, qubits: Vec<usize>) -> Result<()> {
        let mixture = |ch: &CompiledChannel| matches!(ch.unraveling, Unraveling::Mixture { .. });
        // Only channels of one kind are merged: unitary mixtures stay
        // mixtures (sampled without looking at the state), general channels
        // keep a dominant no-jump operator.
        if let Some(Op::Channel {
            channel: last,
            qubits: last_qubits,
        }) = self.program.ops.last()
            && *last_qubits == qubits
            && mixture(&self.program.channels[*last]) == mixture(&self.program.channels[channel])
        {
            let (first, then) = (*last, channel);
            let key = format!("compose:{first}:{then}");
            let composed = match self.cache.get(&key) {
                Some(_) => None,
                None => {
                    let a = &self.program.channels[first].channel;
                    Some(a.compose(&self.program.channels[then].channel)?)
                }
            };
            let merged = self.add_channel(key, || Ok(composed.expect("not cached")))?;
            if let Some(Op::Channel { channel, .. }) = self.program.ops.last_mut() {
                *channel = merged;
            }
            return Ok(());
        }
        self.program.ops.push(Op::Channel { channel, qubits });
        Ok(())
    }

    /// Append a gate (an over-rotated angle may be invalid, e.g. infinite).
    fn push_gate(&mut self, inst: Instruction) -> Result<()> {
        if let Some(Op::Gates(c)) = self.program.ops.last_mut() {
            c.try_push(inst)?;
        } else {
            let mut c = Circuit::new(self.program.n_qubits);
            c.try_push(inst)?;
            self.program.ops.push(Op::Gates(c));
        }
        Ok(())
    }

    /// Relaxation of qubit `q` for `time`.
    fn relax(&mut self, q: usize, time: f64) -> Result<()> {
        if time <= 0.0 {
            return Ok(());
        }
        let Some((t1, t2)) = self.device.qubit(q).relaxation() else {
            return Ok(());
        };
        // Thermal relaxation as amplitude damping followed by a phase flip
        // (the same channel): the damping has a diagonal no-jump operator,
        // and the phase flip is a unitary mixture.
        let gamma = 1.0 - (-time / t1).exp();
        let rate = (1.0 / t2 - 0.5 / t1).max(0.0);
        let flip = 0.5 * (1.0 - (-time * rate).exp());
        if gamma > 0.0 {
            let key = format!("damp:{}", gamma.to_bits());
            let i = self.add_channel(key, || Channel::amplitude_damping(gamma))?;
            self.push_channel(i, vec![q])?;
        }
        if flip > 0.0 {
            let key = format!("dephase:{}", flip.to_bits());
            let i = self.add_channel(key, || Channel::phase_flip(flip))?;
            self.push_channel(i, vec![q])?;
        }
        Ok(())
    }

    /// Channel indices of rule `r` (compiled once).
    fn rule_channels(&mut self, r: usize) -> Result<Vec<usize>> {
        if let Some(list) = &self.rule_channels[r] {
            return Ok(list.clone());
        }
        let specs = self.device.gates[r].channels.clone();
        let mut list = Vec::with_capacity(specs.len());
        for (j, spec) in specs.iter().enumerate() {
            list.push(self.add_channel(format!("rule:{r}:{j}"), || spec.to_channel())?);
        }
        self.rule_channels[r] = Some(list.clone());
        Ok(list)
    }

    /// Apply channel `i` to `qubits`: per qubit for one-qubit channels, on
    /// consecutive pairs for two-qubit channels.
    fn spread(&mut self, i: usize, qubits: &[usize]) -> Result<()> {
        match self.program.channels[i].channel.n_qubits() {
            1 => {
                for &q in qubits {
                    self.push_channel(i, vec![q])?;
                }
            }
            _ => {
                if qubits.len() < 2 {
                    return Err(QsoError::invalid(
                        "channels",
                        format!(
                            "two-qubit channel {} after a one-qubit gate",
                            self.program.channels[i].channel.name()
                        ),
                    ));
                }
                for pair in qubits.windows(2) {
                    self.push_channel(i, pair.to_vec())?;
                }
            }
        }
        Ok(())
    }
}

impl Program {
    /// Compile `circuit(theta)` against `device` (validated by the caller).
    pub(crate) fn compile(circuit: &Circuit, theta: &[f64], device: &DeviceModel) -> Result<Self> {
        circuit.check_parameters(theta)?;
        let n = circuit.n_qubits();
        if let Some(max) = device.num_qubits
            && n > max
        {
            return Err(QsoError::TooManyQubits {
                what: "device",
                max,
                requested: n,
            });
        }
        let mut cc = Compiler {
            device,
            program: Program {
                n_qubits: n,
                ops: Vec::new(),
                channels: Vec::new(),
                readout: device.readout_errors(n),
            },
            cache: HashMap::new(),
            rule_channels: vec![None; device.gates.len()],
        };
        let mut clock = vec![0.0f64; n];
        let mut touched = vec![false; n];
        for inst in circuit.instructions() {
            let qubits: Vec<usize> = inst.qubits().collect();
            let rule = device.gates.iter().position(|r| r.matches(inst));
            let duration = rule.map_or(0.0, |r| device.gates[r].duration);
            let start = qubits.iter().map(|&q| clock[q]).fold(0.0, f64::max);
            if device.idle_noise {
                for &q in &qubits {
                    if touched[q] {
                        cc.relax(q, start - clock[q])?;
                    }
                }
            }
            let mut angles = inst.angles(theta);
            if let Some(r) = rule {
                let eps = device.gates[r].over_rotation;
                if eps != 0.0 && rotation_kind(inst.kind) {
                    for a in &mut angles {
                        *a *= 1.0 + eps;
                    }
                }
            }
            let mut bound = inst.clone();
            bound.params = angles.into_iter().map(Param::Value).collect();
            cc.push_gate(bound)?;
            if let Some(r) = rule {
                if let Some(p) = device.gates[r].depolarizing
                    && p > 0.0
                {
                    let (key, two) = (
                        format!("depol:{}:{}", qubits.len() > 1, p.to_bits()),
                        qubits.len() > 1,
                    );
                    let i = cc.add_channel(key, || {
                        if two {
                            Channel::depolarizing_2q(p)
                        } else {
                            Channel::depolarizing(p)
                        }
                    })?;
                    cc.spread(i, &qubits)?;
                }
                for i in cc.rule_channels(r)? {
                    cc.spread(i, &qubits)?;
                }
                for &q in &qubits {
                    cc.relax(q, duration)?;
                }
            }
            for &q in &qubits {
                clock[q] = start + duration;
                touched[q] = true;
            }
        }
        if device.idle_noise {
            let end = clock.iter().copied().fold(0.0, f64::max);
            for q in 0..n {
                if touched[q] {
                    cc.relax(q, end - clock[q])?;
                }
            }
        }
        Ok(cc.program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::DensityMatrix;
    use crate::quantum::StateVector;

    fn max_diff(a: &DensityMatrix, b: &DensityMatrix) -> f64 {
        (a.to_matrix() - b.to_matrix())
            .iter()
            .map(|z| z.norm())
            .fold(0.0, f64::max)
    }

    fn test_circuit() -> Circuit {
        let mut c = Circuit::new(3);
        c.h(0)
            .ry(1, 0.7)
            .cx(0, 1)
            .rz(2, -0.4)
            .crz(1, 2, 0.9)
            .ccx(0, 1, 2)
            .swap(0, 2)
            .sx(1);
        c
    }

    #[test]
    fn noise_models_compile_to_the_same_channels() {
        let model = NoiseModel {
            single_qubit: vec![
                Channel::depolarizing(0.03).unwrap(),
                Channel::amplitude_damping(0.05).unwrap(),
            ],
            multi_qubit: vec![
                Channel::phase_damping(0.04).unwrap(),
                Channel::depolarizing_2q(0.06).unwrap(),
            ],
            readout: Some((0.01, 0.02)),
        };
        let c = test_circuit();
        let mut expected = DensityMatrix::zero(3).unwrap();
        expected.apply_circuit(&c, &[], Some(&model)).unwrap();
        let device = DeviceModel::from(&model);
        let mut rho = DensityMatrix::zero(3).unwrap();
        rho.apply_device(&c, &[], &device).unwrap();
        assert!(max_diff(&rho, &expected) < 1e-13);
        assert_eq!(device.readout_errors(3), vec![(0.01, 0.02); 3]);
    }

    #[test]
    fn schedule_relaxes_waiting_qubits() {
        let q0 = QubitProperties::new().with_t1_t2(50.0, 30.0);
        let q1 = QubitProperties::new().with_t1_t2(80.0, 100.0);
        let zz = ChannelSpec::Rotation {
            pauli: "ZZ".into(),
            angle: 0.05,
        };
        let device = DeviceModel::new("schedule")
            .with_qubit(0, q0)
            .with_qubit(1, q1)
            .with_rule(GateRule::for_gate("x").with_duration(1.0))
            .with_rule(
                GateRule::for_gate("cx")
                    .with_duration(3.0)
                    .with_depolarizing(0.02)
                    .with_channel(zz.clone()),
            )
            .with_rule(GateRule::any().with_num_qubits(1).with_duration(0.25));
        let mut c = Circuit::new(2);
        c.x(0).h(1).cx(0, 1).h(0);
        let mut rho = DensityMatrix::zero(2).unwrap();
        rho.apply_device(&c, &[], &device).unwrap();

        // The same by hand: x(0) 0→1, h(1) 0→0.25, q1 waits 0.75, cx 1→4,
        // h(0) 4→4.25, q1 waits 0.25 at the end.
        let relax = |t1: f64, t2: f64, t: f64| Channel::thermal_relaxation(t1, t2, t).unwrap();
        let mut e = DensityMatrix::zero(2).unwrap();
        let gate = |e: &mut DensityMatrix, f: &dyn Fn(&mut Circuit)| {
            let mut g = Circuit::new(2);
            f(&mut g);
            e.apply_circuit(&g, &[], None).unwrap();
        };
        gate(&mut e, &|g| {
            g.x(0);
        });
        e.apply_channel(&relax(50.0, 30.0, 1.0), &[0]).unwrap();
        gate(&mut e, &|g| {
            g.h(1);
        });
        e.apply_channel(&relax(80.0, 100.0, 0.25), &[1]).unwrap();
        e.apply_channel(&relax(80.0, 100.0, 0.75), &[1]).unwrap();
        gate(&mut e, &|g| {
            g.cx(0, 1);
        });
        e.apply_channel(&Channel::depolarizing_2q(0.02).unwrap(), &[0, 1])
            .unwrap();
        e.apply_channel(&zz.to_channel().unwrap(), &[0, 1]).unwrap();
        e.apply_channel(&relax(50.0, 30.0, 3.0), &[0]).unwrap();
        e.apply_channel(&relax(80.0, 100.0, 3.0), &[1]).unwrap();
        gate(&mut e, &|g| {
            g.h(0);
        });
        e.apply_channel(&relax(50.0, 30.0, 0.25), &[0]).unwrap();
        e.apply_channel(&relax(80.0, 100.0, 0.25), &[1]).unwrap();
        assert!(max_diff(&rho, &e) < 1e-14, "{}", max_diff(&rho, &e));

        // Without idle noise only the relaxation during gates remains.
        let quiet = device.clone().with_idle_noise(false);
        let mut r2 = DensityMatrix::zero(2).unwrap();
        r2.apply_device(&c, &[], &quiet).unwrap();
        assert!(max_diff(&r2, &rho) > 1e-4);
    }

    #[test]
    fn over_rotation_scales_rotation_angles() {
        let device =
            DeviceModel::new("coherent").with_rule(GateRule::any().with_over_rotation(0.1));
        let mut c = Circuit::new(2);
        c.h(0)
            .rx(0, 0.8)
            .crz(0, 1, -1.2)
            .rzz(0, 1, 0.4)
            .u(1, 0.3, 0.2, 0.1);
        let mut rho = DensityMatrix::zero(2).unwrap();
        rho.apply_device(&c, &[], &device).unwrap();
        let mut ideal = Circuit::new(2);
        ideal
            .h(0)
            .rx(0, 0.88)
            .crz(0, 1, -1.32)
            .rzz(0, 1, 0.44)
            .u(1, 0.3, 0.2, 0.1);
        let psi = StateVector::from_circuit(&ideal, &[]).unwrap();
        assert!((rho.fidelity_with_pure(&psi).unwrap() - 1.0).abs() < 1e-12);
        assert!((rho.purity() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn unset_qubit_fields_follow_the_default() {
        // The order of the builder calls does not matter.
        let a = DeviceModel::new("a")
            .with_qubit(3, QubitProperties::new().with_readout(0.1, 0.2))
            .with_default_qubit(
                QubitProperties::new()
                    .with_t1_t2(50.0, 40.0)
                    .with_readout(0.01, 0.02),
            );
        let b = DeviceModel::new("b")
            .with_default_qubit(
                QubitProperties::new()
                    .with_t1_t2(50.0, 40.0)
                    .with_readout(0.01, 0.02),
            )
            .with_qubit(3, QubitProperties::new().with_readout(0.1, 0.2));
        for d in [&a, &b] {
            assert_eq!(
                d.qubit(0),
                QubitProperties::new()
                    .with_t1_t2(50.0, 40.0)
                    .with_readout(0.01, 0.02)
            );
            assert_eq!(
                d.qubit(3),
                QubitProperties::new()
                    .with_t1_t2(50.0, 40.0)
                    .with_readout(0.1, 0.2)
            );
            assert_eq!(d.qubit(7).readout, Some((0.01, 0.02)));
        }
        // Merged values are validated: T2 = 90 > 2·T1 for qubit 0.
        let bad = DeviceModel::new("c")
            .with_default_qubit(QubitProperties::new().with_t1_t2(40.0, 30.0))
            .with_qubit(
                0,
                QubitProperties {
                    t2: Some(90.0),
                    ..QubitProperties::default()
                },
            );
        assert!(bad.validate().is_err());
    }

    #[test]
    fn rules_match_in_order() {
        let mut c = Circuit::new(4);
        c.cx(0, 1).cx(1, 0).crz(2, 3, 0.1).mcx(&[0, 1, 2], 3).h(2);
        let i = c.instructions();
        assert_eq!(gate_name(&i[2]), "crz");
        assert_eq!(gate_name(&i[3]), "cccx");
        let pair = GateRule::for_gate("cx").on_qubits(&[0, 1]);
        assert!(pair.matches(&i[0]) && !pair.matches(&i[1]));
        let two = GateRule::any().with_num_qubits(2);
        assert!(two.matches(&i[1]) && two.matches(&i[2]) && !two.matches(&i[4]));
        let device = DeviceModel::new("d")
            .with_rule(pair.with_depolarizing(0.5))
            .with_rule(two.with_depolarizing(0.1));
        assert_eq!(device.rule_for(&i[0]).unwrap().depolarizing, Some(0.5));
        assert_eq!(device.rule_for(&i[1]).unwrap().depolarizing, Some(0.1));
        assert!(device.rule_for(&i[4]).is_none());
    }

    #[test]
    fn json_descriptions() {
        let text = r#"{
            "name": "toy",
            "num_qubits": 3,
            "qubits": [{"t1": 100.0, "t2": 70.0, "readout": [0.02, 0.05]}],
            "default_qubit": {"t1": 80.0},
            "gates": [
                {"gate": "cx", "qubits": [0, 1], "duration": 0.3, "depolarizing": 0.01,
                 "channels": [{"pauli": {"ZZ": 0.002}}, {"rotation": {"pauli": "ZZ", "angle": 0.01}}]},
                {"num_qubits": 1, "duration": 0.03, "over_rotation": 0.001,
                 "channels": [{"amplitude_damping": 0.001}]}
            ]
        }"#;
        let device = DeviceModel::from_json(text).unwrap();
        assert_eq!(device.qubit(0).readout, Some((0.02, 0.05)));
        assert_eq!(device.qubit(2).t1, Some(80.0));
        assert!(device.idle_noise);
        assert_eq!(DeviceModel::from_json(&device.to_json()).unwrap(), device);
        let mut c = Circuit::new(4);
        c.h(0);
        assert!(Program::compile(&c, &[], &device).is_err());

        for bad in [
            r#"{"qubits": [{"t1": 10.0, "t2": 30.0}]}"#,
            r#"{"default_qubit": {"readout": [1.5, 0.0]}}"#,
            r#"{"gates": [{"depolarizing": -0.1}]}"#,
            r#"{"gates": [{"num_qubits": 1, "channels": [{"depolarizing_2q": 0.1}]}]}"#,
            r#"{"gates": [{"duration": -1.0}]}"#,
            r#"{"gates": [{"channels": [{"pauli": {"XQ": 0.1}}]}]}"#,
            r#"{"colour": "blue"}"#,
            r#"{"gates": [{"qubits": [0], "num_qubits": 2}]}"#,
        ] {
            assert!(DeviceModel::from_json(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn pauli_channels_and_twirling() {
        let p = 0.09;
        let dep = Channel::depolarizing(p)
            .unwrap()
            .pauli_probabilities()
            .unwrap();
        for (a, b) in dep.iter().zip([1.0 - p, p / 3.0, p / 3.0, p / 3.0]) {
            assert!((a - b).abs() < 1e-15);
        }
        let gamma: f64 = 0.3;
        let ad = Channel::amplitude_damping(gamma).unwrap();
        assert!(ad.pauli_probabilities().is_none());
        let a = (1.0 - gamma).sqrt();
        let tw = ad.twirled().pauli_probabilities().unwrap();
        let expected = [
            (1.0 + a).powi(2) / 4.0,
            gamma / 4.0,
            gamma / 4.0,
            (1.0 - a).powi(2) / 4.0,
        ];
        for (x, y) in tw.iter().zip(expected) {
            assert!((x - y).abs() < 1e-14);
        }
        let zz = Channel::pauli(&[("ZZ", 0.1), ("XI", 0.05)]).unwrap();
        let probs = zz.pauli_probabilities().unwrap();
        assert!((probs[15] - 0.1).abs() < 1e-15 && (probs[4] - 0.05).abs() < 1e-15);
        assert!((probs[0] - 0.85).abs() < 1e-15);
        assert!(Channel::pauli(&[("ZZ", 0.6), ("XX", 0.6)]).is_err());
        assert!(Channel::pauli(&[("Z", 0.1), ("XX", 0.1)]).is_err());
        assert!(Channel::pauli(&[("I", 0.1)]).is_err());

        // Averaging P·C(PρP)·P over all Paulis gives the twirled channel.
        let mut c = Circuit::new(2);
        c.ry(0, 0.7).h(1).cx(0, 1).rz(1, 0.3).rx(0, -0.5);
        let rho = DensityMatrix::from_circuit(&c, &[]).unwrap();
        let ch = Channel::thermal_relaxation(10.0, 6.0, 3.0).unwrap();
        let mut avg = vec![C64::new(0.0, 0.0); 16];
        for label in ["I", "X", "Y", "Z"] {
            let mut r = rho.clone();
            let pc = Channel::rotation(label, std::f64::consts::PI).unwrap();
            r.apply_channel(&pc, &[1]).unwrap();
            r.apply_channel(&ch, &[1]).unwrap();
            r.apply_channel(&pc, &[1]).unwrap();
            for (x, y) in avg.iter_mut().zip(&r.data) {
                *x += y / 4.0;
            }
        }
        let mut twirled = rho.clone();
        twirled.apply_channel(&ch.twirled(), &[1]).unwrap();
        for (x, y) in avg.iter().zip(&twirled.data) {
            assert!((x - y).norm() < 1e-14);
        }
    }

    #[test]
    fn rotations_are_gates() {
        let mut c = Circuit::new(2);
        c.h(0).h(1);
        let base = DensityMatrix::from_circuit(&c, &[]).unwrap();
        for (label, qubits, gate) in [
            ("Z", vec![0], {
                let mut g = c.clone();
                g.rz(0, 0.4);
                g
            }),
            ("X", vec![1], {
                let mut g = c.clone();
                g.rx(1, 0.4);
                g
            }),
            ("ZZ", vec![0, 1], {
                let mut g = c.clone();
                g.rzz(0, 1, 0.4);
                g
            }),
            ("XY", vec![1, 0], {
                let mut g = c.clone();
                g.h(1).sdg(0).h(0).rzz(1, 0, 0.4).h(0).s(0).h(1);
                g
            }),
        ] {
            let mut r = base.clone();
            r.apply_channel(&Channel::rotation(label, 0.4).unwrap(), &qubits)
                .unwrap();
            let expected = DensityMatrix::from_circuit(&gate, &[]).unwrap();
            assert!(max_diff(&r, &expected) < 1e-14, "{label}");
        }
        assert!(Channel::unitary("bad", vec![C64::new(2.0, 0.0); 4]).is_err());
    }

    #[test]
    fn review_regressions() {
        // Over-rotations that overflow the angle are errors, not panics.
        let dev = DeviceModel::from_json(r#"{"gates":[{"over_rotation":1.0}]}"#).unwrap();
        let mut c = Circuit::new(1);
        c.rx(0, 1e308);
        let mut rho = DensityMatrix::zero(1).unwrap();
        assert!(rho.apply_device(&c, &[], &dev).is_err());
        // Long rotation labels are rejected before building matrices.
        let long =
            r#"{"gates":[{"channels":[{"rotation":{"pauli":"ZZZZZZZZZZZZZZZZ","angle":0.1}}]}]}"#;
        assert!(DeviceModel::from_json(long).is_err());
        assert!(Channel::rotation("ZZZ", 0.1).is_err());
        // Gate names match regardless of case.
        let dev =
            DeviceModel::from_json(r#"{"gates":[{"gate":"CX","depolarizing":0.5}]}"#).unwrap();
        let mut c = Circuit::new(2);
        c.cx(0, 1);
        let mut rho = DensityMatrix::zero(2).unwrap();
        rho.apply_device(&c, &[], &dev).unwrap();
        assert!(rho.purity() < 0.9);
        // Two-qubit channels on named one-qubit gates fail validation.
        let bad = DeviceModel::new("h")
            .with_rule(GateRule::for_gate("h").with_channel(ChannelSpec::Depolarizing2q(0.1)));
        assert!(bad.validate().is_err());
        // T1 = ∞ alone means no relaxation.
        let mut q = QubitProperties::new();
        q.t1 = Some(f64::INFINITY);
        assert!(q.relaxation().is_none());
        assert!(
            DeviceModel::new("inf")
                .with_default_qubit(q)
                .validate()
                .is_ok()
        );
        // Composing many unitary mixtures stays bounded.
        let half = Channel::from_kraus(
            "half",
            vec![
                vec![
                    C64::new(0.5f64.sqrt(), 0.0),
                    C64::new(0.0, 0.0),
                    C64::new(0.0, 0.0),
                    C64::new(0.5f64.sqrt(), 0.0),
                ],
                vec![
                    C64::new(0.5, 0.0),
                    C64::new(0.5, 0.0),
                    C64::new(0.5, 0.0),
                    C64::new(-0.5, 0.0),
                ],
            ],
        )
        .unwrap();
        let mut acc = half.clone();
        for _ in 0..20 {
            acc = acc.compose(&half).unwrap();
            assert!(acc.kraus_operators().len() <= 64);
        }
    }

    #[test]
    fn unravelings() {
        let dep = CompiledChannel::new(Channel::depolarizing(0.1).unwrap());
        match &dep.unraveling {
            Unraveling::Mixture {
                cumulative,
                unitaries,
            } => {
                assert_eq!(unitaries.len(), 4);
                assert!(unitaries[0].is_none() && unitaries[1].is_some());
                assert!((cumulative[3] - 1.0).abs() < 1e-15);
            }
            Unraveling::General { .. } => panic!("depolarising is a unitary mixture"),
        }
        let ad = CompiledChannel::new(Channel::amplitude_damping(0.1).unwrap());
        match &ad.unraveling {
            Unraveling::General { no_jump, kraus, .. } => {
                // The no-jump operator diag(1, √(1−γ)) carries most weight.
                assert!((kraus[*no_jump][0].re - 1.0).abs() < 1e-15);
            }
            Unraveling::Mixture { .. } => panic!("amplitude damping is not a unitary mixture"),
        }
    }
}
