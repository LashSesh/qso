//! Quantum circuits with symbolic parameters.
//!
//! A [`Circuit`] is a list of [`Instruction`]s. Gate angles are [`Param`]s:
//! either fixed numbers or affine functions `scale·θ[index] + offset` of a
//! parameter vector supplied at evaluation time. This is what variational
//! algorithms optimise and what the gradient engines differentiate.
//!
//! ```
//! use metatron_qso::quantum::{Circuit, StateVector};
//!
//! // Bell state preparation.
//! let mut bell = Circuit::new(2);
//! bell.h(0).cx(0, 1);
//! let psi = StateVector::from_circuit(&bell, &[]).unwrap();
//! assert!((psi.probabilities()[0b00] - 0.5).abs() < 1e-12);
//! assert!((psi.probabilities()[0b11] - 0.5).abs() < 1e-12);
//!
//! // A parametrised rotation.
//! let mut ansatz = Circuit::new(1);
//! let theta = ansatz.new_parameter();
//! ansatz.ry(0, theta);
//! let psi = StateVector::from_circuit(&ansatz, &[std::f64::consts::PI]).unwrap();
//! assert!((psi.probabilities()[1] - 1.0).abs() < 1e-12);
//! ```

use std::collections::BTreeMap;
use std::ops::{Add, Mul, Neg, Sub};
use std::sync::Arc;

use num_complex::Complex64 as C64;
use serde::{Deserialize, Serialize};

use super::gates::{GateKind, dagger, is_unitary};
use super::pauli::{Pauli, PauliString};
use crate::error::{QsoError, Result};

/// Handle to circuit parameter `index`, optionally scaled and shifted.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub index: usize,
    pub scale: f64,
    pub offset: f64,
}

impl Parameter {
    /// The bare parameter `θ[index]`.
    pub fn new(index: usize) -> Self {
        Self {
            index,
            scale: 1.0,
            offset: 0.0,
        }
    }
}

impl Mul<f64> for Parameter {
    type Output = Parameter;
    fn mul(self, rhs: f64) -> Parameter {
        Parameter {
            index: self.index,
            scale: self.scale * rhs,
            offset: self.offset * rhs,
        }
    }
}

impl Mul<Parameter> for f64 {
    type Output = Parameter;
    fn mul(self, rhs: Parameter) -> Parameter {
        rhs * self
    }
}

impl Add<f64> for Parameter {
    type Output = Parameter;
    fn add(self, rhs: f64) -> Parameter {
        Parameter {
            offset: self.offset + rhs,
            ..self
        }
    }
}

impl Sub<f64> for Parameter {
    type Output = Parameter;
    fn sub(self, rhs: f64) -> Parameter {
        self + (-rhs)
    }
}

impl Neg for Parameter {
    type Output = Parameter;
    fn neg(self) -> Parameter {
        self * -1.0
    }
}

/// A gate angle: fixed value or (affine function of) a circuit parameter.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Param {
    Value(f64),
    Symbol(Parameter),
}

impl Param {
    /// Numerical value for the parameter vector `theta`.
    #[inline]
    pub fn value(&self, theta: &[f64]) -> f64 {
        match *self {
            Param::Value(v) => v,
            Param::Symbol(p) => p.scale * theta[p.index] + p.offset,
        }
    }

    fn negated(self) -> Param {
        self.scaled(-1.0)
    }

    /// The angle multiplied by `s`.
    pub(crate) fn scaled(self, s: f64) -> Param {
        match self {
            Param::Value(v) => Param::Value(s * v),
            Param::Symbol(p) => Param::Symbol(p * s),
        }
    }
}

impl From<f64> for Param {
    fn from(v: f64) -> Self {
        Param::Value(v)
    }
}

impl From<Parameter> for Param {
    fn from(p: Parameter) -> Self {
        Param::Symbol(p)
    }
}

/// A gate applied to concrete qubits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    pub kind: GateKind,
    /// Target qubits; for two-qubit kinds the first is the more significant
    /// local bit of the gate matrix.
    pub targets: Vec<usize>,
    /// Control qubits (gate acts when all are |1⟩).
    pub controls: Vec<usize>,
    pub params: Vec<Param>,
    /// Row-major matrix for [`GateKind::Unitary`].
    #[serde(default, with = "matrix_serde")]
    pub matrix: Option<Arc<Vec<C64>>>,
}

/// (De)serialise `Option<Arc<Vec<C64>>>` as an optional list of `[re, im]`.
mod matrix_serde {
    use std::sync::Arc;

    use num_complex::Complex64 as C64;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S: Serializer>(
        m: &Option<Arc<Vec<C64>>>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        m.as_deref().serialize(s)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Arc<Vec<C64>>>, D::Error> {
        Ok(Option::<Vec<C64>>::deserialize(d)?.map(Arc::new))
    }
}

impl Instruction {
    /// All qubits touched (controls first, then targets).
    pub fn qubits(&self) -> impl Iterator<Item = usize> + '_ {
        self.controls.iter().chain(self.targets.iter()).copied()
    }

    /// Bit mask of the control qubits.
    pub fn control_mask(&self) -> usize {
        self.controls.iter().fold(0, |m, &c| m | (1 << c))
    }

    /// Whether any angle depends on circuit parameters.
    pub fn is_parametrised(&self) -> bool {
        self.params.iter().any(|p| matches!(p, Param::Symbol(_)))
    }

    /// Bind parameters, returning concrete angles.
    pub fn angles(&self, theta: &[f64]) -> Vec<f64> {
        self.params.iter().map(|p| p.value(theta)).collect()
    }

    /// The inverse instruction.
    pub fn inverse(&self) -> Instruction {
        use GateKind::*;
        let mut inv = self.clone();
        match self.kind {
            H | X | Y | Z | Swap => {}
            S => inv.kind = Sdg,
            Sdg => inv.kind = S,
            T => inv.kind = Tdg,
            Tdg => inv.kind = T,
            SX => inv.kind = SXdg,
            SXdg => inv.kind = SX,
            RX | RY | RZ | P | RXX | RYY | RZZ => inv.params = vec![self.params[0].negated()],
            U => {
                inv.params = vec![
                    self.params[0].negated(),
                    self.params[2].negated(),
                    self.params[1].negated(),
                ]
            }
            Unitary => {
                let m = self
                    .matrix
                    .as_ref()
                    .expect("unitary instruction carries a matrix");
                inv.matrix = Some(Arc::new(dagger(m)));
            }
        }
        inv
    }
}

/// An ordered list of instructions on `n_qubits` qubits.
///
/// Deserialisation re-validates every instruction, so a circuit read from
/// untrusted JSON is as safe to simulate as one built in code.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "CircuitData")]
pub struct Circuit {
    n_qubits: usize,
    n_params: usize,
    instructions: Vec<Instruction>,
}

/// Unvalidated serialised form of a [`Circuit`].
#[derive(Deserialize)]
struct CircuitData {
    n_qubits: usize,
    n_params: usize,
    instructions: Vec<Instruction>,
}

impl TryFrom<CircuitData> for Circuit {
    type Error = QsoError;

    fn try_from(data: CircuitData) -> Result<Self> {
        let mut c = Circuit::new(data.n_qubits);
        for inst in data.instructions {
            c.try_push(inst)?;
        }
        c.reserve_parameters(data.n_params);
        Ok(c)
    }
}

macro_rules! fixed_1q {
    ($($(#[$doc:meta])* $name:ident => $kind:ident),* $(,)?) => {
        $(
            $(#[$doc])*
            pub fn $name(&mut self, q: usize) -> &mut Self {
                self.push_gate(GateKind::$kind, &[q], &[], &[])
            }
        )*
    };
}

macro_rules! rot_1q {
    ($($(#[$doc:meta])* $name:ident => $kind:ident),* $(,)?) => {
        $(
            $(#[$doc])*
            pub fn $name(&mut self, q: usize, angle: impl Into<Param>) -> &mut Self {
                self.push_gate(GateKind::$kind, &[q], &[], &[angle.into()])
            }
        )*
    };
}

macro_rules! rot_2q {
    ($($(#[$doc:meta])* $name:ident => $kind:ident),* $(,)?) => {
        $(
            $(#[$doc])*
            pub fn $name(&mut self, a: usize, b: usize, angle: impl Into<Param>) -> &mut Self {
                self.push_gate(GateKind::$kind, &[a, b], &[], &[angle.into()])
            }
        )*
    };
}

macro_rules! controlled_rot {
    ($($(#[$doc:meta])* $name:ident => $kind:ident),* $(,)?) => {
        $(
            $(#[$doc])*
            pub fn $name(&mut self, control: usize, target: usize, angle: impl Into<Param>) -> &mut Self {
                self.push_gate(GateKind::$kind, &[target], &[control], &[angle.into()])
            }
        )*
    };
}

impl Circuit {
    /// Empty circuit on `n_qubits` qubits.
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            n_params: 0,
            instructions: Vec::new(),
        }
    }

    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Number of free parameters the circuit expects.
    pub fn num_parameters(&self) -> usize {
        self.n_params
    }

    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// Mutable access for in-crate rewrites that preserve validity (e.g.
    /// shifting a bound angle).
    pub(crate) fn instructions_mut(&mut self) -> &mut [Instruction] {
        &mut self.instructions
    }

    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Allocate a fresh parameter `θ[k]`.
    pub fn new_parameter(&mut self) -> Parameter {
        let p = Parameter::new(self.n_params);
        self.n_params += 1;
        p
    }

    /// Allocate `n` fresh parameters.
    pub fn new_parameters(&mut self, n: usize) -> Vec<Parameter> {
        (0..n).map(|_| self.new_parameter()).collect()
    }

    /// Declare that the circuit expects at least `n` parameters (useful
    /// when building instructions with [`Parameter::new`] directly).
    pub fn reserve_parameters(&mut self, n: usize) {
        self.n_params = self.n_params.max(n);
    }

    /// Append a validated instruction.
    pub fn try_push(&mut self, inst: Instruction) -> Result<&mut Self> {
        self.validate(&inst)?;
        for p in &inst.params {
            if let Param::Symbol(s) = p {
                self.n_params = self.n_params.max(s.index + 1);
            }
        }
        self.instructions.push(inst);
        Ok(self)
    }

    fn validate(&self, inst: &Instruction) -> Result<()> {
        inst.validate(self.n_qubits)
    }

    /// Append an instruction without validation (for crate-internal
    /// non-unitary matrices such as the no-jump Kraus operators of quantum
    /// trajectories; qubits and shapes must be valid).
    pub(crate) fn push_trusted(&mut self, inst: Instruction) {
        debug_assert!(inst.qubits().all(|q| q < self.n_qubits));
        self.instructions.push(inst);
    }
}

impl Instruction {
    /// Check arity, parameter count, qubit range and distinctness, finite
    /// angles, and (for custom unitaries) shape and unitarity of the matrix.
    pub fn validate(&self, n_qubits: usize) -> Result<()> {
        let inst = self;
        let n_targets = match inst.kind {
            GateKind::Unitary => {
                let m = inst
                    .matrix
                    .as_ref()
                    .ok_or_else(|| QsoError::invalid("matrix", "unitary gate without matrix"))?;
                if inst.targets.len() > super::kernels::MAX_DENSE_TARGETS {
                    return Err(QsoError::invalid(
                        "targets",
                        format!(
                            "custom unitaries act on at most {} qubits",
                            super::kernels::MAX_DENSE_TARGETS
                        ),
                    ));
                }
                let dim = 1usize << inst.targets.len();
                if m.len() != dim * dim {
                    return Err(QsoError::DimensionMismatch {
                        expected: dim * dim,
                        actual: m.len(),
                    });
                }
                if !is_unitary(m, 1e-9) {
                    return Err(QsoError::invalid("matrix", "matrix is not unitary"));
                }
                inst.targets.len()
            }
            kind => kind.num_targets().expect("fixed arity"),
        };
        if inst.targets.len() != n_targets || n_targets == 0 {
            return Err(QsoError::invalid(
                "targets",
                format!(
                    "{} expects {n_targets} target(s), got {}",
                    inst.kind.name(),
                    inst.targets.len()
                ),
            ));
        }
        if inst.params.len() != inst.kind.num_params() {
            return Err(QsoError::invalid(
                "params",
                format!(
                    "{} expects {} parameter(s), got {}",
                    inst.kind.name(),
                    inst.kind.num_params(),
                    inst.params.len()
                ),
            ));
        }
        let mut seen: Vec<usize> = Vec::with_capacity(inst.targets.len() + inst.controls.len());
        for q in inst.qubits() {
            if q >= n_qubits {
                return Err(QsoError::QubitOutOfRange { qubit: q, n_qubits });
            }
            if seen.contains(&q) {
                return Err(QsoError::DuplicateQubit {
                    gate: inst.kind.name(),
                    qubit: q,
                });
            }
            seen.push(q);
        }
        for p in &inst.params {
            let finite = match p {
                Param::Value(v) => v.is_finite(),
                Param::Symbol(s) => s.scale.is_finite() && s.offset.is_finite(),
            };
            if !finite {
                return Err(QsoError::invalid("angle", "non-finite gate angle"));
            }
        }
        Ok(())
    }
}

impl Circuit {
    /// Builder helper: append a gate and panic on invalid input.
    ///
    /// The fluent methods (`h`, `cx`, `ry`, …) use this; invalid qubit
    /// indices in hand-written circuits are programming errors. Use
    /// [`Circuit::try_push`] for untrusted input.
    fn push_gate(
        &mut self,
        kind: GateKind,
        targets: &[usize],
        controls: &[usize],
        params: &[Param],
    ) -> &mut Self {
        let inst = Instruction {
            kind,
            targets: targets.to_vec(),
            controls: controls.to_vec(),
            params: params.to_vec(),
            matrix: None,
        };
        if let Err(e) = self.try_push(inst) {
            panic!("invalid gate: {e}");
        }
        self
    }

    fixed_1q! {
        /// Hadamard.
        h => H,
        /// Pauli X.
        x => X,
        /// Pauli Y.
        y => Y,
        /// Pauli Z.
        z => Z,
        /// Phase gate S = √Z.
        s => S,
        /// S†.
        sdg => Sdg,
        /// T = ⁴√Z.
        t => T,
        /// T†.
        tdg => Tdg,
        /// √X.
        sx => SX,
        /// (√X)†.
        sxdg => SXdg,
    }

    rot_1q! {
        /// `RX(θ) = exp(−iθX/2)`.
        rx => RX,
        /// `RY(θ) = exp(−iθY/2)`.
        ry => RY,
        /// `RZ(θ) = exp(−iθZ/2)`.
        rz => RZ,
        /// `P(λ) = diag(1, e^{iλ})`.
        p => P,
    }

    rot_2q! {
        /// `RXX(θ) = exp(−iθ X⊗X/2)`.
        rxx => RXX,
        /// `RYY(θ) = exp(−iθ Y⊗Y/2)`.
        ryy => RYY,
        /// `RZZ(θ) = exp(−iθ Z⊗Z/2)`.
        rzz => RZZ,
    }

    controlled_rot! {
        /// Controlled RX.
        crx => RX,
        /// Controlled RY.
        cry => RY,
        /// Controlled RZ.
        crz => RZ,
        /// Controlled phase.
        cp => P,
    }

    /// Generic single-qubit gate `U(θ, φ, λ)`.
    pub fn u(
        &mut self,
        q: usize,
        theta: impl Into<Param>,
        phi: impl Into<Param>,
        lambda: impl Into<Param>,
    ) -> &mut Self {
        self.push_gate(
            GateKind::U,
            &[q],
            &[],
            &[theta.into(), phi.into(), lambda.into()],
        )
    }

    /// CNOT.
    pub fn cx(&mut self, control: usize, target: usize) -> &mut Self {
        self.push_gate(GateKind::X, &[target], &[control], &[])
    }

    /// Controlled Y.
    pub fn cy(&mut self, control: usize, target: usize) -> &mut Self {
        self.push_gate(GateKind::Y, &[target], &[control], &[])
    }

    /// Controlled Z (symmetric in its qubits).
    pub fn cz(&mut self, control: usize, target: usize) -> &mut Self {
        self.push_gate(GateKind::Z, &[target], &[control], &[])
    }

    /// Controlled Hadamard.
    pub fn ch(&mut self, control: usize, target: usize) -> &mut Self {
        self.push_gate(GateKind::H, &[target], &[control], &[])
    }

    /// SWAP.
    pub fn swap(&mut self, a: usize, b: usize) -> &mut Self {
        self.push_gate(GateKind::Swap, &[a, b], &[], &[])
    }

    /// Toffoli (CCX).
    pub fn ccx(&mut self, c1: usize, c2: usize, target: usize) -> &mut Self {
        self.push_gate(GateKind::X, &[target], &[c1, c2], &[])
    }

    /// Fredkin (controlled SWAP).
    pub fn cswap(&mut self, control: usize, a: usize, b: usize) -> &mut Self {
        self.push_gate(GateKind::Swap, &[a, b], &[control], &[])
    }

    /// Multi-controlled X.
    pub fn mcx(&mut self, controls: &[usize], target: usize) -> &mut Self {
        self.push_gate(GateKind::X, &[target], controls, &[])
    }

    /// Multi-controlled Z (phase −1 on |1…1⟩ of all listed qubits).
    pub fn mcz(&mut self, qubits: &[usize]) -> &mut Self {
        let (target, controls) = qubits.split_last().expect("mcz needs at least one qubit");
        self.push_gate(GateKind::Z, &[*target], controls, &[])
    }

    /// Append `exp(−iθP/2)` for a Pauli string `P` (the convention of
    /// `rx`, `rzz`, …). Single-qubit strings and two-qubit `XX`, `YY`, `ZZ`
    /// become one rotation; longer strings a basis change, a CNOT ladder and
    /// one `rz`. The identity string is only a global phase and adds no
    /// gate. A symbolic angle stays differentiable.
    pub fn pauli_rotation(
        &mut self,
        string: &PauliString,
        angle: impl Into<Param>,
    ) -> Result<&mut Self> {
        let angle = angle.into();
        let finite = match angle {
            Param::Value(v) => v.is_finite(),
            Param::Symbol(p) => p.scale.is_finite() && p.offset.is_finite(),
        };
        if !finite {
            return Err(QsoError::invalid("angle", "non-finite gate angle"));
        }
        let factors: Vec<(usize, Pauli)> = (0..64)
            .map(|q| (q, string.get(q)))
            .filter(|&(_, p)| p != Pauli::I)
            .collect();
        if let Some(&(q, _)) = factors.iter().find(|&&(q, _)| q >= self.n_qubits) {
            return Err(QsoError::QubitOutOfRange {
                qubit: q,
                n_qubits: self.n_qubits,
            });
        }
        let kind = |p: Pauli| match p {
            Pauli::X => GateKind::RX,
            Pauli::Y => GateKind::RY,
            _ => GateKind::RZ,
        };
        match factors[..] {
            [] => {}
            [(q, p)] => {
                self.push_gate(kind(p), &[q], &[], &[angle]);
            }
            [(a, p), (b, r)] if p == r => {
                let kind = match p {
                    Pauli::X => GateKind::RXX,
                    Pauli::Y => GateKind::RYY,
                    _ => GateKind::RZZ,
                };
                self.push_gate(kind, &[a, b], &[], &[angle]);
            }
            _ => {
                // V P V† = Z…Z with V = H (X), H·S† (Y), I (Z).
                for &(q, p) in &factors {
                    match p {
                        Pauli::X => {
                            self.h(q);
                        }
                        Pauli::Y => {
                            self.sdg(q).h(q);
                        }
                        _ => {}
                    }
                }
                for w in factors.windows(2) {
                    self.cx(w[0].0, w[1].0);
                }
                let last = factors[factors.len() - 1].0;
                self.push_gate(GateKind::RZ, &[last], &[], &[angle]);
                for w in factors.windows(2).rev() {
                    self.cx(w[0].0, w[1].0);
                }
                for &(q, p) in &factors {
                    match p {
                        Pauli::X => {
                            self.h(q);
                        }
                        Pauli::Y => {
                            self.h(q).s(q);
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(self)
    }

    /// Append an identity on qubit `q`: it changes nothing, but a device
    /// rule for the gate `"id"` gives it a duration, so that the qubit
    /// waits (relaxes and dephases) for that time.
    ///
    /// # Panics
    /// If `q` is out of range, like the other builder methods.
    pub fn id(&mut self, q: usize) -> &mut Self {
        let one = C64::new(1.0, 0.0);
        let zero = C64::new(0.0, 0.0);
        self.unitary(vec![one, zero, zero, one], &[q], &[])
            .expect("identity on a qubit of the circuit")
    }

    /// Append an arbitrary unitary (row-major, dimension `2^targets.len()`)
    /// with optional controls. `targets[0]` is the most significant local bit.
    pub fn unitary(
        &mut self,
        matrix: Vec<C64>,
        targets: &[usize],
        controls: &[usize],
    ) -> Result<&mut Self> {
        self.try_push(Instruction {
            kind: GateKind::Unitary,
            targets: targets.to_vec(),
            controls: controls.to_vec(),
            params: Vec::new(),
            matrix: Some(Arc::new(matrix)),
        })
    }

    /// Add control qubits to an existing gate kind (e.g. a controlled `U`).
    pub fn controlled(
        &mut self,
        kind: GateKind,
        targets: &[usize],
        controls: &[usize],
        params: &[Param],
    ) -> Result<&mut Self> {
        self.try_push(Instruction {
            kind,
            targets: targets.to_vec(),
            controls: controls.to_vec(),
            params: params.to_vec(),
            matrix: None,
        })
    }

    /// Append all instructions of `other` (parameter indices are shared).
    pub fn append(&mut self, other: &Circuit) -> Result<&mut Self> {
        if other.n_qubits > self.n_qubits {
            return Err(QsoError::DimensionMismatch {
                expected: self.n_qubits,
                actual: other.n_qubits,
            });
        }
        for inst in &other.instructions {
            self.try_push(inst.clone())?;
        }
        self.n_params = self.n_params.max(other.n_params);
        Ok(self)
    }

    /// Append `other` with its qubit `q` mapped to `mapping[q]` of this
    /// circuit (parameter indices are shared).
    pub fn append_mapped(&mut self, other: &Circuit, mapping: &[usize]) -> Result<&mut Self> {
        if mapping.len() != other.n_qubits {
            return Err(QsoError::DimensionMismatch {
                expected: other.n_qubits,
                actual: mapping.len(),
            });
        }
        for inst in &other.instructions {
            let mut mapped = inst.clone();
            mapped.targets.iter_mut().for_each(|q| *q = mapping[*q]);
            mapped.controls.iter_mut().for_each(|q| *q = mapping[*q]);
            self.try_push(mapped)?;
        }
        self.n_params = self.n_params.max(other.n_params);
        Ok(self)
    }

    /// The inverse circuit U†.
    pub fn inverse(&self) -> Circuit {
        Circuit {
            n_qubits: self.n_qubits,
            n_params: self.n_params,
            instructions: self
                .instructions
                .iter()
                .rev()
                .map(Instruction::inverse)
                .collect(),
        }
    }

    /// Replace every symbolic parameter by its value (result has no parameters).
    pub fn bind(&self, theta: &[f64]) -> Result<Circuit> {
        self.check_parameters(theta)?;
        let instructions = self
            .instructions
            .iter()
            .map(|inst| Instruction {
                params: inst
                    .params
                    .iter()
                    .map(|p| Param::Value(p.value(theta)))
                    .collect(),
                ..inst.clone()
            })
            .collect();
        Ok(Circuit {
            n_qubits: self.n_qubits,
            n_params: 0,
            instructions,
        })
    }

    pub(crate) fn check_parameters(&self, theta: &[f64]) -> Result<()> {
        if theta.len() != self.n_params {
            return Err(QsoError::ParameterCount {
                expected: self.n_params,
                actual: theta.len(),
            });
        }
        if theta.iter().any(|t| !t.is_finite()) {
            return Err(QsoError::invalid("theta", "parameters must be finite"));
        }
        Ok(())
    }

    /// Circuit depth (longest chain of gates sharing qubits).
    pub fn depth(&self) -> usize {
        let mut level = vec![0usize; self.n_qubits];
        let mut depth = 0;
        for inst in &self.instructions {
            let d = inst.qubits().map(|q| level[q]).max().unwrap_or(0) + 1;
            for q in inst.qubits() {
                level[q] = d;
            }
            depth = depth.max(d);
        }
        depth
    }

    /// Gate counts keyed by name, with a `c`-prefix per control
    /// (e.g. `cx`, `ccx`, `crz`).
    pub fn gate_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for inst in &self.instructions {
            let name = format!("{}{}", "c".repeat(inst.controls.len()), inst.kind.name());
            *counts.entry(name).or_insert(0) += 1;
        }
        counts
    }

    /// Number of gates acting on two or more qubits.
    pub fn multi_qubit_gate_count(&self) -> usize {
        self.instructions
            .iter()
            .filter(|i| i.targets.len() + i.controls.len() >= 2)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_tracks_parameters_and_depth() {
        let mut c = Circuit::new(3);
        let a = c.new_parameter();
        let b = c.new_parameter();
        c.h(0)
            .cx(0, 1)
            .ry(2, a)
            .rzz(1, 2, 2.0 * b + 0.5)
            .ccx(0, 1, 2);
        assert_eq!(c.num_parameters(), 2);
        assert_eq!(c.len(), 5);
        assert_eq!(c.depth(), 4);
        assert_eq!(c.gate_counts()["ccx"], 1);
        assert_eq!(c.multi_qubit_gate_count(), 3);
        let bound = c.bind(&[0.1, 0.2]).unwrap();
        assert_eq!(bound.num_parameters(), 0);
        assert_eq!(bound.instructions()[3].params[0], Param::Value(0.9));
    }

    #[test]
    fn invalid_gates_are_rejected() {
        let mut c = Circuit::new(2);
        let bad = Instruction {
            kind: GateKind::X,
            targets: vec![1],
            controls: vec![1],
            params: vec![],
            matrix: None,
        };
        assert!(matches!(
            c.try_push(bad),
            Err(QsoError::DuplicateQubit { .. })
        ));
        let out = Instruction {
            kind: GateKind::H,
            targets: vec![2],
            controls: vec![],
            params: vec![],
            matrix: None,
        };
        assert!(matches!(
            c.try_push(out),
            Err(QsoError::QubitOutOfRange { .. })
        ));
        let not_unitary = vec![C64::new(1.0, 0.0); 4];
        assert!(c.unitary(not_unitary, &[0], &[]).is_err());
        assert!(c.bind(&[1.0]).is_err());
    }
}
