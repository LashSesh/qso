//! `Parameter`, `Circuit` and `Program`.

use std::collections::BTreeMap;

use metatron_qso::quantum::{
    Circuit, GateKind, MAX_QUBITS, Param, Parameter, PauliString, PauliSum, Program, Register,
    from_qasm2, record_bitstring, to_qasm2, to_qasm3,
};
use metatron_qso::stabilizer::program::{is_clifford_program, sample_program};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::{
    ComplexMatrixArg, Raise, RealArg, check_finite, check_qubits, invalid, real_vec, square_complex,
};

/// Symbolic circuit parameter `scale·θ[index] + offset`.
///
/// Created by `Circuit.new_parameter()`; supports `2.0 * p`, `p + 0.5`,
/// `p - 0.5`, `p / 2` and `-p`.
#[pyclass(name = "Parameter", module = "metatron_qso", frozen)]
#[derive(Clone, Copy)]
pub struct PyParameter {
    pub inner: Parameter,
}

fn finite(x: f64) -> PyResult<f64> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(invalid("parameter arithmetic needs finite numbers"))
    }
}

fn param(inner: Parameter) -> PyParameter {
    PyParameter { inner }
}

#[pymethods]
impl PyParameter {
    #[getter]
    fn index(&self) -> usize {
        self.inner.index
    }

    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    #[getter]
    fn offset(&self) -> f64 {
        self.inner.offset
    }

    /// Value for the parameter vector `theta`.
    fn value(&self, theta: RealArg<'_>) -> PyResult<f64> {
        let theta = real_vec(&theta);
        if self.inner.index >= theta.len() {
            return Err(invalid(format!(
                "parameter index {} out of range for {} values",
                self.inner.index,
                theta.len()
            )));
        }
        Ok(Param::Symbol(self.inner).value(&theta))
    }

    fn __mul__(&self, other: f64) -> PyResult<Self> {
        Ok(param(self.inner * finite(other)?))
    }

    fn __rmul__(&self, other: f64) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: f64) -> PyResult<Self> {
        if other == 0.0 {
            return Err(invalid("division of a parameter by zero"));
        }
        Ok(param(self.inner * finite(1.0 / other)?))
    }

    fn __add__(&self, other: f64) -> PyResult<Self> {
        Ok(param(self.inner + finite(other)?))
    }

    fn __radd__(&self, other: f64) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: f64) -> PyResult<Self> {
        Ok(param(self.inner - finite(other)?))
    }

    fn __rsub__(&self, other: f64) -> PyResult<Self> {
        Ok(param(-self.inner + finite(other)?))
    }

    fn __neg__(&self) -> Self {
        param(-self.inner)
    }

    fn __repr__(&self) -> String {
        let p = self.inner;
        let mut s = if p.scale == 1.0 {
            format!("theta[{}]", p.index)
        } else {
            format!("{:?}*theta[{}]", p.scale, p.index)
        };
        if p.offset != 0.0 {
            s += &format!(
                " {} {:?}",
                if p.offset < 0.0 { '-' } else { '+' },
                p.offset.abs()
            );
        }
        format!("Parameter({s})")
    }
}

/// A gate angle: a number or a `Parameter`.
pub struct Angle(pub Param);

impl<'a, 'py> FromPyObject<'a, 'py> for Angle {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(p) = obj.cast::<PyParameter>() {
            return Ok(Angle(Param::Symbol(p.get().inner)));
        }
        obj.extract::<f64>()
            .map(|v| Angle(Param::Value(v)))
            .map_err(|_| PyTypeError::new_err("gate angles must be numbers or Parameters"))
    }
}

/// Quantum circuit with symbolic, differentiable parameters.
///
/// Builder methods return the circuit, so calls chain:
/// `Circuit(2).h(0).cx(0, 1)`. Qubit 0 is the least significant bit.
#[pyclass(name = "Circuit", module = "metatron_qso")]
#[derive(Clone)]
pub struct PyCircuit {
    pub inner: Circuit,
}

impl PyCircuit {
    /// Wrap a core circuit, enforcing the register limit.
    pub fn wrap(inner: Circuit) -> PyResult<Self> {
        check_qubits(inner.n_qubits())?;
        Ok(Self { inner })
    }

    fn push(
        &mut self,
        kind: GateKind,
        targets: &[usize],
        controls: &[usize],
        params: &[Param],
    ) -> PyResult<()> {
        self.inner
            .controlled(kind, targets, controls, params)
            .raise()?;
        Ok(())
    }
}

type Slf<'py> = PyRefMut<'py, PyCircuit>;

/// Parse a single Pauli string with coefficient 1, e.g. `"X0 Y1"`.
pub(crate) fn single_pauli_string(label: &str) -> PyResult<PauliString> {
    let sum: PauliSum = label.parse().raise()?;
    match sum.terms() {
        [(c, s)] if *c == 1.0 => Ok(*s),
        _ => Err(invalid(format!(
            "expected a single Pauli string such as 'X0 Y1', got '{label}'"
        ))),
    }
}

/// Parameter vector for `circuit` (`None` → empty).
pub(crate) fn theta_or_empty(theta: Option<RealArg<'_>>) -> Vec<f64> {
    theta.map(|t| real_vec(&t)).unwrap_or_default()
}

#[pymethods]
impl PyCircuit {
    /// Empty circuit on `n_qubits` qubits.
    #[new]
    fn new(n_qubits: usize) -> PyResult<Self> {
        Self::wrap(Circuit::new(n_qubits))
    }

    /// Hadamard.
    fn h(mut slf: Slf<'_>, q: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::H, &[q], &[], &[])?;
        Ok(slf)
    }

    /// Pauli X.
    fn x(mut slf: Slf<'_>, q: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::X, &[q], &[], &[])?;
        Ok(slf)
    }

    /// Pauli Y.
    fn y(mut slf: Slf<'_>, q: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::Y, &[q], &[], &[])?;
        Ok(slf)
    }

    /// Pauli Z.
    fn z(mut slf: Slf<'_>, q: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::Z, &[q], &[], &[])?;
        Ok(slf)
    }

    /// Phase gate S = diag(1, i).
    fn s(mut slf: Slf<'_>, q: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::S, &[q], &[], &[])?;
        Ok(slf)
    }

    /// S†.
    fn sdg(mut slf: Slf<'_>, q: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::Sdg, &[q], &[], &[])?;
        Ok(slf)
    }

    /// T = diag(1, e^{iπ/4}).
    fn t(mut slf: Slf<'_>, q: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::T, &[q], &[], &[])?;
        Ok(slf)
    }

    /// T†.
    fn tdg(mut slf: Slf<'_>, q: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::Tdg, &[q], &[], &[])?;
        Ok(slf)
    }

    /// √X.
    fn sx(mut slf: Slf<'_>, q: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::SX, &[q], &[], &[])?;
        Ok(slf)
    }

    /// (√X)†.
    fn sxdg(mut slf: Slf<'_>, q: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::SXdg, &[q], &[], &[])?;
        Ok(slf)
    }

    /// RX(θ) = exp(−iθX/2).
    fn rx(mut slf: Slf<'_>, q: usize, angle: Angle) -> PyResult<Slf<'_>> {
        slf.push(GateKind::RX, &[q], &[], &[angle.0])?;
        Ok(slf)
    }

    /// RY(θ) = exp(−iθY/2).
    fn ry(mut slf: Slf<'_>, q: usize, angle: Angle) -> PyResult<Slf<'_>> {
        slf.push(GateKind::RY, &[q], &[], &[angle.0])?;
        Ok(slf)
    }

    /// RZ(θ) = exp(−iθZ/2).
    fn rz(mut slf: Slf<'_>, q: usize, angle: Angle) -> PyResult<Slf<'_>> {
        slf.push(GateKind::RZ, &[q], &[], &[angle.0])?;
        Ok(slf)
    }

    /// Phase gate P(λ) = diag(1, e^{iλ}).
    fn p(mut slf: Slf<'_>, q: usize, angle: Angle) -> PyResult<Slf<'_>> {
        slf.push(GateKind::P, &[q], &[], &[angle.0])?;
        Ok(slf)
    }

    /// Generic single-qubit gate U(θ, φ, λ) (OpenQASM convention).
    fn u(mut slf: Slf<'_>, q: usize, theta: Angle, phi: Angle, lam: Angle) -> PyResult<Slf<'_>> {
        slf.push(GateKind::U, &[q], &[], &[theta.0, phi.0, lam.0])?;
        Ok(slf)
    }

    /// RXX(θ) = exp(−iθ X⊗X/2).
    fn rxx(mut slf: Slf<'_>, a: usize, b: usize, angle: Angle) -> PyResult<Slf<'_>> {
        slf.push(GateKind::RXX, &[a, b], &[], &[angle.0])?;
        Ok(slf)
    }

    /// RYY(θ) = exp(−iθ Y⊗Y/2).
    fn ryy(mut slf: Slf<'_>, a: usize, b: usize, angle: Angle) -> PyResult<Slf<'_>> {
        slf.push(GateKind::RYY, &[a, b], &[], &[angle.0])?;
        Ok(slf)
    }

    /// RZZ(θ) = exp(−iθ Z⊗Z/2).
    fn rzz(mut slf: Slf<'_>, a: usize, b: usize, angle: Angle) -> PyResult<Slf<'_>> {
        slf.push(GateKind::RZZ, &[a, b], &[], &[angle.0])?;
        Ok(slf)
    }

    /// exp(−iθP/2) for a Pauli string label such as `"X0 Y1 Z3"`.
    fn pauli_rotation<'py>(mut slf: Slf<'py>, pauli: &str, angle: Angle) -> PyResult<Slf<'py>> {
        let string = single_pauli_string(pauli)?;
        slf.inner.pauli_rotation(&string, angle.0).raise()?;
        Ok(slf)
    }

    /// Controlled X (CNOT).
    fn cx(mut slf: Slf<'_>, control: usize, target: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::X, &[target], &[control], &[])?;
        Ok(slf)
    }

    /// Controlled Y.
    fn cy(mut slf: Slf<'_>, control: usize, target: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::Y, &[target], &[control], &[])?;
        Ok(slf)
    }

    /// Controlled Z.
    fn cz(mut slf: Slf<'_>, control: usize, target: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::Z, &[target], &[control], &[])?;
        Ok(slf)
    }

    /// Controlled Hadamard.
    fn ch(mut slf: Slf<'_>, control: usize, target: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::H, &[target], &[control], &[])?;
        Ok(slf)
    }

    /// SWAP.
    fn swap(mut slf: Slf<'_>, a: usize, b: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::Swap, &[a, b], &[], &[])?;
        Ok(slf)
    }

    /// Toffoli.
    fn ccx(mut slf: Slf<'_>, c1: usize, c2: usize, target: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::X, &[target], &[c1, c2], &[])?;
        Ok(slf)
    }

    /// Fredkin (controlled SWAP).
    fn cswap(mut slf: Slf<'_>, control: usize, a: usize, b: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::Swap, &[a, b], &[control], &[])?;
        Ok(slf)
    }

    /// X on `target` controlled by all `controls`.
    fn mcx(mut slf: Slf<'_>, controls: Vec<usize>, target: usize) -> PyResult<Slf<'_>> {
        slf.push(GateKind::X, &[target], &controls, &[])?;
        Ok(slf)
    }

    /// Multi-controlled Z on `qubits` (symmetric in its qubits).
    fn mcz(mut slf: Slf<'_>, qubits: Vec<usize>) -> PyResult<Slf<'_>> {
        let (target, controls) = qubits
            .split_last()
            .ok_or_else(|| invalid("mcz needs at least one qubit"))?;
        slf.push(GateKind::Z, &[*target], controls, &[])?;
        Ok(slf)
    }

    /// Controlled RX.
    fn crx(mut slf: Slf<'_>, control: usize, target: usize, angle: Angle) -> PyResult<Slf<'_>> {
        slf.push(GateKind::RX, &[target], &[control], &[angle.0])?;
        Ok(slf)
    }

    /// Controlled RY.
    fn cry(mut slf: Slf<'_>, control: usize, target: usize, angle: Angle) -> PyResult<Slf<'_>> {
        slf.push(GateKind::RY, &[target], &[control], &[angle.0])?;
        Ok(slf)
    }

    /// Controlled RZ.
    fn crz(mut slf: Slf<'_>, control: usize, target: usize, angle: Angle) -> PyResult<Slf<'_>> {
        slf.push(GateKind::RZ, &[target], &[control], &[angle.0])?;
        Ok(slf)
    }

    /// Controlled phase.
    fn cp(mut slf: Slf<'_>, control: usize, target: usize, angle: Angle) -> PyResult<Slf<'_>> {
        slf.push(GateKind::P, &[target], &[control], &[angle.0])?;
        Ok(slf)
    }

    /// Custom unitary on 1–6 `targets` with optional `controls`.
    ///
    /// `matrix` is `2^k × 2^k` (checked for unitarity); `targets[0]` is the
    /// most significant bit of its row/column index.
    #[pyo3(signature = (matrix, targets, controls = Vec::new()))]
    fn unitary<'py>(
        mut slf: Slf<'py>,
        matrix: ComplexMatrixArg<'_>,
        targets: Vec<usize>,
        controls: Vec<usize>,
    ) -> PyResult<Slf<'py>> {
        let (_, data) = square_complex("matrix", &matrix)?;
        if targets.is_empty() {
            return Err(invalid("unitary needs at least one target"));
        }
        slf.inner.unitary(data, &targets, &controls).raise()?;
        Ok(slf)
    }

    /// Fresh symbolic parameter.
    fn new_parameter(&mut self) -> PyParameter {
        param(self.inner.new_parameter())
    }

    /// `n` fresh symbolic parameters.
    fn new_parameters(&mut self, n: usize) -> Vec<PyParameter> {
        self.inner
            .new_parameters(n)
            .into_iter()
            .map(param)
            .collect()
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    #[getter]
    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    #[getter]
    fn depth(&self) -> usize {
        self.inner.depth()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Gate counts by name; controls are prefixed with `c` (`ccx`, `cry`…).
    fn gate_counts(&self) -> BTreeMap<String, usize> {
        self.inner.gate_counts()
    }

    /// Instructions as `(name, targets, controls, params)` tuples; params
    /// are floats or `Parameter`s.
    fn instructions<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        self.inner
            .instructions()
            .iter()
            .map(|inst| {
                let params = inst
                    .params
                    .iter()
                    .map(|p| -> PyResult<Bound<'py, PyAny>> {
                        Ok(match *p {
                            Param::Value(v) => v.into_pyobject(py)?.into_any(),
                            Param::Symbol(s) => Bound::new(py, param(s))?.into_any(),
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                (
                    inst.kind.name(),
                    inst.targets.clone(),
                    inst.controls.clone(),
                    params,
                )
                    .into_pyobject(py)
            })
            .collect()
    }

    /// Adjoint circuit (gates reversed and inverted).
    fn inverse(&self) -> Self {
        Self {
            inner: self.inner.inverse(),
        }
    }

    /// Copy with every parameter replaced by its value for `theta`.
    fn bind(&self, theta: RealArg<'_>) -> PyResult<Self> {
        let theta = real_vec(&theta);
        check_finite("theta", &theta)?;
        Ok(Self {
            inner: self.inner.bind(&theta).raise()?,
        })
    }

    /// Append the gates of `other` (on the first `other.n_qubits` qubits).
    fn append<'py>(mut slf: Slf<'py>, other: &PyCircuit) -> PyResult<Slf<'py>> {
        slf.inner.append(&other.inner).raise()?;
        Ok(slf)
    }

    /// Append `other` with its qubit `k` mapped to `qubits[k]`.
    fn append_mapped<'py>(
        mut slf: Slf<'py>,
        other: &PyCircuit,
        qubits: Vec<usize>,
    ) -> PyResult<Slf<'py>> {
        slf.inner.append_mapped(&other.inner, &qubits).raise()?;
        Ok(slf)
    }

    /// OpenQASM 2.0 text with parameters `theta` bound.
    #[pyo3(signature = (theta = None, measure = false))]
    fn to_qasm(&self, theta: Option<RealArg<'_>>, measure: bool) -> PyResult<String> {
        to_qasm2(&self.inner, &theta_or_empty(theta), measure).raise()
    }

    /// OpenQASM 3.0 text with parameters `theta` bound.
    #[pyo3(signature = (theta = None, measure = false))]
    fn to_qasm3(&self, theta: Option<RealArg<'_>>, measure: bool) -> PyResult<String> {
        to_qasm3(&self.inner, &theta_or_empty(theta), measure).raise()
    }

    /// Parse an OpenQASM 2.0 program; measurements are dropped. Use
    /// `Program.from_qasm` for OpenQASM 3 and for programs with mid-circuit
    /// measurements, resets or conditions.
    #[staticmethod]
    fn from_qasm(text: &str) -> PyResult<Self> {
        let c = from_qasm2(text).raise()?;
        if c.n_qubits() > MAX_QUBITS {
            return Err(invalid(format!(
                "program declares {} qubits, at most {MAX_QUBITS} are supported",
                c.n_qubits()
            )));
        }
        Self::wrap(c)
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Circuit(n_qubits={}, gates={}, parameters={}, depth={})",
            self.inner.n_qubits(),
            self.inner.len(),
            self.inner.num_parameters(),
            self.inner.depth()
        )
    }
}

/// A quantum program: gates, measurements into classical bits, resets and
/// gates conditioned on bits, read from OpenQASM 2 or 3.
///
/// Bit strings put bit 0 on the right, like `Circuit` puts qubit 0.
#[pyclass(name = "Program", module = "metatron_qso", frozen)]
pub struct PyProgram {
    inner: Program,
}

fn registers<'py>(py: Python<'py>, regs: &[Register]) -> PyResult<Vec<Bound<'py, PyDict>>> {
    regs.iter()
        .map(|r| {
            let d = PyDict::new(py);
            d.set_item("name", &r.name)?;
            d.set_item("offset", r.offset)?;
            d.set_item("size", r.size)?;
            Ok(d)
        })
        .collect()
}

#[pymethods]
impl PyProgram {
    /// Parse an OpenQASM 2 or 3 program (the version is read from its
    /// header).
    #[staticmethod]
    fn from_qasm(py: Python<'_>, text: &str) -> PyResult<Self> {
        let inner = py.detach(|| Program::from_qasm(text)).raise()?;
        Ok(Self { inner })
    }

    /// The program of `circuit` with parameters `theta` bound, measuring
    /// every qubit into the bit of the same index if `measure`.
    #[staticmethod]
    #[pyo3(signature = (circuit, theta = None, measure = false))]
    fn from_circuit(
        circuit: &PyCircuit,
        theta: Option<RealArg<'_>>,
        measure: bool,
    ) -> PyResult<Self> {
        let inner =
            Program::from_circuit(&circuit.inner, &theta_or_empty(theta), measure).raise()?;
        Ok(Self { inner })
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.inner.n_qubits()
    }

    #[getter]
    fn n_bits(&self) -> usize {
        self.inner.n_bits()
    }

    /// Whether the program has only gates.
    #[getter]
    fn is_unitary(&self) -> bool {
        self.inner.is_unitary()
    }

    /// Qubit registers as `{"name", "offset", "size"}` dictionaries.
    #[getter]
    fn qubit_registers<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        registers(py, self.inner.qubit_registers())
    }

    /// Bit registers as `{"name", "offset", "size"}` dictionaries.
    #[getter]
    fn bit_registers<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        registers(py, self.inner.bit_registers())
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Exact probabilities of the classical records, `{bitstring: p}`
    /// (records with probability below 1e-15 are left out). Mid-circuit
    /// measurements are followed branch by branch.
    fn probabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let p = &self.inner;
        let dist = py.detach(|| p.probabilities()).raise()?;
        let d = PyDict::new(py);
        for (record, q) in dist {
            d.set_item(record_bitstring(record, p.n_bits()), q)?;
        }
        Ok(d)
    }

    /// Whether every gate is a Clifford gate (the program then runs on the
    /// stabilizer simulator).
    #[getter]
    fn is_clifford(&self) -> bool {
        is_clifford_program(&self.inner)
    }

    /// `shots` (at most 10⁷) seeded samples of the classical record,
    /// `{bitstring: count}`. `method` is `"statevector"`, `"stabilizer"`
    /// (Clifford programs, up to 16384 qubits) or `"auto"`: the stabilizer
    /// simulator for Clifford programs too large for a state vector.
    #[pyo3(signature = (shots, seed = 0, method = "auto"))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
        seed: u64,
        method: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        if shots > 10_000_000 {
            return Err(invalid("shots must be at most 10⁷"));
        }
        let p = &self.inner;
        let tableau = match method {
            "statevector" => false,
            "stabilizer" => true,
            "auto" => p.n_qubits() > MAX_QUBITS && is_clifford_program(p),
            other => {
                return Err(invalid(format!(
                    "unknown method '{other}' (use auto, statevector or stabilizer)"
                )));
            }
        };
        let counts = if tableau {
            py.detach(|| sample_program(p, shots, seed)).raise()?
        } else {
            py.detach(|| p.sample(shots, seed)).raise()?
        };
        let d = PyDict::new(py);
        for (record, n) in counts {
            d.set_item(record_bitstring(record, p.n_bits()), n)?;
        }
        Ok(d)
    }

    /// The gates as a `Circuit`; a `ValueError` unless the program is
    /// unitary.
    fn to_circuit(&self) -> PyResult<PyCircuit> {
        PyCircuit::wrap(self.inner.to_circuit().raise()?)
    }

    /// OpenQASM 3.0 text of the program.
    fn to_qasm3(&self) -> PyResult<String> {
        self.inner.to_qasm3().raise()
    }

    /// OpenQASM 2.0 text of the program; a `ValueError` if it needs
    /// OpenQASM 3 (for example a condition on part of a register).
    fn to_qasm2(&self) -> PyResult<String> {
        self.inner.to_qasm2().raise()
    }

    fn __repr__(&self) -> String {
        format!(
            "Program(n_qubits={}, n_bits={}, operations={})",
            self.inner.n_qubits(),
            self.inner.n_bits(),
            self.inner.len()
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyParameter>()?;
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyProgram>()?;
    Ok(())
}
