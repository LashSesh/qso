//! Fermionic operators, qubit mappings, molecular Hamiltonians, UCC
//! ansätze and ADAPT-VQE.

use metatron_qso::C64;
use metatron_qso::chemistry::adapt::{AdaptOptions, adapt_vqe_molecule};
use metatron_qso::chemistry::{
    FermionOperator, Mapping, MolecularIntegrals, QubitEncoding, k_upccgsd, uccsd,
};
use metatron_qso::quantum::QubitOperator;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::circuit::PyCircuit;
use crate::state::{PyPauliSum, wrap_sum};
use crate::{Raise, invalid, pydict};

fn encoding(mapping: &str, two_qubit_reduction: bool) -> PyResult<QubitEncoding> {
    let mapping = Mapping::from_name(mapping).raise()?;
    if two_qubit_reduction && mapping != Mapping::Parity {
        return Err(invalid("two_qubit_reduction requires mapping='parity'"));
    }
    Ok(QubitEncoding {
        mapping,
        two_qubit_reduction,
    })
}

/// Linear combination of Pauli strings with complex coefficients.
#[pyclass(name = "QubitOperator", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyQubitOperator {
    pub inner: QubitOperator,
}

#[pymethods]
impl PyQubitOperator {
    /// From a `PauliSum` (real coefficients).
    #[new]
    fn new(sum: &PyPauliSum) -> Self {
        Self {
            inner: (&sum.inner).into(),
        }
    }

    /// `(coefficient, label)` pairs.
    fn terms(&self) -> Vec<(C64, String)> {
        self.inner
            .terms()
            .map(|(c, s)| (c, s.to_string()))
            .collect()
    }

    #[getter]
    fn num_terms(&self) -> usize {
        self.inner.num_terms()
    }

    #[pyo3(signature = (tol = 1e-12))]
    fn is_hermitian(&self, tol: f64) -> bool {
        self.inner.is_hermitian(tol)
    }

    /// The operator as a `PauliSum`; fails if it is not Hermitian.
    #[pyo3(signature = (tol = 1e-12))]
    fn to_pauli_sum(&self, tol: f64) -> PyResult<PyPauliSum> {
        self.inner.to_pauli_sum(tol).raise().map(wrap_sum)
    }

    fn adjoint(&self) -> Self {
        Self {
            inner: self.inner.adjoint(),
        }
    }

    /// `[self, other]`.
    fn commutator(&self, other: &PyQubitOperator) -> Self {
        Self {
            inner: self.inner.commutator(&other.inner),
        }
    }

    /// Sum of the absolute coefficients.
    fn one_norm(&self) -> f64 {
        self.inner.one_norm()
    }

    fn __add__(&self, other: &PyQubitOperator) -> Self {
        Self {
            inner: &self.inner + &other.inner,
        }
    }

    fn __sub__(&self, other: &PyQubitOperator) -> Self {
        Self {
            inner: &self.inner - &other.inner,
        }
    }

    fn __mul__(&self, other: &PyQubitOperator) -> Self {
        Self {
            inner: &self.inner * &other.inner,
        }
    }

    fn __eq__(&self, other: &PyQubitOperator) -> bool {
        self.inner == other.inner
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("QubitOperator({} terms)", self.inner.num_terms())
    }
}

/// Fermionic operator, parsed from text such as `"0.5 [2^ 0] + 0.5 [0^ 2]"`
/// (`p^` creates, `p` annihilates a fermion in mode `p`).
#[pyclass(name = "FermionOperator", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyFermionOperator {
    pub inner: FermionOperator,
}

#[pymethods]
impl PyFermionOperator {
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        text.parse().raise().map(|inner| Self { inner })
    }

    /// `(coefficient, [(mode, is_creation), …])` pairs.
    fn terms(&self) -> Vec<(C64, Vec<(usize, bool)>)> {
        self.inner
            .terms()
            .map(|(c, ops)| (c, ops.iter().map(|l| (l.mode, l.dagger)).collect()))
            .collect()
    }

    #[getter]
    fn num_terms(&self) -> usize {
        self.inner.num_terms()
    }

    #[getter]
    fn num_modes(&self) -> usize {
        self.inner.num_modes()
    }

    fn adjoint(&self) -> Self {
        Self {
            inner: self.inner.adjoint(),
        }
    }

    /// Canonical form: creators first, indices descending.
    fn normal_ordered(&self) -> Self {
        Self {
            inner: self.inner.normal_ordered(),
        }
    }

    #[pyo3(signature = (tol = 1e-12))]
    fn is_hermitian(&self, tol: f64) -> bool {
        self.inner.is_hermitian(tol)
    }

    /// Map to qubits: `mapping` is "jordan_wigner", "parity" or
    /// "bravyi_kitaev"; `n_modes` defaults to `num_modes`.
    #[pyo3(signature = (mapping = "jordan_wigner", n_modes = None))]
    fn to_qubit(&self, mapping: &str, n_modes: Option<usize>) -> PyResult<PyQubitOperator> {
        let m = Mapping::from_name(mapping).raise()?;
        let n = n_modes.unwrap_or_else(|| self.inner.num_modes().max(1));
        m.map(&self.inner, n)
            .raise()
            .map(|inner| PyQubitOperator { inner })
    }

    fn __add__(&self, other: &PyFermionOperator) -> Self {
        Self {
            inner: &self.inner + &other.inner,
        }
    }

    fn __sub__(&self, other: &PyFermionOperator) -> Self {
        Self {
            inner: &self.inner - &other.inner,
        }
    }

    fn __mul__(&self, other: &PyFermionOperator) -> Self {
        Self {
            inner: &self.inner * &other.inner,
        }
    }

    fn __eq__(&self, other: &PyFermionOperator) -> bool {
        self.inner == other.inner
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "FermionOperator({} terms, {} modes)",
            self.inner.num_terms(),
            self.inner.num_modes()
        )
    }
}

/// One- and two-electron integrals of a molecule (FCIDUMP).
#[pyclass(name = "MolecularIntegrals", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyMolecularIntegrals {
    pub inner: MolecularIntegrals,
}

#[pymethods]
impl PyMolecularIntegrals {
    /// Parse the text of an FCIDUMP file.
    #[staticmethod]
    fn from_fcidump(text: &str) -> PyResult<Self> {
        MolecularIntegrals::from_fcidump(text)
            .raise()
            .map(|inner| Self { inner })
    }

    /// Read an FCIDUMP file.
    #[staticmethod]
    fn from_file(path: std::path::PathBuf) -> PyResult<Self> {
        let text = std::fs::read_to_string(&path)
            .map_err(|e| invalid(format!("cannot read {}: {e}", path.display())))?;
        Self::from_fcidump(&text)
    }

    fn to_fcidump(&self) -> String {
        self.inner.to_fcidump()
    }

    #[getter]
    fn n_orbitals(&self) -> usize {
        self.inner.n_orbitals()
    }

    #[getter]
    fn n_spin_orbitals(&self) -> usize {
        self.inner.n_spin_orbitals()
    }

    #[getter]
    fn n_alpha(&self) -> usize {
        self.inner.n_alpha()
    }

    #[getter]
    fn n_beta(&self) -> usize {
        self.inner.n_beta()
    }

    #[getter]
    fn core_energy(&self) -> f64 {
        self.inner.core_energy()
    }

    fn one_body(&self, p: usize, q: usize) -> PyResult<f64> {
        let m = self.inner.n_orbitals();
        if p >= m || q >= m {
            return Err(invalid("orbital index out of range"));
        }
        Ok(self.inner.one_body(p, q))
    }

    fn two_body(&self, p: usize, q: usize, r: usize, s: usize) -> PyResult<f64> {
        let m = self.inner.n_orbitals();
        if [p, q, r, s].iter().any(|&i| i >= m) {
            return Err(invalid("orbital index out of range"));
        }
        Ok(self.inner.two_body(p, q, r, s))
    }

    fn fermion_hamiltonian(&self) -> PyFermionOperator {
        PyFermionOperator {
            inner: self.inner.fermion_hamiltonian(),
        }
    }

    /// Qubit Hamiltonian (constant term = core energy).
    #[pyo3(signature = (mapping = "jordan_wigner", two_qubit_reduction = false))]
    fn qubit_hamiltonian(
        &self,
        py: Python<'_>,
        mapping: &str,
        two_qubit_reduction: bool,
    ) -> PyResult<PyPauliSum> {
        let e = encoding(mapping, two_qubit_reduction)?;
        let m = &self.inner;
        py.detach(|| m.qubit_hamiltonian(e)).raise().map(wrap_sum)
    }

    fn hartree_fock_energy(&self) -> f64 {
        self.inner.hartree_fock_energy()
    }

    /// X gates preparing the Hartree–Fock determinant.
    #[pyo3(signature = (mapping = "jordan_wigner", two_qubit_reduction = false))]
    fn hartree_fock_circuit(
        &self,
        mapping: &str,
        two_qubit_reduction: bool,
    ) -> PyResult<PyCircuit> {
        let e = encoding(mapping, two_qubit_reduction)?;
        PyCircuit::wrap(self.inner.hartree_fock_circuit(e).raise()?)
    }

    /// The lowest `n_roots` full-CI energies (core energy included).
    #[pyo3(signature = (n_roots = 1))]
    fn fci_energies(&self, py: Python<'_>, n_roots: usize) -> PyResult<Vec<f64>> {
        let m = &self.inner;
        py.detach(|| m.fci_energies(n_roots)).raise()
    }

    fn __repr__(&self) -> String {
        format!(
            "MolecularIntegrals(n_orbitals={}, n_alpha={}, n_beta={})",
            self.inner.n_orbitals(),
            self.inner.n_alpha(),
            self.inner.n_beta()
        )
    }
}

/// UCCSD circuit (Hartree–Fock state, then one parameter per excitation).
#[pyfunction(name = "uccsd")]
#[pyo3(signature = (molecule, mapping = "jordan_wigner", two_qubit_reduction = false))]
fn py_uccsd(
    molecule: &PyMolecularIntegrals,
    mapping: &str,
    two_qubit_reduction: bool,
) -> PyResult<PyCircuit> {
    let e = encoding(mapping, two_qubit_reduction)?;
    PyCircuit::wrap(uccsd(&molecule.inner, e).raise()?)
}

/// k-UpCCGSD circuit with `k` layers.
#[pyfunction(name = "k_upccgsd")]
#[pyo3(signature = (molecule, k = 1, mapping = "jordan_wigner", two_qubit_reduction = false))]
fn py_k_upccgsd(
    molecule: &PyMolecularIntegrals,
    k: usize,
    mapping: &str,
    two_qubit_reduction: bool,
) -> PyResult<PyCircuit> {
    let e = encoding(mapping, two_qubit_reduction)?;
    PyCircuit::wrap(k_upccgsd(&molecule.inner, k, e).raise()?)
}

/// ADAPT-VQE from the Hartree–Fock state with the UCCSD pool (or the
/// generalised pair-double and single pool).
#[pyfunction]
#[pyo3(signature = (
    molecule, mapping = "jordan_wigner", two_qubit_reduction = false, generalized = false,
    gradient_tolerance = 1e-3, max_operators = 40
))]
fn adapt_vqe<'py>(
    py: Python<'py>,
    molecule: &PyMolecularIntegrals,
    mapping: &str,
    two_qubit_reduction: bool,
    generalized: bool,
    gradient_tolerance: f64,
    max_operators: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let e = encoding(mapping, two_qubit_reduction)?;
    if !(gradient_tolerance >= 0.0 && gradient_tolerance.is_finite()) {
        return Err(invalid(
            "gradient_tolerance must be finite and non-negative",
        ));
    }
    let options = AdaptOptions::default()
        .with_gradient_tolerance(gradient_tolerance)
        .with_max_operators(max_operators);
    let m = &molecule.inner;
    let r = py
        .detach(|| adapt_vqe_molecule(m, e, generalized, &options))
        .raise()?;
    Ok(pydict!(py, {
        "energy" => r.energy,
        "parameters" => r.parameters.into_pyarray(py),
        "operators" => r.operators,
        "energies" => r.energies.into_pyarray(py),
        "max_gradients" => r.max_gradients.into_pyarray(py),
        "converged" => r.converged,
        "circuit" => PyCircuit::wrap(r.circuit)?,
    }))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQubitOperator>()?;
    m.add_class::<PyFermionOperator>()?;
    m.add_class::<PyMolecularIntegrals>()?;
    m.add_function(wrap_pyfunction!(py_uccsd, m)?)?;
    m.add_function(wrap_pyfunction!(py_k_upccgsd, m)?)?;
    m.add_function(wrap_pyfunction!(adapt_vqe, m)?)?;
    Ok(())
}
