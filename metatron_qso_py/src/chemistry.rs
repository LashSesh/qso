//! Fermionic operators, qubit mappings, molecular Hamiltonians, UCC
//! ansätze and ADAPT-VQE.

use metatron_qso::C64;
use metatron_qso::chemistry::adapt::{AdaptOptions, adapt_vqe_molecule};
use metatron_qso::chemistry::{
    AoIntegrals, Basis, FermionOperator, Guess, Harmonics, Mapping, MolecularIntegrals, Molecule,
    QubitEncoding, ScfOptions, ScfResult, k_upccgsd, rhf, rohf, uccsd, uhf,
};
use metatron_qso::quantum::QubitOperator;
use numpy::ndarray::{Array2, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray4};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::circuit::PyCircuit;
use crate::state::{PyPauliSum, wrap_sum};
use crate::{Raise, invalid, pydict, real_array, rmatrix};

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
    fn normal_ordered(&self, py: Python<'_>) -> PyResult<Self> {
        let f = &self.inner;
        py.detach(|| f.normal_ordered())
            .raise()
            .map(|inner| Self { inner })
    }

    #[pyo3(signature = (tol = 1e-12))]
    fn is_hermitian(&self, py: Python<'_>, tol: f64) -> PyResult<bool> {
        let f = &self.inner;
        py.detach(|| f.is_hermitian(tol)).raise()
    }

    /// Map to qubits: `mapping` is "jordan_wigner", "parity" or
    /// "bravyi_kitaev"; `n_modes` defaults to `num_modes`.
    #[pyo3(signature = (mapping = "jordan_wigner", n_modes = None))]
    fn to_qubit(
        &self,
        py: Python<'_>,
        mapping: &str,
        n_modes: Option<usize>,
    ) -> PyResult<PyQubitOperator> {
        let m = Mapping::from_name(mapping).raise()?;
        let n = n_modes.unwrap_or_else(|| self.inner.num_modes().max(1));
        let f = &self.inner;
        py.detach(|| m.map(f, n))
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

    fn __mul__(&self, py: Python<'_>, other: &PyFermionOperator) -> PyResult<Self> {
        let terms = self
            .inner
            .num_terms()
            .saturating_mul(other.inner.num_terms());
        if terms > 1 << 22 {
            return Err(invalid(format!(
                "the product would have up to {terms} terms"
            )));
        }
        let (a, b) = (&self.inner, &other.inner);
        Ok(Self {
            inner: py.detach(|| a * b),
        })
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
    py: Python<'_>,
    molecule: &PyMolecularIntegrals,
    mapping: &str,
    two_qubit_reduction: bool,
) -> PyResult<PyCircuit> {
    let e = encoding(mapping, two_qubit_reduction)?;
    let m = &molecule.inner;
    PyCircuit::wrap(py.detach(|| uccsd(m, e)).raise()?)
}

/// k-UpCCGSD circuit with `k` layers.
#[pyfunction(name = "k_upccgsd")]
#[pyo3(signature = (molecule, k = 1, mapping = "jordan_wigner", two_qubit_reduction = false))]
fn py_k_upccgsd(
    py: Python<'_>,
    molecule: &PyMolecularIntegrals,
    k: usize,
    mapping: &str,
    two_qubit_reduction: bool,
) -> PyResult<PyCircuit> {
    let e = encoding(mapping, two_qubit_reduction)?;
    let m = &molecule.inner;
    PyCircuit::wrap(py.detach(|| k_upccgsd(m, k, e)).raise()?)
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
    m.add_class::<PyMolecule>()?;
    m.add_class::<PyAoIntegrals>()?;
    m.add_class::<PyHartreeFock>()?;
    m.add_function(wrap_pyfunction!(hartree_fock, m)?)?;
    m.add_function(wrap_pyfunction!(py_uccsd, m)?)?;
    m.add_function(wrap_pyfunction!(py_k_upccgsd, m)?)?;
    m.add_function(wrap_pyfunction!(adapt_vqe, m)?)?;
    Ok(())
}

/// A molecule: nuclei, total charge and spin multiplicity `2S + 1`.
#[pyclass(name = "Molecule", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyMolecule {
    inner: Molecule,
}

#[pymethods]
impl PyMolecule {
    /// From `"H 0 0 0; H 0 0 0.735"` (element and position in ångström
    /// per atom, separated by `;` or new lines), H to Ar.
    #[new]
    #[pyo3(signature = (atoms, charge = 0, multiplicity = 1))]
    fn new(atoms: &str, charge: i32, multiplicity: u32) -> PyResult<Self> {
        Molecule::from_atoms(atoms, charge, multiplicity)
            .raise()
            .map(|inner| Self { inner })
    }

    /// Atomic numbers.
    #[getter]
    fn charges(&self) -> Vec<u32> {
        self.inner.charges().to_vec()
    }

    /// Positions in bohr, one row per atom.
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let p = self.inner.positions();
        Array2::from_shape_fn((p.len(), 3), |(i, k)| p[i][k]).into_pyarray(py)
    }

    #[getter]
    fn charge(&self) -> i32 {
        self.inner.charge()
    }

    #[getter]
    fn multiplicity(&self) -> u32 {
        self.inner.multiplicity()
    }

    #[getter]
    fn n_electrons(&self) -> usize {
        self.inner.n_electrons()
    }

    /// `(n_alpha, n_beta)`.
    #[getter]
    fn n_alpha_beta(&self) -> (usize, usize) {
        self.inner.n_alpha_beta()
    }

    /// Nuclear repulsion energy (hartree).
    fn nuclear_repulsion(&self) -> f64 {
        self.inner.nuclear_repulsion()
    }

    fn __repr__(&self) -> String {
        format!(
            "Molecule(atoms={}, charge={}, multiplicity={})",
            self.inner.charges().len(),
            self.inner.charge(),
            self.inner.multiplicity()
        )
    }
}

/// Overlap, kinetic, nuclear-attraction and electron-repulsion integrals
/// over the basis functions of a molecule.
#[pyclass(name = "AoIntegrals", module = "metatron_qso", frozen)]
#[derive(Clone)]
pub struct PyAoIntegrals {
    inner: AoIntegrals,
}

fn matrix<'py>(
    py: Python<'py>,
    n: usize,
    f: impl Fn(usize, usize) -> f64,
) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((n, n), |(i, j)| f(i, j)).into_pyarray(py)
}

#[pymethods]
impl PyAoIntegrals {
    /// The integrals of `molecule` in `basis` (`"sto-3g"`, `"6-31g"`,
    /// `"6-31g*"` or `"cc-pvdz"`), with `"spherical"` (default, as PySCF)
    /// or `"cartesian"` d shells.
    #[new]
    #[pyo3(signature = (molecule, basis = "sto-3g", harmonics = "spherical"))]
    fn new(py: Python<'_>, molecule: &PyMolecule, basis: &str, harmonics: &str) -> PyResult<Self> {
        let basis = Basis::from_name(basis).raise()?;
        let harmonics = parse_harmonics(harmonics)?;
        let mol = &molecule.inner;
        py.detach(|| AoIntegrals::compute_with(mol, basis, harmonics))
            .raise()
            .map(|inner| Self { inner })
    }

    /// The basis set's name.
    #[getter]
    fn basis(&self) -> &'static str {
        self.inner.basis().name()
    }

    /// `"spherical"` or `"cartesian"`.
    #[getter]
    fn harmonics(&self) -> &'static str {
        match self.inner.harmonics() {
            Harmonics::Cartesian => "cartesian",
            Harmonics::Spherical => "spherical",
        }
    }

    #[getter]
    fn n_functions(&self) -> usize {
        self.inner.n_functions()
    }

    #[getter]
    fn overlap<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix(py, self.inner.n_functions(), |i, j| {
            self.inner.overlap(i, j)
        })
    }

    #[getter]
    fn kinetic<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix(py, self.inner.n_functions(), |i, j| {
            self.inner.kinetic(i, j)
        })
    }

    #[getter]
    fn nuclear<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix(py, self.inner.n_functions(), |i, j| {
            self.inner.nuclear(i, j)
        })
    }

    /// Core Hamiltonian `T + V`.
    #[getter]
    fn core<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix(py, self.inner.n_functions(), |i, j| self.inner.core(i, j))
    }

    /// `(ij|kl)` in chemists' notation, shape `(n, n, n, n)`.
    #[getter]
    fn eri<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray4<f64>>> {
        let n = self.inner.n_functions();
        Array4::from_shape_vec((n, n, n, n), self.inner.eri_tensor().to_vec())
            .map(|a| a.into_pyarray(py))
            .map_err(|e| invalid(e.to_string()))
    }

    /// Atom of every basis function.
    #[getter]
    fn atoms(&self) -> Vec<usize> {
        self.inner.atoms().to_vec()
    }

    /// Angular momentum of every basis function.
    #[getter]
    fn angular_momenta(&self) -> Vec<u32> {
        self.inner.angular_momenta().to_vec()
    }

    fn __repr__(&self) -> String {
        format!("AoIntegrals(n_functions={})", self.inner.n_functions())
    }
}

/// A converged Hartree–Fock solution (see `hartree_fock`).
#[pyclass(name = "HartreeFock", module = "metatron_qso", frozen)]
pub struct PyHartreeFock {
    inner: ScfResult,
    /// Shared, not copied (the n⁴ tensor can be large).
    integrals: Py<PyAoIntegrals>,
}

#[pymethods]
impl PyHartreeFock {
    /// Total energy including the nuclear repulsion (hartree).
    #[getter]
    fn energy(&self) -> f64 {
        self.inner.energy
    }

    #[getter]
    fn nuclear_repulsion(&self) -> f64 {
        self.inner.nuclear_repulsion
    }

    /// Orbital energies: one array for RHF and ROHF (eigenvalues of the
    /// effective Fock matrix), `[alpha, beta]` for UHF.
    #[getter]
    fn orbital_energies<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray1<f64>>> {
        self.inner
            .orbital_energies
            .iter()
            .map(|e| real_array(py, e.clone()))
            .collect()
    }

    /// Orbital coefficients (columns): one matrix for RHF and ROHF,
    /// `[alpha, beta]` for UHF.
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<f64>>> {
        self.inner
            .coefficients
            .iter()
            .map(|c| rmatrix(py, c))
            .collect()
    }

    #[getter]
    fn n_alpha(&self) -> usize {
        self.inner.n_alpha
    }

    #[getter]
    fn n_beta(&self) -> usize {
        self.inner.n_beta
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    fn restricted(&self) -> bool {
        self.inner.is_restricted()
    }

    /// `⟨S²⟩` of the determinant.
    fn s_squared(&self) -> f64 {
        self.inner.s_squared()
    }

    /// The integrals the calculation used.
    #[getter]
    fn integrals(&self, py: Python<'_>) -> Py<PyAoIntegrals> {
        self.integrals.clone_ref(py)
    }

    /// The Hamiltonian in the restricted orbitals, the lowest `frozen`
    /// doubly occupied and folded into the core energy, keeping `active`
    /// orbitals (all others when `None`).
    #[pyo3(signature = (frozen = 0, active = None))]
    fn molecular_integrals(
        &self,
        py: Python<'_>,
        frozen: usize,
        active: Option<usize>,
    ) -> PyResult<PyMolecularIntegrals> {
        let ints = &self.integrals.get().inner;
        py.detach(|| self.inner.molecular_integrals(ints, frozen, active))
            .raise()
            .map(|inner| PyMolecularIntegrals { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "HartreeFock(energy={:.10}, restricted={}, iterations={})",
            self.inner.energy,
            self.inner.is_restricted(),
            self.inner.iterations
        )
    }
}

fn parse_harmonics(name: &str) -> PyResult<Harmonics> {
    match name.to_ascii_lowercase().as_str() {
        "spherical" => Ok(Harmonics::Spherical),
        "cartesian" => Ok(Harmonics::Cartesian),
        other => Err(invalid(format!(
            "unknown harmonics '{other}' (expected spherical or cartesian)"
        ))),
    }
}

/// Restricted (default), unrestricted or restricted open-shell
/// (`restricted_open=True`) Hartree–Fock of `molecule` in `basis`, with
/// DIIS; `integrals` reuses integrals computed before.
#[pyfunction]
#[pyo3(signature = (
    molecule,
    basis = "sto-3g",
    unrestricted = false,
    restricted_open = false,
    harmonics = "spherical",
    max_iterations = 200,
    energy_tolerance = 1e-10,
    gradient_tolerance = 1e-8,
    diis_space = 8,
    linear_dependence = 1e-8,
    level_shift = 0.0,
    guess = "atoms",
    integrals = None,
))]
#[allow(clippy::too_many_arguments)]
fn hartree_fock(
    py: Python<'_>,
    molecule: &PyMolecule,
    basis: &str,
    unrestricted: bool,
    restricted_open: bool,
    harmonics: &str,
    max_iterations: usize,
    energy_tolerance: f64,
    gradient_tolerance: f64,
    diis_space: usize,
    linear_dependence: f64,
    level_shift: f64,
    guess: &str,
    integrals: Option<Py<PyAoIntegrals>>,
) -> PyResult<PyHartreeFock> {
    let basis = Basis::from_name(basis).raise()?;
    let harmonics = parse_harmonics(harmonics)?;
    if unrestricted && restricted_open {
        return Err(invalid("choose unrestricted or restricted_open, not both"));
    }
    let guess = match guess {
        "atoms" => Guess::Atoms,
        "core" => Guess::Core,
        other => {
            return Err(invalid(format!(
                "unknown guess '{other}' (expected atoms or core)"
            )));
        }
    };
    let options = ScfOptions {
        max_iterations,
        energy_tolerance,
        gradient_tolerance,
        diis_space,
        linear_dependence,
        guess,
        level_shift,
    };
    if let Some(given) = &integrals
        && (given.get().inner.basis() != basis || given.get().inner.harmonics() != harmonics)
    {
        return Err(invalid(
            "the integrals were computed in another basis or with other harmonics",
        ));
    }
    let mol = &molecule.inner;
    let (inner, computed) = py
        .detach(
            || -> metatron_qso::Result<(ScfResult, Option<AoIntegrals>)> {
                let computed = match &integrals {
                    Some(_) => None,
                    None => Some(AoIntegrals::compute_with(mol, basis, harmonics)?),
                };
                let ints = match (&integrals, &computed) {
                    (Some(p), _) => &p.get().inner,
                    (None, Some(c)) => c,
                    (None, None) => unreachable!(),
                };
                let inner = if unrestricted {
                    uhf(mol, ints, &options)?
                } else if restricted_open {
                    rohf(mol, ints, &options)?
                } else {
                    rhf(mol, ints, &options)?
                };
                Ok((inner, computed))
            },
        )
        .raise()?;
    let integrals = match (integrals, computed) {
        (Some(p), _) => p,
        (None, Some(c)) => Py::new(py, PyAoIntegrals { inner: c })?,
        (None, None) => unreachable!(),
    };
    Ok(PyHartreeFock { inner, integrals })
}
