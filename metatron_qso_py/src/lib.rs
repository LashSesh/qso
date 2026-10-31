//! Python bindings for QSO ([`metatron_qso`]).
//!
//! The extension module is `metatron_qso._core`; the pure-Python package
//! `metatron_qso` re-exports it. Conventions:
//!
//! * every [`metatron_qso::QsoError`] is raised as `ValueError` with the
//!   error's message;
//! * inputs the core treats as programming errors (and would panic on) are
//!   validated here first;
//! * vectors and matrices are returned as NumPy `float64` / `complex128`
//!   arrays; array arguments accept anything `numpy.asarray` accepts;
//! * long-running calls release the GIL.

mod algorithms;
mod chemistry;
mod circuit;
mod discrete;
mod estimation;
mod graph;
mod learning;
mod mitigation;
mod mps;
mod noise;
mod nv;
mod optimization;
mod photonics;
mod qec;
mod reduced;
mod simulation;
mod slow;
mod spin;
mod stabilizer;
mod state;
mod symmetric;
mod walks;

use metatron_qso::C64;
use metatron_qso::QsoError;
use metatron_qso::linalg::{CMatrix, RMatrix};
use metatron_qso::quantum::{Counts, MAX_QUBITS, bitstring};
use numpy::ndarray::Array2;
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Real vector argument (lists, tuples and arrays of any real dtype).
pub(crate) type RealArg<'py> = PyArrayLike1<'py, f64, AllowTypeChange>;
/// Complex vector argument.
pub(crate) type ComplexArg<'py> = PyArrayLike1<'py, C64, AllowTypeChange>;
/// Real matrix argument.
pub(crate) type RealMatrixArg<'py> = PyArrayLike2<'py, f64, AllowTypeChange>;
/// Complex matrix argument.
pub(crate) type ComplexMatrixArg<'py> = PyArrayLike2<'py, C64, AllowTypeChange>;

/// Conversion of core results into Python results.
pub(crate) trait Raise<T> {
    fn raise(self) -> PyResult<T>;
}

impl<T> Raise<T> for Result<T, QsoError> {
    fn raise(self) -> PyResult<T> {
        self.map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// `ValueError` with a message.
pub(crate) fn invalid(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Reject registers the state-vector simulator cannot hold.
pub(crate) fn check_qubits(n_qubits: usize) -> PyResult<()> {
    if n_qubits == 0 || n_qubits > MAX_QUBITS {
        return Err(invalid(format!(
            "number of qubits must lie in 1..={MAX_QUBITS}, got {n_qubits}"
        )));
    }
    Ok(())
}

/// Reject non-finite real inputs.
pub(crate) fn check_finite(name: &str, values: &[f64]) -> PyResult<()> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(invalid(format!("`{name}` must be finite")))
    }
}

pub(crate) fn real_vec(a: &RealArg<'_>) -> Vec<f64> {
    a.as_array().to_vec()
}

pub(crate) fn complex_vec(a: &ComplexArg<'_>) -> Vec<C64> {
    a.as_array().to_vec()
}

/// Square complex matrix in row-major order with its dimension.
pub(crate) fn square_complex(name: &str, a: &ComplexMatrixArg<'_>) -> PyResult<(usize, Vec<C64>)> {
    let view = a.as_array();
    let (rows, cols) = view.dim();
    if rows != cols || rows == 0 {
        return Err(invalid(format!(
            "`{name}` must be a non-empty square matrix, got shape ({rows}, {cols})"
        )));
    }
    Ok((rows, view.iter().copied().collect()))
}

pub(crate) fn real_array<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    v.into_pyarray(py)
}

pub(crate) fn complex_array<'py>(py: Python<'py>, v: Vec<C64>) -> Bound<'py, PyArray1<C64>> {
    v.into_pyarray(py)
}

pub(crate) fn rmatrix<'py>(py: Python<'py>, m: &RMatrix) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(r, c)| m[(r, c)]).into_pyarray(py)
}

pub(crate) fn cmatrix<'py>(py: Python<'py>, m: &CMatrix) -> Bound<'py, PyArray2<C64>> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(r, c)| m[(r, c)]).into_pyarray(py)
}

/// Row-major `dim × dim` data as a NumPy matrix.
pub(crate) fn square_array<'py>(
    py: Python<'py>,
    dim: usize,
    data: &[C64],
) -> Bound<'py, PyArray2<C64>> {
    Array2::from_shape_fn((dim, dim), |(r, c)| data[r * dim + c]).into_pyarray(py)
}

/// Measurement histogram keyed by bit strings (qubit `n−1` first).
pub(crate) fn counts_dict<'py>(
    py: Python<'py>,
    counts: &Counts,
    n_qubits: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (&b, &n) in counts {
        d.set_item(bitstring(b, n_qubits), n)?;
    }
    Ok(d)
}

/// Build a `dict` from `key => value` pairs (inside a `PyResult` function).
macro_rules! pydict {
    ($py:expr, { $($k:literal => $v:expr),* $(,)? }) => {{
        let d = pyo3::types::PyDict::new($py);
        $( pyo3::types::PyDictMethods::set_item(&d, $k, $v)?; )*
        d
    }};
}
pub(crate) use pydict;

/// Format basis index `index` as a bit string with qubit `n−1` first.
#[pyfunction(name = "bitstring")]
fn py_bitstring(index: usize, n_qubits: usize) -> PyResult<String> {
    if n_qubits < usize::BITS as usize && index >> n_qubits != 0 {
        return Err(invalid(format!(
            "index {index} does not fit into {n_qubits} qubits"
        )));
    }
    Ok(bitstring(index, n_qubits))
}

/// The instruction set of the amplitude kernels: "avx", "neon" or
/// "portable".
#[pyfunction]
fn simd_level() -> &'static str {
    metatron_qso::quantum::simd_level()
}

/// Use the vector kernels where the processor has them (the default) or
/// the portable ones; results are bit for bit the same. Returns whether
/// vector kernels are in use.
#[pyfunction]
fn set_simd(enabled: bool) -> bool {
    metatron_qso::quantum::set_simd(enabled)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("MAX_QUBITS", MAX_QUBITS)?;
    m.add(
        "MAX_DENSITY_QUBITS",
        metatron_qso::noise::MAX_DENSITY_QUBITS,
    )?;
    m.add_function(wrap_pyfunction!(py_bitstring, m)?)?;
    m.add_function(wrap_pyfunction!(simd_level, m)?)?;
    m.add_function(wrap_pyfunction!(set_simd, m)?)?;
    graph::register(m)?;
    circuit::register(m)?;
    state::register(m)?;
    algorithms::register(m)?;
    walks::register(m)?;
    reduced::register(m)?;
    discrete::register(m)?;
    noise::register(m)?;
    slow::register(m)?;
    estimation::register(m)?;
    mitigation::register(m)?;
    learning::register(m)?;
    photonics::register(m)?;
    chemistry::register(m)?;
    simulation::register(m)?;
    optimization::register(m)?;
    stabilizer::register(m)?;
    qec::register(m)?;
    mps::register(m)?;
    spin::register(m)?;
    nv::register(m)?;
    symmetric::register(m)?;
    Ok(())
}
