//! Hamiltonian simulation, state geometry and excited states.

use metatron_qso::algorithms::excited::{
    ExcitedStateOptions, ssvqe as ssvqe_core, vqd as vqd_core,
};
use metatron_qso::algorithms::geometry::{
    NaturalGradientOptions, imaginary_time_evolution as varqite, minimize_natural_gradient,
    natural_gradient as natural_gradient_core, quantum_fisher_information as qfi,
};
use metatron_qso::algorithms::time_evolution;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::circuit::PyCircuit;
use crate::state::{PyPauliSum, PyStateVector};
use crate::{Raise, RealArg, check_finite, check_qubits, invalid, pydict, real_vec, rmatrix};

/// Circuit for exp(−iHt) from `steps` steps of the product formula of
/// `order` (1, 2, 4, 6, …).
#[pyfunction]
#[pyo3(signature = (hamiltonian, n_qubits, t, steps = 1, order = 2))]
fn trotter_circuit(
    py: Python<'_>,
    hamiltonian: &PyPauliSum,
    n_qubits: usize,
    t: f64,
    steps: usize,
    order: usize,
) -> PyResult<PyCircuit> {
    check_qubits(n_qubits)?;
    let h = &hamiltonian.inner;
    PyCircuit::wrap(
        py.detach(|| time_evolution::trotter_circuit(h, n_qubits, t, steps, order))
            .raise()?,
    )
}

/// Upper bound on ‖U − exp(−iHt)‖ for `trotter_circuit` with the same
/// arguments (commutator bounds for orders 1 and 2).
#[pyfunction]
#[pyo3(signature = (hamiltonian, t, steps = 1, order = 2))]
fn trotter_error_bound(
    py: Python<'_>,
    hamiltonian: &PyPauliSum,
    t: f64,
    steps: usize,
    order: usize,
) -> PyResult<f64> {
    let h = &hamiltonian.inner;
    py.detach(|| time_evolution::trotter_error_bound(h, t, steps, order))
        .raise()
}

/// One random qDRIFT circuit for exp(−iHt) with `samples` rotations.
#[pyfunction]
#[pyo3(signature = (hamiltonian, n_qubits, t, samples, seed = 0))]
fn qdrift_circuit(
    hamiltonian: &PyPauliSum,
    n_qubits: usize,
    t: f64,
    samples: usize,
    seed: u64,
) -> PyResult<PyCircuit> {
    check_qubits(n_qubits)?;
    if samples > 10_000_000 {
        return Err(invalid("samples must be at most 10^7"));
    }
    PyCircuit::wrap(
        time_evolution::qdrift_circuit(&hamiltonian.inner, n_qubits, t, samples, seed).raise()?,
    )
}

/// Campbell's bound 2λ²t²/N·e^{2λt/N} on the qDRIFT channel error.
#[pyfunction]
fn qdrift_error_bound(hamiltonian: &PyPauliSum, t: f64, samples: usize) -> PyResult<f64> {
    time_evolution::qdrift_error_bound(&hamiltonian.inner, t, samples).raise()
}

fn theta_vec(theta: RealArg<'_>) -> PyResult<Vec<f64>> {
    let theta = real_vec(&theta);
    check_finite("theta", &theta)?;
    Ok(theta)
}

/// Quantum Fisher information matrix F_ij = 4 Re(⟨∂iψ|∂jψ⟩ − ⟨∂iψ|ψ⟩⟨ψ|∂jψ⟩).
#[pyfunction]
#[pyo3(signature = (circuit, theta, initial_state = None))]
fn quantum_fisher_information<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    theta: RealArg<'_>,
    initial_state: Option<&PyStateVector>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let theta = theta_vec(theta)?;
    let c = &circuit.inner;
    let init = initial_state.map(|s| &s.inner);
    let f = py.detach(|| qfi(c, &theta, init)).raise()?;
    Ok(rmatrix(py, &f))
}

/// Energy, gradient and natural-gradient direction (F/4 + λI)⁻¹∇E.
#[pyfunction]
#[pyo3(signature = (circuit, observable, theta, initial_state = None, regularization = 1e-3))]
fn natural_gradient<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    observable: &PyPauliSum,
    theta: RealArg<'_>,
    initial_state: Option<&PyStateVector>,
    regularization: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let theta = theta_vec(theta)?;
    let (c, h) = (&circuit.inner, &observable.inner);
    let init = initial_state.map(|s| &s.inner);
    let r = py
        .detach(|| natural_gradient_core(c, h, &theta, init, regularization))
        .raise()?;
    Ok(pydict!(py, {
        "value" => r.value,
        "gradient" => r.gradient.into_pyarray(py),
        "direction" => r.direction.into_pyarray(py),
    }))
}

/// Minimise ⟨H⟩ by quantum natural gradient descent.
#[pyfunction]
#[pyo3(signature = (
    circuit, observable, theta, initial_state = None, learning_rate = 0.1,
    regularization = 1e-3, max_iterations = 500, gradient_tolerance = 1e-6
))]
#[allow(clippy::too_many_arguments)]
fn quantum_natural_gradient<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    observable: &PyPauliSum,
    theta: RealArg<'_>,
    initial_state: Option<&PyStateVector>,
    learning_rate: f64,
    regularization: f64,
    max_iterations: usize,
    gradient_tolerance: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let theta = theta_vec(theta)?;
    let (c, h) = (&circuit.inner, &observable.inner);
    let init = initial_state.map(|s| &s.inner);
    let opts = NaturalGradientOptions::default()
        .with_learning_rate(learning_rate)
        .with_regularization(regularization)
        .with_max_iterations(max_iterations)
        .with_gradient_tolerance(gradient_tolerance);
    let r = py
        .detach(|| minimize_natural_gradient(c, h, &theta, init, &opts))
        .raise()?;
    Ok(pydict!(py, {
        "energy" => r.energy,
        "parameters" => r.parameters.into_pyarray(py),
        "iterations" => r.iterations,
        "converged" => r.converged,
        "history" => r.history.into_pyarray(py),
    }))
}

/// Variational imaginary-time evolution (McLachlan, RK4).
#[pyfunction]
#[pyo3(signature = (circuit, observable, theta, time, steps = 100, initial_state = None, rcond = 1e-8))]
#[allow(clippy::too_many_arguments)]
fn imaginary_time_evolution<'py>(
    py: Python<'py>,
    circuit: &PyCircuit,
    observable: &PyPauliSum,
    theta: RealArg<'_>,
    time: f64,
    steps: usize,
    initial_state: Option<&PyStateVector>,
    rcond: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let theta = theta_vec(theta)?;
    let (c, h) = (&circuit.inner, &observable.inner);
    let init = initial_state.map(|s| &s.inner);
    let r = py
        .detach(|| varqite(c, h, &theta, init, time, steps, rcond))
        .raise()?;
    let p = theta.len();
    let rows = r.parameters.len();
    let params = Array2::from_shape_fn((rows, p), |(i, j)| r.parameters[i][j]);
    Ok(pydict!(py, {
        "times" => r.times.into_pyarray(py),
        "energies" => r.energies.into_pyarray(py),
        "parameters" => params.into_pyarray(py),
    }))
}

fn excited_options(
    restarts: usize,
    seed: u64,
    betas: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
) -> ExcitedStateOptions {
    let mut o = ExcitedStateOptions::default()
        .with_restarts(restarts)
        .with_seed(seed);
    if let Some(b) = betas {
        o = o.with_betas(b);
    }
    if let Some(w) = weights {
        o = o.with_weights(w);
    }
    o
}

/// The lowest `n_states` eigenstates by variational quantum deflation.
#[pyfunction]
#[pyo3(signature = (hamiltonian, ansatz, n_states, restarts = 4, seed = 0, betas = None))]
fn vqd<'py>(
    py: Python<'py>,
    hamiltonian: &PyPauliSum,
    ansatz: &PyCircuit,
    n_states: usize,
    restarts: usize,
    seed: u64,
    betas: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = excited_options(restarts, seed, betas, None);
    let (h, c) = (&hamiltonian.inner, &ansatz.inner);
    let r = py.detach(|| vqd_core(h, c, n_states, &opts)).raise()?;
    let p = c.num_parameters();
    let params = Array2::from_shape_fn((r.parameters.len(), p), |(i, j)| r.parameters[i][j]);
    Ok(pydict!(py, {
        "energies" => r.energies.into_pyarray(py),
        "parameters" => params.into_pyarray(py),
    }))
}

/// The lowest `n_states` eigenvalues by weighted subspace-search VQE.
#[pyfunction]
#[pyo3(signature = (hamiltonian, ansatz, n_states, restarts = 4, seed = 0, weights = None))]
fn ssvqe<'py>(
    py: Python<'py>,
    hamiltonian: &PyPauliSum,
    ansatz: &PyCircuit,
    n_states: usize,
    restarts: usize,
    seed: u64,
    weights: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = excited_options(restarts, seed, None, weights);
    let (h, c) = (&hamiltonian.inner, &ansatz.inner);
    let r = py.detach(|| ssvqe_core(h, c, n_states, &opts)).raise()?;
    Ok(pydict!(py, {
        "energies" => r.energies.into_pyarray(py),
        "parameters" => r.parameters[0].clone().into_pyarray(py),
    }))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trotter_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(trotter_error_bound, m)?)?;
    m.add_function(wrap_pyfunction!(qdrift_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(qdrift_error_bound, m)?)?;
    m.add_function(wrap_pyfunction!(quantum_fisher_information, m)?)?;
    m.add_function(wrap_pyfunction!(natural_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(quantum_natural_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(imaginary_time_evolution, m)?)?;
    m.add_function(wrap_pyfunction!(vqd, m)?)?;
    m.add_function(wrap_pyfunction!(ssvqe, m)?)?;
    Ok(())
}
