//! Discrete-time quantum walks and walk circuits.

use metatron_qso::C64;
use metatron_qso::linalg::CMatrix;
use metatron_qso::walks::{Coin, CoinedWalk, LineWalk, SzegedyWalk, circuits};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::algorithms::Indices;
use crate::circuit::PyCircuit;
use crate::graph::PyGraph;
use crate::walks::{check_time, generator};
use crate::{
    ComplexArg, ComplexMatrixArg, Raise, RealArg, complex_array, complex_vec, invalid, pydict,
    real_array, real_vec, square_complex,
};

/// A coin given by name or as a unitary matrix.
#[derive(FromPyObject)]
enum CoinArg<'py> {
    Name(String),
    Matrix(ComplexMatrixArg<'py>),
}

fn coin(arg: CoinArg<'_>) -> PyResult<Coin> {
    match arg {
        CoinArg::Name(name) => match name.to_ascii_lowercase().as_str() {
            "grover" => Ok(Coin::Grover),
            "fourier" => Ok(Coin::Fourier),
            "hadamard" => Ok(Coin::Hadamard),
            _ => Err(invalid(format!(
                "unknown coin '{name}' (expected grover, fourier, hadamard or a unitary matrix)"
            ))),
        },
        CoinArg::Matrix(m) => {
            let (dim, data) = square_complex("coin", &m)?;
            Ok(Coin::Unitary(CMatrix::from_row_slice(dim, dim, &data)))
        }
    }
}

/// Coined quantum walk on the arcs `u → v` of a graph with the flip-flop
/// shift. `coin` is `"grover"` (default), `"fourier"`, `"hadamard"` (degree
/// 2 only) or a unitary matrix (regular graphs); `marked` nodes get the
/// coin `−I` (Shenvi–Kempe–Whaley search).
#[pyclass(name = "CoinedWalk", module = "metatron_qso", frozen)]
pub struct PyCoinedWalk {
    inner: CoinedWalk,
}

#[pymethods]
impl PyCoinedWalk {
    #[new]
    #[pyo3(
        signature = (graph, coin = CoinArg::Name("grover".to_string()), marked = None),
        text_signature = "(graph, coin='grover', marked=None)"
    )]
    fn new(graph: &PyGraph, coin: CoinArg<'_>, marked: Option<Indices>) -> PyResult<Self> {
        let mut inner = CoinedWalk::new(&graph.inner, self::coin(coin)?).raise()?;
        if let Some(m) = marked {
            inner = inner.with_marked(&m.into_vec()).raise()?;
        }
        Ok(Self { inner })
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Dimension of the walk (twice the number of edges).
    #[getter]
    fn num_arcs(&self) -> usize {
        self.inner.num_arcs()
    }

    /// The arcs `(tail, head)` in the order of the state vector.
    fn arcs(&self) -> Vec<(usize, usize)> {
        self.inner.arcs()
    }

    /// Walker at `node` with coin state `coin` (one amplitude per arc
    /// leaving the node), by default the uniform superposition.
    #[pyo3(signature = (node, coin = None))]
    fn localized_state<'py>(
        &self,
        py: Python<'py>,
        node: usize,
        coin: Option<ComplexArg<'_>>,
    ) -> PyResult<Bound<'py, PyArray1<C64>>> {
        let c = coin.map(|c| complex_vec(&c));
        Ok(complex_array(
            py,
            self.inner.localized_state(node, c.as_deref()).raise()?,
        ))
    }

    /// Uniform superposition over all arcs.
    fn uniform_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<C64>> {
        complex_array(py, self.inner.uniform_state())
    }

    /// One step: coin, then shift.
    fn step<'py>(
        &self,
        py: Python<'py>,
        psi: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<C64>>> {
        Ok(complex_array(
            py,
            self.inner.step(&complex_vec(&psi)).raise()?,
        ))
    }

    /// `steps` steps from `psi0`.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        psi0: ComplexArg<'_>,
        steps: usize,
    ) -> PyResult<Bound<'py, PyArray1<C64>>> {
        let psi0 = complex_vec(&psi0);
        let w = &self.inner;
        Ok(complex_array(
            py,
            py.detach(|| w.evolve(&psi0, steps)).raise()?,
        ))
    }

    /// Probability of each node (total weight of the arcs leaving it).
    fn node_probabilities<'py>(
        &self,
        py: Python<'py>,
        psi: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(real_array(
            py,
            self.inner.node_probabilities(&complex_vec(&psi)).raise()?,
        ))
    }

    /// Node probabilities after `steps` steps from `start` (uniform coin).
    fn probabilities<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        steps: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let w = &self.inner;
        Ok(real_array(
            py,
            py.detach(|| w.probabilities(start, steps)).raise()?,
        ))
    }

    /// Search from the uniform superposition: `{"steps",
    /// "success_probability", "history"}` with the first peak of the
    /// success probability within `max_steps` steps.
    fn search<'py>(&self, py: Python<'py>, max_steps: usize) -> PyResult<Bound<'py, PyDict>> {
        let w = &self.inner;
        let r = py.detach(|| w.search(max_steps)).raise()?;
        Ok(pydict!(py, {
            "steps" => r.steps,
            "success_probability" => r.success_probability,
            "history" => real_array(py, r.history),
        }))
    }
}

/// Coined walk on the integer line with the moving shift; coin state 0
/// moves left, 1 right. `coin=None` is the Hadamard walk.
#[pyclass(name = "LineWalk", module = "metatron_qso", frozen)]
pub struct PyLineWalk {
    inner: LineWalk,
}

#[pymethods]
impl PyLineWalk {
    #[new]
    #[pyo3(signature = (coin = None))]
    fn new(coin: Option<ComplexMatrixArg<'_>>) -> PyResult<Self> {
        let inner = match coin {
            None => LineWalk::hadamard(),
            Some(m) => {
                let (dim, data) = square_complex("coin", &m)?;
                LineWalk::new(&CMatrix::from_row_slice(dim, dim, &data)).raise()?
            }
        };
        Ok(Self { inner })
    }

    /// Position distribution after `steps` steps from the origin; entry
    /// `k` is position `k − steps`. The default coin state
    /// `(|L⟩ + i|R⟩)/√2` gives the symmetric Hadamard walk.
    #[pyo3(signature = (steps, coin_state = None))]
    fn distribution<'py>(
        &self,
        py: Python<'py>,
        steps: usize,
        coin_state: Option<ComplexArg<'_>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let state = match coin_state {
            None => [C64::new(s, 0.0), C64::new(0.0, s)],
            Some(c) => {
                let v = complex_vec(&c);
                <[C64; 2]>::try_from(v.as_slice())
                    .map_err(|_| invalid("`coin_state` must have two amplitudes"))?
            }
        };
        if steps > 1 << 16 {
            return Err(invalid("at most 2^16 steps"));
        }
        let w = &self.inner;
        Ok(real_array(
            py,
            py.detach(|| w.distribution(steps, state)).raise()?,
        ))
    }
}

/// Szegedy walk of the random walk `P_uv = w_uv/d_u` with the `marked`
/// nodes absorbing; `hitting_time` is Szegedy's quantum hitting time,
/// which grows as the square root of the classical one.
#[pyclass(name = "SzegedyWalk", module = "metatron_qso", frozen)]
pub struct PySzegedyWalk {
    inner: SzegedyWalk,
}

#[pymethods]
impl PySzegedyWalk {
    #[new]
    fn new(graph: &PyGraph, marked: Indices) -> PyResult<Self> {
        let inner = SzegedyWalk::new(&graph.inner, &marked.into_vec()).raise()?;
        Ok(Self { inner })
    }

    /// Number of arcs the walk acts on.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Stationary probability of the marked set.
    #[getter]
    fn marked_weight(&self) -> f64 {
        self.inner.marked_weight()
    }

    /// The start state |π⟩ in the arc basis.
    fn stationary_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        real_array(py, self.inner.stationary_state().to_vec())
    }

    /// One step `W' = (S·R)²`.
    fn step<'py>(
        &self,
        py: Python<'py>,
        psi: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<C64>>> {
        Ok(complex_array(
            py,
            self.inner.step(&complex_vec(&psi)).raise()?,
        ))
    }

    /// `‖W'^t|π⟩ − |π⟩‖²` for `t = 0, …, steps`.
    fn deviations<'py>(&self, py: Python<'py>, steps: usize) -> Bound<'py, PyArray1<f64>> {
        let w = &self.inner;
        real_array(py, py.detach(|| w.deviations(steps)))
    }

    /// Quantum hitting time, or `None` if not reached within `max_steps`.
    fn hitting_time(&self, py: Python<'_>, max_steps: usize) -> Option<usize> {
        let w = &self.inner;
        py.detach(|| w.hitting_time(max_steps))
    }
}

/// Exact walk circuit `exp(−iHt)` for a circulant graph on 2^n nodes
/// (inverse QFT, diagonal phases, QFT).
#[pyfunction]
#[pyo3(signature = (graph, t, generator = "laplacian", gamma = 1.0))]
fn circulant_walk_circuit(
    graph: &PyGraph,
    t: f64,
    generator: &str,
    gamma: f64,
) -> PyResult<PyCircuit> {
    check_time("t", t)?;
    PyCircuit::wrap(
        circuits::circulant_walk(&graph.inner, self::generator(generator)?, gamma, t).raise()?,
    )
}

/// Exact walk circuit for a Cayley graph of Z_2^n (hypercubes and
/// relatives): commuting X-string rotations.
#[pyfunction]
#[pyo3(signature = (graph, t, generator = "laplacian", gamma = 1.0))]
fn cayley_walk_circuit(
    graph: &PyGraph,
    t: f64,
    generator: &str,
    gamma: f64,
) -> PyResult<PyCircuit> {
    check_time("t", t)?;
    PyCircuit::wrap(
        circuits::cayley_walk(&graph.inner, self::generator(generator)?, gamma, t).raise()?,
    )
}

/// Product-formula walk circuit for any graph: `steps` steps of order 1 or
/// 2, one two-level rotation per edge; the error falls as `steps^-order`.
#[pyfunction]
#[pyo3(signature = (graph, t, steps, order = 2, generator = "laplacian", gamma = 1.0))]
fn trotter_walk_circuit(
    py: Python<'_>,
    graph: &PyGraph,
    t: f64,
    steps: usize,
    order: usize,
    generator: &str,
    gamma: f64,
) -> PyResult<PyCircuit> {
    check_time("t", t)?;
    let gen_ = self::generator(generator)?;
    if (graph.inner.num_edges() as u128) * (steps as u128) * (order as u128) > 1 << 22 {
        return Err(invalid("the circuit would exceed 2^22 edge rotations"));
    }
    let g = &graph.inner;
    PyCircuit::wrap(
        py.detach(|| circuits::trotter_walk(g, gen_, gamma, t, steps, order))
            .raise()?,
    )
}

/// Exact circuit for the diagonal unitary `diag(exp(i·phases))` (length a
/// power of two), global phase included.
#[pyfunction]
fn diagonal_unitary_circuit(phases: RealArg<'_>) -> PyResult<PyCircuit> {
    let phases = real_vec(&phases);
    if phases.len() > 1 << 20 {
        return Err(invalid("at most 2^20 phases"));
    }
    PyCircuit::wrap(circuits::diagonal_unitary(&phases).raise()?)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCoinedWalk>()?;
    m.add_class::<PyLineWalk>()?;
    m.add_class::<PySzegedyWalk>()?;
    m.add_function(wrap_pyfunction!(circulant_walk_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(cayley_walk_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(trotter_walk_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal_unitary_circuit, m)?)?;
    Ok(())
}
