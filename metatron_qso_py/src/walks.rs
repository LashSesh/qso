//! Continuous-time quantum walks, spatial search, dephasing and classical
//! baselines.

use metatron_qso::walks::{
    ContinuousWalk, DephasingWalk, Generator, SpatialSearch, classical_hitting_times,
    classical_walk_probabilities,
};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::algorithms::Indices;
use crate::graph::PyGraph;
use crate::{
    ComplexArg, ComplexMatrixArg, Raise, cmatrix, complex_array, complex_vec, invalid, pydict,
    real_array, square_complex,
};

fn generator(name: &str) -> PyResult<Generator> {
    match name.to_ascii_lowercase().as_str() {
        "laplacian" => Ok(Generator::Laplacian),
        "adjacency" => Ok(Generator::Adjacency),
        _ => Err(invalid(format!(
            "unknown generator '{name}' (expected laplacian or adjacency)"
        ))),
    }
}

fn generator_name(g: Generator) -> &'static str {
    match g {
        Generator::Laplacian => "laplacian",
        Generator::Adjacency => "adjacency",
    }
}

fn check_time(name: &str, t: f64) -> PyResult<()> {
    if t.is_finite() {
        Ok(())
    } else {
        Err(invalid(format!("`{name}` must be finite")))
    }
}

/// Continuous-time quantum walk `|ψ(t)⟩ = exp(−iHt)|ψ(0)⟩` with
/// `H = γL` (`"laplacian"`) or `H = −γA` (`"adjacency"`).
///
/// Graphs with up to 2048 nodes are diagonalised once; larger graphs use
/// Krylov propagation (no limiting/average distributions or propagator).
#[pyclass(name = "ContinuousWalk", module = "metatron_qso", frozen)]
pub struct PyContinuousWalk {
    inner: ContinuousWalk,
}

#[pymethods]
impl PyContinuousWalk {
    #[new]
    #[pyo3(signature = (graph, generator = "laplacian", gamma = 1.0))]
    fn new(py: Python<'_>, graph: &PyGraph, generator: &str, gamma: f64) -> PyResult<Self> {
        let gen_ = self::generator(generator)?;
        let g = &graph.inner;
        let inner = py.detach(|| ContinuousWalk::new(g, gen_, gamma)).raise()?;
        Ok(Self { inner })
    }

    /// Node probabilities at time `t` for a walker started at `start`.
    fn probabilities<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_time("t", t)?;
        let w = &self.inner;
        Ok(real_array(
            py,
            py.detach(|| w.probabilities(start, t)).raise()?,
        ))
    }

    /// `exp(−iHt)|ψ₀⟩` for an arbitrary (not necessarily normalised) vector.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        psi0: ComplexArg<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        check_time("t", t)?;
        let psi0 = complex_vec(&psi0);
        let w = &self.inner;
        Ok(complex_array(py, py.detach(|| w.evolve(&psi0, t)).raise()?))
    }

    /// Infinite-time-averaged distribution (degenerate eigenspaces grouped).
    fn limiting_distribution<'py>(
        &self,
        py: Python<'py>,
        start: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let w = &self.inner;
        Ok(real_array(
            py,
            py.detach(|| w.limiting_distribution(start)).raise()?,
        ))
    }

    /// Exact time average of the node probabilities over `[0, t_max]`.
    fn average_distribution<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        t_max: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_time("t_max", t_max)?;
        let w = &self.inner;
        Ok(real_array(
            py,
            py.detach(|| w.average_distribution(start, t_max)).raise()?,
        ))
    }

    /// Mixing time (Aharonov et al. 2001): smallest `T` on the grid
    /// `t_max·k/steps` from which on the time-averaged distribution stays
    /// within total-variation distance `epsilon` of the limit, up to
    /// `t_max` (`None` if the bound fails at `t_max`).
    #[pyo3(signature = (start, epsilon, t_max, steps = 200))]
    fn mixing_time(
        &self,
        py: Python<'_>,
        start: usize,
        epsilon: f64,
        t_max: f64,
        steps: usize,
    ) -> PyResult<Option<f64>> {
        check_time("t_max", t_max)?;
        if epsilon.is_nan() {
            return Err(invalid("`epsilon` must not be NaN"));
        }
        let w = &self.inner;
        py.detach(|| w.mixing_time(start, epsilon, t_max, steps))
            .raise()
    }

    /// Dense propagator `U(t) = exp(−iHt)`.
    fn propagator<'py>(
        &self,
        py: Python<'py>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray2<metatron_qso::C64>>> {
        check_time("t", t)?;
        let w = &self.inner;
        let u = py.detach(|| w.propagator(t)).raise()?;
        Ok(cmatrix(py, &u))
    }

    /// Eigenvalues of `H` (ascending), or `None` for Krylov-only walks.
    #[getter]
    fn spectrum<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.spectrum().map(|s| real_array(py, s.to_vec()))
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    #[getter]
    fn generator(&self) -> &'static str {
        generator_name(self.inner.generator())
    }

    fn __repr__(&self) -> String {
        format!(
            "ContinuousWalk(nodes={}, generator={:?}, gamma={})",
            self.inner.num_nodes(),
            generator_name(self.inner.generator()),
            self.inner.gamma()
        )
    }
}

/// Childs–Goldstone spatial search: start in the uniform superposition and
/// evolve under `γL − Σ_w |w⟩⟨w|` (or `−γA − Σ_w |w⟩⟨w|`).
#[pyclass(name = "SpatialSearch", module = "metatron_qso", frozen)]
pub struct PySpatialSearch {
    inner: SpatialSearch,
}

#[pymethods]
impl PySpatialSearch {
    /// `gamma=None` selects `SpatialSearch.critical_gamma(graph)`.
    #[new]
    #[pyo3(signature = (graph, marked, generator = "laplacian", gamma = None))]
    fn new(
        py: Python<'_>,
        graph: &PyGraph,
        marked: Indices,
        generator: &str,
        gamma: Option<f64>,
    ) -> PyResult<Self> {
        let gen_ = self::generator(generator)?;
        let marked = marked.into_vec();
        let g = &graph.inner;
        let inner = py
            .detach(|| {
                let gamma = match gamma {
                    Some(x) => x,
                    None => SpatialSearch::critical_gamma(g)?,
                };
                SpatialSearch::new(g, &marked, gen_, gamma)
            })
            .raise()?;
        Ok(Self { inner })
    }

    /// Childs–Goldstone critical hopping rate `(1/N) Σ_{λ>0} 1/λ` over the
    /// Laplacian spectrum.
    #[staticmethod]
    fn critical_gamma(py: Python<'_>, graph: &PyGraph) -> PyResult<f64> {
        let g = &graph.inner;
        py.detach(|| SpatialSearch::critical_gamma(g)).raise()
    }

    /// Optimise the hopping rate (around the critical rate) and the
    /// measurement time together. `t_max=None` uses `π√N`. On complete
    /// graphs this recovers the exact optimum γ = 1/N with success 1.
    #[staticmethod]
    #[pyo3(signature = (graph, marked, generator = "laplacian", t_max = None, resolution = 400))]
    fn optimize<'py>(
        py: Python<'py>,
        graph: &PyGraph,
        marked: Indices,
        generator: &str,
        t_max: Option<f64>,
        resolution: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let gen_ = self::generator(generator)?;
        let marked = marked.into_vec();
        let g = &graph.inner;
        let t_max = t_max.unwrap_or(std::f64::consts::PI * (g.num_nodes() as f64).sqrt());
        if !(t_max.is_finite() && t_max > 0.0) {
            return Err(invalid("`t_max` must be positive and finite"));
        }
        let r = py
            .detach(|| SpatialSearch::optimize(g, &marked, gen_, t_max, resolution.max(2)))
            .raise()?;
        Ok(pydict!(py, {
            "gamma" => r.gamma,
            "time" => r.time,
            "success_probability" => r.success_probability,
            "expected_time" => r.expected_time,
        }))
    }

    /// Probability of measuring a marked node at time `t`.
    fn success_probability(&self, t: f64) -> PyResult<f64> {
        check_time("t", t)?;
        Ok(self.inner.success_probability(t))
    }

    /// Maximise the success probability over `[0, t_max]` (grid scan plus
    /// golden-section refinement).
    #[pyo3(signature = (t_max, resolution = 400))]
    fn optimize_time<'py>(
        &self,
        py: Python<'py>,
        t_max: f64,
        resolution: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        if !(t_max.is_finite() && t_max > 0.0) {
            return Err(invalid("`t_max` must be positive and finite"));
        }
        let s = &self.inner;
        let r = py.detach(|| s.optimize_time(t_max, resolution));
        Ok(pydict!(py, {
            "gamma" => r.gamma,
            "time" => r.time,
            "success_probability" => r.success_probability,
            "expected_time" => r.expected_time,
        }))
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    #[getter]
    fn marked(&self) -> Vec<usize> {
        self.inner.marked().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "SpatialSearch(marked={:?}, gamma={})",
            self.inner.marked(),
            self.inner.gamma()
        )
    }
}

/// Quantum walk with site dephasing at rate `rate` (Lindblad evolution of
/// the density matrix; graphs with up to 512 nodes).
#[pyclass(name = "DephasingWalk", module = "metatron_qso", frozen)]
pub struct PyDephasingWalk {
    inner: DephasingWalk,
    n: usize,
}

#[pymethods]
impl PyDephasingWalk {
    #[new]
    #[pyo3(signature = (graph, generator = "laplacian", gamma = 1.0, rate = 0.0))]
    fn new(graph: &PyGraph, generator: &str, gamma: f64, rate: f64) -> PyResult<Self> {
        let inner =
            DephasingWalk::new(&graph.inner, self::generator(generator)?, gamma, rate).raise()?;
        Ok(Self {
            inner,
            n: graph.inner.num_nodes(),
        })
    }

    /// Node populations at time `t ≥ 0` for a walker started at `start`.
    fn populations<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let w = &self.inner;
        Ok(real_array(
            py,
            py.detach(|| w.populations(start, t)).raise()?,
        ))
    }

    /// Evolve the density matrix `rho0` (`n × n`) for time `t ≥ 0`.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        rho0: ComplexMatrixArg<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray2<metatron_qso::C64>>> {
        let (dim, data) = square_complex("rho0", &rho0)?;
        if dim != self.n {
            return Err(invalid(format!(
                "rho0 must be {n}x{n}, got {dim}x{dim}",
                n = self.n
            )));
        }
        let rho0 = metatron_qso::linalg::CMatrix::from_row_slice(dim, dim, &data);
        let w = &self.inner;
        let rho = py.detach(|| w.evolve(&rho0, t)).raise()?;
        Ok(cmatrix(py, &rho))
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.n
    }
}

/// Classical continuous-time random walk `p(t) = exp(−r·t·L)·e_start`.
#[pyfunction(name = "classical_walk_probabilities")]
#[pyo3(signature = (graph, start, t, rate = 1.0))]
fn classical_walk_probabilities_py<'py>(
    py: Python<'py>,
    graph: &PyGraph,
    start: usize,
    t: f64,
    rate: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    check_time("t", t)?;
    check_time("rate", rate)?;
    let g = &graph.inner;
    Ok(real_array(
        py,
        py.detach(|| classical_walk_probabilities(g, start, t, rate))
            .raise()?,
    ))
}

/// Expected hitting times of `target` for the discrete simple random walk
/// (`inf` for unreachable nodes).
#[pyfunction(name = "classical_hitting_times")]
fn classical_hitting_times_py<'py>(
    py: Python<'py>,
    graph: &PyGraph,
    target: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let g = &graph.inner;
    Ok(real_array(
        py,
        py.detach(|| classical_hitting_times(g, target)).raise()?,
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyContinuousWalk>()?;
    m.add_class::<PySpatialSearch>()?;
    m.add_class::<PyDephasingWalk>()?;
    m.add_function(wrap_pyfunction!(classical_walk_probabilities_py, m)?)?;
    m.add_function(wrap_pyfunction!(classical_hitting_times_py, m)?)?;
    Ok(())
}
