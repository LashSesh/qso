//! Continuous-time quantum walks, spatial search, dephasing and classical
//! baselines.

use metatron_qso::walks::driven::{DrivenWalk, MagnusOptions, Term};
use metatron_qso::walks::interacting::{InteractingWalk, Statistics, certificate_distance};
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
    ComplexArg, ComplexMatrixArg, Raise, RealArg, cmatrix, complex_array, complex_vec, invalid,
    pydict, real_array, real_vec, rmatrix, square_complex,
};

pub(crate) fn generator(name: &str) -> PyResult<Generator> {
    match name.to_ascii_lowercase().as_str() {
        "laplacian" => Ok(Generator::Laplacian),
        "adjacency" => Ok(Generator::Adjacency),
        _ => Err(invalid(format!(
            "unknown generator '{name}' (expected laplacian or adjacency)"
        ))),
    }
}

pub(crate) fn generator_name(g: Generator) -> &'static str {
    match g {
        Generator::Laplacian => "laplacian",
        Generator::Adjacency => "adjacency",
    }
}

pub(crate) fn check_time(name: &str, t: f64) -> PyResult<()> {
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
/// Krylov propagation, and limiting and averaged distributions through the
/// start node's cyclic subspace (small for symmetric graphs); the dense
/// propagator needs at most 2048 nodes.
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

    /// Dimension of the smallest invariant subspace containing the start
    /// node: the number of distinct eigenvalues the walker sees.
    fn cyclic_dimension(&self, py: Python<'_>, start: usize) -> PyResult<usize> {
        let w = &self.inner;
        py.detach(|| w.cyclic_dimension(start)).raise()
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

/// One term of a driven walk: `("laplacian", graph)`, `("adjacency",
/// graph)` (minus the adjacency matrix), `("projector", nodes)`,
/// `("potential", values)` or `("uniform", None)` (the projector onto the
/// uniform superposition).
fn term(n: usize, spec: &Bound<'_, PyAny>) -> PyResult<Term> {
    let (kind, arg): (String, Bound<'_, PyAny>) = spec
        .extract()
        .map_err(|_| invalid("each term is a pair (kind, argument)"))?;
    match kind.to_ascii_lowercase().as_str() {
        "laplacian" | "adjacency" => {
            let g = arg
                .cast::<PyGraph>()
                .map_err(|_| invalid(format!("a {kind} term takes a Graph")))?;
            Ok(Term::graph(&g.borrow().inner, generator(&kind)?))
        }
        "projector" => {
            let nodes: Indices = arg.extract()?;
            Term::projector(n, &nodes.into_vec()).raise()
        }
        "potential" => {
            let values: Vec<f64> = arg.extract()?;
            Term::potential(&values).raise()
        }
        "uniform" => Ok(Term::Uniform),
        _ => Err(invalid(format!(
            "unknown term '{kind}' (expected laplacian, adjacency, projector, potential or uniform)"
        ))),
    }
}

/// A continuous-time walk with a time-dependent Hamiltonian
/// `H(t) = Σ_j c_j(t)·H_j` on `n_nodes` nodes; see `term` kinds in
/// `terms`. `evolve` integrates it with the fourth-order commutator-free
/// Magnus integrator and step-size control.
#[pyclass(name = "DrivenWalk", module = "metatron_qso", frozen)]
pub struct PyDrivenWalk {
    inner: DrivenWalk,
}

#[pymethods]
impl PyDrivenWalk {
    #[new]
    fn new(n_nodes: usize, terms: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        let terms = terms
            .iter()
            .map(|t| term(n_nodes, t))
            .collect::<PyResult<Vec<_>>>()?;
        DrivenWalk::new(n_nodes, terms)
            .raise()
            .map(|inner| Self { inner })
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[getter]
    fn num_terms(&self) -> usize {
        self.inner.num_terms()
    }

    /// Evolve `psi0` from `t0` to `t1`; `coefficients(t)` returns the
    /// coefficients of the terms at time `t`. `tol` bounds the error of the
    /// final state; `fixed_step` switches step-size control off; `max_step`
    /// caps the step (the coefficients are sampled inside steps only:
    /// split evolutions at jumps, and resolve pulses with `max_step`).
    /// Holds the GIL, since it calls `coefficients`. Returns a dict with
    /// `state`, `steps`, `rejected`, `error_estimate` and `forced` (steps
    /// accepted at the minimum step size above the tolerance).
    #[pyo3(signature = (
        psi0, t0, t1, coefficients, tol = 1e-8, initial_step = None, fixed_step = None,
        max_step = None, max_steps = 1_000_000
    ))]
    #[allow(clippy::too_many_arguments)]
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        psi0: ComplexArg<'_>,
        t0: f64,
        t1: f64,
        coefficients: &Bound<'py, PyAny>,
        tol: f64,
        initial_step: Option<f64>,
        fixed_step: Option<f64>,
        max_step: Option<f64>,
        max_steps: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let options = MagnusOptions {
            tol,
            initial_step,
            fixed_step,
            max_step,
            max_steps,
        };
        let mut error: Option<PyErr> = None;
        let result = self.inner.evolve(
            &complex_vec(&psi0),
            t0,
            t1,
            |t| {
                coefficients
                    .call1((t,))
                    .and_then(|v| v.extract::<Vec<f64>>())
                    .map_err(|e| {
                        error = Some(e);
                        metatron_qso::QsoError::invalid(
                            "coefficients",
                            "the callable raised an exception",
                        )
                    })
            },
            &options,
        );
        if let Some(e) = error {
            return Err(e);
        }
        let r = result.raise()?;
        Ok(pydict!(py, {
            "state" => complex_array(py, r.state),
            "steps" => r.steps,
            "rejected" => r.rejected,
            "error_estimate" => r.error_estimate,
            "forced" => r.forced,
        }))
    }
}

fn statistics(name: &str) -> PyResult<Statistics> {
    match name.to_ascii_lowercase().as_str() {
        "bosons" => Ok(Statistics::Bosons),
        "fermions" => Ok(Statistics::Fermions),
        "distinguishable" => Ok(Statistics::Distinguishable),
        _ => Err(invalid(format!(
            "unknown statistics '{name}' (expected bosons, fermions or distinguishable)"
        ))),
    }
}

/// Walk of `particles` bosons, fermions or distinguishable particles on a
/// graph: hopping with `H = −γA` (`"adjacency"`, the default: the
/// Bose–Hubbard model) or `H = γL` (`"laplacian"`), on-site interaction
/// `on_site` (U) and interaction `neighbour` (V) between particles on
/// adjacent sites.
///
/// States are vectors over the configurations: multisets of sites for
/// bosons, sets for fermions, the site of each particle for
/// distinguishable particles. Up to 2048 configurations the walk is
/// diagonalised on first use (needed for `propagator`,
/// `transition_probabilities` and `certificate`); above, it propagates by
/// Krylov. Walks beyond 2^22 configurations, 2^25 stored sites or 2^25
/// nonzero Hamiltonian entries are refused.
#[pyclass(name = "InteractingWalk", module = "metatron_qso", frozen)]
pub struct PyInteractingWalk {
    inner: InteractingWalk,
}

#[pymethods]
impl PyInteractingWalk {
    #[new]
    #[pyo3(signature = (
        graph,
        particles,
        statistics = "bosons",
        generator = "adjacency",
        gamma = 1.0,
        on_site = 0.0,
        neighbour = 0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        graph: &PyGraph,
        particles: usize,
        statistics: &str,
        generator: &str,
        gamma: f64,
        on_site: f64,
        neighbour: f64,
    ) -> PyResult<Self> {
        let (s, gen_) = (self::statistics(statistics)?, self::generator(generator)?);
        let g = &graph.inner;
        let inner = py
            .detach(|| {
                InteractingWalk::new(g, gen_, gamma, s, particles)?
                    .with_interaction(on_site, neighbour)
            })
            .raise()?;
        Ok(Self { inner })
    }

    /// The sites of configuration `index`.
    fn configuration(&self, index: usize) -> PyResult<Vec<usize>> {
        Ok(self.inner.configuration(index).raise()?.to_vec())
    }

    /// Index of the configuration with particles on `sites` (any order
    /// for bosons and fermions).
    fn index_of(&self, sites: Vec<usize>) -> PyResult<usize> {
        self.inner.index_of(&sites).raise()
    }

    /// The basis state with particles on `sites`.
    fn basis_state<'py>(
        &self,
        py: Python<'py>,
        sites: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        Ok(complex_array(py, self.inner.basis_state(&sites).raise()?))
    }

    /// `exp(−iHt)|psi0⟩`.
    fn evolve<'py>(
        &self,
        py: Python<'py>,
        psi0: ComplexArg<'_>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<metatron_qso::C64>>> {
        let psi0 = complex_vec(&psi0);
        let w = &self.inner;
        Ok(complex_array(py, py.detach(|| w.evolve(&psi0, t)).raise()?))
    }

    /// Probabilities of all configurations at time `t`, started with
    /// particles on `sites`.
    fn probabilities<'py>(
        &self,
        py: Python<'py>,
        sites: Vec<usize>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let w = &self.inner;
        Ok(real_array(
            py,
            py.detach(|| w.probabilities(&sites, t)).raise()?,
        ))
    }

    /// Mean site occupations `⟨n_i⟩` in the state `psi`.
    fn densities<'py>(
        &self,
        py: Python<'py>,
        psi: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let psi = complex_vec(&psi);
        Ok(real_array(py, self.inner.densities(&psi).raise()?))
    }

    /// Density correlations `⟨n_i n_j⟩` in the state `psi`.
    fn pair_correlations<'py>(
        &self,
        py: Python<'py>,
        psi: ComplexArg<'_>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let psi = complex_vec(&psi);
        Ok(rmatrix(py, &self.inner.pair_correlations(&psi).raise()?))
    }

    /// Dense propagator `exp(−iHt)` over the configurations.
    fn propagator<'py>(
        &self,
        py: Python<'py>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray2<metatron_qso::C64>>> {
        let w = &self.inner;
        Ok(cmatrix(py, &py.detach(|| w.propagator(t)).raise()?))
    }

    /// `|⟨a|exp(−iHt)|b⟩|²` for all pairs of configurations.
    fn transition_probabilities<'py>(
        &self,
        py: Python<'py>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let w = &self.inner;
        Ok(rmatrix(
            py,
            &py.detach(|| w.transition_probabilities(t)).raise()?,
        ))
    }

    /// Graph certificate: the sorted transition probabilities at each of
    /// `times`, concatenated. Compare two with
    /// `interacting_certificate_distance` against `certificate_tolerance`.
    fn certificate<'py>(
        &self,
        py: Python<'py>,
        times: RealArg<'_>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let times = real_vec(&times);
        let w = &self.inner;
        Ok(real_array(py, py.detach(|| w.certificate(&times)).raise()?))
    }

    /// Tolerance for comparing certificates at `times`:
    /// `1e-12 · max(1, max|t| · ‖H‖)`. Rounding grows with the time;
    /// certificates of isomorphic graphs stay far below it, and a larger
    /// distance shows, up to floating-point error, that two graphs are not
    /// isomorphic.
    fn certificate_tolerance(&self, py: Python<'_>, times: RealArg<'_>) -> PyResult<f64> {
        let times = real_vec(&times);
        let w = &self.inner;
        py.detach(|| w.certificate_tolerance(&times)).raise()
    }

    /// Eigenvalues of the Hamiltonian (ascending), or `None` above 2048
    /// configurations.
    #[getter]
    fn spectrum<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.spectrum().map(|v| real_array(py, v.to_vec()))
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[getter]
    fn particles(&self) -> usize {
        self.inner.particles()
    }

    #[getter]
    fn statistics(&self) -> &'static str {
        self.inner.statistics().name()
    }

    #[getter]
    fn generator(&self) -> &'static str {
        generator_name(self.inner.generator())
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    #[getter]
    fn on_site(&self) -> f64 {
        self.inner.on_site()
    }

    #[getter]
    fn neighbour(&self) -> f64 {
        self.inner.neighbour()
    }

    fn __repr__(&self) -> String {
        format!(
            "InteractingWalk({} {}, {} configurations, U={}, V={})",
            self.inner.particles(),
            self.inner.statistics().name(),
            self.inner.dimension(),
            self.inner.on_site(),
            self.inner.neighbour()
        )
    }
}

/// Largest difference between two certificates of
/// `InteractingWalk.certificate`: infinite if their lengths differ, NaN if
/// either holds NaN. Above `InteractingWalk.certificate_tolerance` the
/// graphs are not isomorphic, up to floating-point error.
#[pyfunction(name = "interacting_certificate_distance")]
fn certificate_distance_py(a: RealArg<'_>, b: RealArg<'_>) -> f64 {
    certificate_distance(&real_vec(&a), &real_vec(&b))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyContinuousWalk>()?;
    m.add_class::<PyInteractingWalk>()?;
    m.add_function(wrap_pyfunction!(certificate_distance_py, m)?)?;
    m.add_class::<PySpatialSearch>()?;
    m.add_class::<PyDephasingWalk>()?;
    m.add_class::<PyDrivenWalk>()?;
    m.add_function(wrap_pyfunction!(classical_walk_probabilities_py, m)?)?;
    m.add_function(wrap_pyfunction!(classical_hitting_times_py, m)?)?;
    Ok(())
}
